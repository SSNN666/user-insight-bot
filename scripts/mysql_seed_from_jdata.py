"""一次性:把 JData 数据面灌入本机 MySQL,激活降级链 Tier-1。

做法:复用 pipeline.data_loader 自己的 JData→canonical orders 变换
(与 tianchi 层同一份代码) → 拆三表灌 MySQL ecommerce 库:
  orders  行为表 type=4(下单)事件(采样上限与 tianchi 层一致)
  users   订单涉及的用户的画像维(user_id/username/reg_date/city/age/gender)
  products 订单涉及的 SKU(product_id/product_name/category/price)

验证目标:Tier-1 与 tianchi 层数字一致(同一数据面两条管线),
拔掉 MySQL(停服务/改错密码)→ 自动降级 tianchi 的演示闭环。

用法: uv run python scripts/mysql_seed_from_jdata.py
"""
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pymysql

from config.settings import get_settings
from pipeline import data_loader as dl

# 密码只从 .env 的 DB_PASSWORD 读取(不入库;与仓库"密钥勿硬编码"哲学一致)
DB_PASSWORD = get_settings().DB_PASSWORD or os.environ.get("DB_PASSWORD", "")
if not DB_PASSWORD:
    raise SystemExit("请先在 .env 配置 DB_PASSWORD(本机 MySQL root 密码)再运行本脚本")


def build_dataframe() -> pd.DataFrame:
    """与 dl.load_tianchi_orders 同口径:读文件 → canonical 变换 → 时间线平移。

    不做滚动揭晓截断(reveal 默认关)、不并入运行时商城订单(MySQL Tier-1
    本身不合并——保持与真实 Tier-1 路径完全一致的行为)。
    """
    settings = get_settings()
    found = dl._find_jdata_files(settings.TIANCHI_DATA_DIR)
    if not (found["user"] and found["action"]):
        raise SystemExit(f"JData 文件缺失: {settings.TIANCHI_DATA_DIR}")

    t0 = time.monotonic()
    print(f"[1/4] 读 JData CSVs (actions ≤ {settings.TIANCHI_MAX_ACTIONS})...")
    users = dl._read_jdata_csvs(settings.TIANCHI_DATA_DIR, found["user"])
    actions = dl._read_jdata_csvs(
        settings.TIANCHI_DATA_DIR, found["action"], nrows=settings.TIANCHI_MAX_ACTIONS)

    print(f"[2/4] canonical 变换 (users={len(users)}, actions={len(actions)})...")
    df = dl._build_orders_from_jdata(users, actions,
                                     max_users=settings.TIANCHI_MAX_USERS)
    if df is None or df.empty:
        raise SystemExit("变换结果为空——检查 JData 文件与采样上限")
    shift = dl.jdata_date_shift(df["order_date"])
    if shift != pd.Timedelta(0):
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce") + shift
        df["reg_date"] = pd.to_datetime(df["reg_date"], errors="coerce") + shift
    df = df.dropna(subset=["order_date"])
    print(f"       orders={len(df)}, users={df['user_id'].nunique()}, "
          f"products={df['product_id'].nunique()}, 耗时 {time.monotonic()-t0:.0f}s")
    return df


def split_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    orders = df[["order_id", "user_id", "product_id", "quantity",
                 "total_amount", "order_date"]].copy()
    orders["total_amount"] = pd.to_numeric(orders["total_amount"], errors="coerce")

    users = df[["user_id", "reg_date", "city", "age", "gender"]].copy()
    users = users.drop_duplicates("user_id", keep="last")
    users["username"] = "user_" + users["user_id"].astype(str)

    products = df[["product_id", "product_name", "category", "price"]].copy()
    products = products.drop_duplicates("product_id", keep="last")
    return orders, users, products


def _dto(v):
    """datetime 兼容值 → pymysql 可写(naT → None)。"""
    return v.to_pydatetime() if pd.notna(v) else None


def seed_mysql(orders, users, products) -> None:
    settings = get_settings()
    conn = pymysql.connect(host=settings.DB_HOST, port=settings.DB_PORT,
                           user=settings.DB_USER, password=DB_PASSWORD,
                           database=settings.DB_NAME, charset="utf8mb4",
                           connect_timeout=10)
    print(f"[3/4] 已连接 MySQL {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME},"
          f" 重建三表(覆盖现有数据)...")
    with conn.cursor() as cur:
        cur.execute("SET FOREIGN_KEY_CHECKS=0")
        for t in ("orders", "users", "products"):
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute("""
            CREATE TABLE orders (
                order_id BIGINT PRIMARY KEY,
                user_id BIGINT NOT NULL,
                product_id BIGINT NOT NULL,
                quantity INT NOT NULL DEFAULT 1,
                total_amount DOUBLE NOT NULL,
                order_date DATETIME NOT NULL,
                INDEX idx_user (user_id), INDEX idx_date (order_date)
            ) ENGINE=InnoDB""")
        cur.execute("""
            CREATE TABLE users (
                user_id BIGINT PRIMARY KEY,
                username VARCHAR(64) NOT NULL DEFAULT '',
                reg_date DATETIME NULL,
                city VARCHAR(32) NOT NULL DEFAULT '未知',
                age VARCHAR(32) NOT NULL DEFAULT '未知',
                gender VARCHAR(8) NOT NULL DEFAULT '保密'
            ) ENGINE=InnoDB""")
        cur.execute("""
            CREATE TABLE products (
                product_id BIGINT PRIMARY KEY,
                product_name VARCHAR(128) NOT NULL,
                category VARCHAR(32) NOT NULL DEFAULT '未知',
                price DOUBLE NOT NULL
            ) ENGINE=InnoDB""")

        def bulk(table: str, rows: list[tuple], sql: str):
            B = 5000
            for i in range(0, len(rows), B):
                cur.executemany(sql, rows[i:i + B])
            print(f"       {table}: {len(rows)} 行")

        bulk("users", [tuple(r) for r in users[["user_id", "username", "reg_date",
                                                "city", "age", "gender"]]
             .itertuples(index=False, name=None)],
             "INSERT INTO users (user_id, username, reg_date, city, age, gender) "
             "VALUES (%s,%s,%s,%s,%s,%s)")
        bulk("products", [tuple(r) for r in products[["product_id", "product_name",
                                                      "category", "price"]]
             .itertuples(index=False, name=None)],
             "INSERT INTO products (product_id, product_name, category, price) "
             "VALUES (%s,%s,%s,%s)")
        # reg_date 为 Timestamp → to_pydatetime
        order_rows = []
        for r in orders.itertuples(index=False, name=None):
            oid, uid, pid, qty, amt, odate = r
            order_rows.append((int(oid), int(uid), int(pid), int(qty),
                               float(amt), _dto(odate)))
        bulk("orders", order_rows,
             "INSERT INTO orders (order_id, user_id, product_id, quantity, "
             "total_amount, order_date) VALUES (%s,%s,%s,%s,%s,%s)")
    conn.commit()
    conn.close()


def main() -> None:
    df = build_dataframe()
    orders, users, products = split_tables(df)
    seed_mysql(orders, users, products)
    print(f"[4/4] 完成: orders={len(orders)}, users={len(users)}, "
          f"products={len(products)} → 覆盖 ecommerce 库")


if __name__ == "__main__":
    main()
