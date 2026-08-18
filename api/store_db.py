"""data_store 的 SQLite 持久化层(写穿透,重启恢复)。

内存 dict 保持为主(读零成本),所有变更经本模块同步落库:
  users / products / orders / cart_items / store_meta(seeded 标志)

连接模式仿 watcher/task_manager.py:每操作新建连接 + WAL(简单、线程安全);
写失败不抛给调用方(调用方防御性 try/except),保证写穿透不阻塞主链路。
"""

import json
import os
import sqlite3

from config.settings import get_settings


def store_path() -> str:
    """库文件路径(CACHE_DIR/store.db,conftest 已重定向,测试自动隔离)。"""
    return os.path.join(get_settings().CACHE_DIR, "store.db")


def _conn() -> sqlite3.Connection:
    path = store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """建表(幂等)。"""
    with _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS store_meta (
                key   TEXT PRIMARY KEY,
                value TEXT
            );
            CREATE TABLE IF NOT EXISTS users (
                user_id     INTEGER PRIMARY KEY,
                age,                  -- 无类型声明:保留存储类(age 可能是 int(mock)或 str(JData))
                city,                 -- TEXT 亲和会把 int 转文本,必须裸列
                preferences TEXT
            );
            CREATE TABLE IF NOT EXISTS products (
                product_id   INTEGER PRIMARY KEY,
                product_name TEXT NOT NULL,
                category     TEXT,
                price        REAL
            );
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                user_id  INTEGER NOT NULL,
                data     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cart_items (
                user_id    INTEGER NOT NULL,
                product_id INTEGER NOT NULL,
                quantity   INTEGER NOT NULL,
                PRIMARY KEY (user_id, product_id)
            );
        """)
        conn.commit()


# ── Meta ─────────────────────────────────────────────────────────


def save_meta(key: str, value: str) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO store_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()


def load_meta(key: str) -> str | None:
    with _conn() as conn:
        row = conn.execute(
            "SELECT value FROM store_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None


# ── Users ────────────────────────────────────────────────────────


def load_users() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM users").fetchall()
    out = []
    for r in rows:
        u = {"user_id": r["user_id"], "age": r["age"], "city": r["city"]}
        if r["preferences"]:
            try:
                u["preferences"] = json.loads(r["preferences"])
            except json.JSONDecodeError:
                pass
        out.append(u)
    return out


def upsert_user(user: dict) -> None:
    """保留原始类型(age 可能是 int(mock)或 str(JData 分桶),SQLite 动态类型)。"""
    with _conn() as conn:
        conn.execute(
            "INSERT INTO users(user_id, age, city, preferences) VALUES(?, ?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET "
            "age = excluded.age, city = excluded.city, preferences = excluded.preferences",
            (user["user_id"], user.get("age", ""), user.get("city", ""),
             json.dumps(user.get("preferences", {}), ensure_ascii=False)),
        )
        conn.commit()


def save_users_all(users: dict[int, dict]) -> None:
    """全量重写(播种/重置用,单次事务批量写入)。"""
    with _conn() as conn:
        conn.execute("DELETE FROM users")
        conn.executemany(
            "INSERT INTO users(user_id, age, city, preferences) VALUES(?, ?, ?, ?)",
            [(u["user_id"], u.get("age", ""), u.get("city", ""),
              json.dumps(u.get("preferences", {}), ensure_ascii=False))
             for u in users.values()],
        )
        conn.commit()


# ── Products ─────────────────────────────────────────────────────


def load_products() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT * FROM products").fetchall()
    return [dict(r) for r in rows]


def upsert_product(product: dict) -> None:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO products(product_id, product_name, category, price) "
            "VALUES(?, ?, ?, ?) "
            "ON CONFLICT(product_id) DO UPDATE SET "
            "product_name = excluded.product_name, category = excluded.category, "
            "price = excluded.price",
            (product["product_id"], product["product_name"],
             product.get("category", ""), float(product.get("price", 0))),
        )
        conn.commit()


def save_products_all(products: dict[int, dict]) -> None:
    """全量重写(播种/重置用,单次事务批量写入)。"""
    with _conn() as conn:
        conn.execute("DELETE FROM products")
        conn.executemany(
            "INSERT INTO products(product_id, product_name, category, price) "
            "VALUES(?, ?, ?, ?)",
            [(p["product_id"], p["product_name"], p.get("category", ""),
              float(p.get("price", 0))) for p in products.values()],
        )
        conn.commit()


def delete_product_row(product_id: int) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM products WHERE product_id = ?", (product_id,))
        conn.commit()


# ── Orders(全量重写,运行时订单量小)─────────────────────────────


def load_orders() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute("SELECT data FROM orders ORDER BY order_id").fetchall()
    out = []
    for r in rows:
        try:
            out.append(json.loads(r["data"]))
        except json.JSONDecodeError:
            continue
    return out


def save_orders(orders: list[dict]) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM orders")
        conn.executemany(
            "INSERT INTO orders(order_id, user_id, data) VALUES(?, ?, ?)",
            [(o["order_id"], o["user_id"], json.dumps(o, ensure_ascii=False))
             for o in orders],
        )
        conn.commit()


# ── Cart(全量重写)───────────────────────────────────────────────


def load_carts() -> dict[int, list[dict]]:
    """cart_items + products 重建完整 item(名称/价格)。"""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT c.user_id, c.product_id, c.quantity, p.product_name, p.price "
            "FROM cart_items c LEFT JOIN products p ON c.product_id = p.product_id"
        ).fetchall()
    carts: dict[int, list[dict]] = {}
    for r in rows:
        carts.setdefault(r["user_id"], []).append({
            "product_id": r["product_id"],
            "product_name": r["product_name"] or "未知商品",
            "price": r["price"] or 0,
            "quantity": r["quantity"],
        })
    return carts


def save_carts(carts: dict[int, list[dict]]) -> None:
    with _conn() as conn:
        conn.execute("DELETE FROM cart_items")
        conn.executemany(
            "INSERT INTO cart_items(user_id, product_id, quantity) VALUES(?, ?, ?)",
            [(uid, ci["product_id"], ci["quantity"])
             for uid, items in carts.items() for ci in items],
        )
        conn.commit()


# ── 全量 / 清空 ─────────────────────────────────────────────────


def clear_all() -> None:
    """清空全部表(重置/重播种子前调用)。"""
    with _conn() as conn:
        for t in ("orders", "cart_items", "users", "products", "store_meta"):
            conn.execute(f"DELETE FROM {t}")
        conn.commit()


def has_data() -> bool:
    """库中是否存在已持久化的业务数据(用于启动恢复判定)。"""
    with _conn() as conn:
        users = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        products = conn.execute("SELECT COUNT(*) AS c FROM products").fetchone()["c"]
        seeded = conn.execute(
            "SELECT value FROM store_meta WHERE key = 'seeded'"
        ).fetchone()
    return seeded is not None and seeded["value"] == "1" and users > 0 and products > 0
