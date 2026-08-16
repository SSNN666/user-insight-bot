"""Data loading with pagination and three-tier fallback.

Tier 1: MySQL paginated query
Tier 2: Local TTL file cache
Tier 3: Built-in mock dataset

Phase 3: incremental loading + expanded mock data with product/category.
"""

import hashlib
import os

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from config.settings import get_settings
from errors.exceptions import DatabaseError, ComputationError
from cache.ttl_cache import TTLCache
from log.logger import get_logger

logger = get_logger(__name__)
_cache = TTLCache(namespace="orders")

CACHE_KEY_JOINED = "orders_joined"
CACHE_KEY_RFM = "rfm_from_db"


def get_db_engine():
    """Create SQLAlchemy engine for MySQL."""
    settings = get_settings()
    conn_str = (
        f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}"
        f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        f"?charset={settings.DB_CHARSET}"
    )
    return create_engine(conn_str)


# ── Tier 1: MySQL ───────────────────────────────────────────────


def _try_mysql_joined(since_date: str | None = None) -> pd.DataFrame:
    """Tier 1: Paginated MySQL query with JOIN, optionally since a date."""
    settings = get_settings()
    engine = get_db_engine()
    page_size = settings.DATA_PAGE_SIZE

    where_clause = ""
    params: dict = {"limit": page_size, "offset": 0}
    if since_date:
        where_clause = "WHERE o.order_date > :since_date"
        params["since_date"] = since_date

    try:
        with engine.connect() as conn:
            count_query = text(f"SELECT COUNT(*) FROM orders o {where_clause}")
            total = conn.execute(count_query, params).scalar()
            logger.info("mysql_count", extra={"total_orders": total, "since": since_date})

        dfs: list[pd.DataFrame] = []
        for offset in range(0, total, page_size):
            params["offset"] = offset
            query_str = f"""
                SELECT
                    o.order_id, o.user_id, u.username, u.reg_date,
                    u.city, u.age, u.gender,
                    o.product_id, p.product_name, p.category,
                    p.price AS unit_price,
                    o.quantity, o.total_amount, o.order_date
                FROM orders o
                JOIN users u ON o.user_id = u.user_id
                JOIN products p ON o.product_id = p.product_id
                {where_clause}
                ORDER BY o.order_date
                LIMIT :limit OFFSET :offset
            """
            chunk = pd.read_sql(text(query_str), engine, params=params)
            dfs.append(chunk)
            logger.debug("page_loaded", extra={"offset": offset, "rows": len(chunk)})

        engine.dispose()
        result = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if not since_date and len(result) > 0:
            try:
                _cache.set(CACHE_KEY_JOINED, result)
            except Exception:
                logger.warning("cache_write_failed")
        logger.info("tier1_success", extra={"rows": len(result)})
        return result

    except (SQLAlchemyError, OSError) as e:
        engine.dispose()
        raise DatabaseError(f"MySQL query failed: {e}", context={"tier": 1}) from e


# ── Tier 2: TTL cache ───────────────────────────────────────────


def _try_cache_joined() -> pd.DataFrame | None:
    """Tier 2: Read from local TTL file cache.

    Returns None on cache miss so the caller can fall through to Tier 3
    without treating a cache miss as an error.
    """
    data = _cache.get(CACHE_KEY_JOINED)
    if data is None:
        logger.debug("cache_miss")
        return None
    logger.info("tier2_success", extra={"rows": len(data)})
    return data


# ── Tier 2.5: 京东 JData 公开脱敏数据集(可选) ──────────────────
#
# 「京东 JData 算法大赛——高潜用户购买意向预测」数据集(官方发布在 DataFountain
# 竞赛平台,学术研究许可、已脱敏;下载渠道见 docs/DATA_COMPLIANCE.md):
#   JData_User.csv    user_id(脱敏), age, sex, user_lv_cd, user_reg_tm
#   JData_Product.csv sku_id, a1, a2, a3, cate, brand
#   JData_Action.csv  user_id, sku_id, time, model_id, type, cate, brand(2016-02~04 分月)
#   JData_Comment.csv dt, sku_id, comment_num, has_bad_comment, bad_comment_rate
#
# 映射口径(如实记录,面试可讲):
#   - type=4(下单)行为行 → 一条订单明细(quantity=1);F=下单次数,M=Σ价格
#   - JData 不含价格字段 → 按 (cate, brand) 做 md5 确定性合成价格(50-5000 元,
#     同一商品跨运行稳定,快照对比不受影响)
#   - 分类/品牌为数字编码,不做中文映射(推荐 Skill 品类不匹配时自动回退热门商品)
#   - Action 表的 user_id/sku_id 是浮点("1.0"),加载时归一化为 int 才能与 User 表 join

JDATA_ACTION_TYPE_ORDER = 4      # 官方行为编码:1=浏览 2=加购 3=删除 4=下单 5=关注 6=点击
JDATA_SEX_MAP = {0: "男", 1: "女", 2: "保密"}


def _synth_price(cate, brand) -> float:
    """JData 无价格字段:(品类,品牌) → md5 → 50-5000 元确定性合成。"""
    h = int(hashlib.md5(f"{cate}|{brand}".encode("utf-8")).hexdigest()[:8], 16)
    return round(50 + (h % 4950), 2)


def _find_jdata_files(data_dir: str) -> dict[str, list[str]]:
    """按文件名关键字匹配 JData 四个表(支持 JData_Action_201602.csv 分月格式)。"""
    found: dict[str, list[str]] = {"user": [], "product": [], "action": [], "comment": []}
    if not os.path.isdir(data_dir):
        return found
    for fn in sorted(os.listdir(data_dir)):
        if not fn.lower().endswith(".csv"):
            continue
        low = fn.lower()
        if "user" in low:
            found["user"].append(fn)
        elif "product" in low or "sku" in low:
            found["product"].append(fn)
        elif "action" in low:
            found["action"].append(fn)
        elif "comment" in low:
            found["comment"].append(fn)
    return found


def _read_jdata_csvs(data_dir: str, names: list[str], nrows: int | None = None) -> pd.DataFrame:
    """拼接同一表的分月 CSV;任一文件读取失败抛 ComputationError(整档跳过)。"""
    frames = []
    for fn in names:
        # on_bad_lines="skip":官方数据存在个别字段数不一致的行(6 列 vs 7 列)
        # 编码:官方文件混用 utf-8 与 GBK(实测 JData_User.csv 为 GBK)→ 依次尝试
        path = os.path.join(data_dir, fn)
        try:
            df = pd.read_csv(path, nrows=nrows, encoding="utf-8-sig",
                             low_memory=False, on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(path, nrows=nrows, encoding="gbk",
                             low_memory=False, on_bad_lines="skip")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_orders_from_jdata(
    users: pd.DataFrame,
    actions: pd.DataFrame,
    max_users: int | None = None,
    action_type_order: int = JDATA_ACTION_TYPE_ORDER,
) -> pd.DataFrame | None:
    """JData 原始表 → canonical orders DataFrame(纯函数,便于单测)。

    Returns None when no order actions exist.
    """
    if users.empty or actions.empty or "type" not in actions.columns:
        return None

    orders = actions[actions["type"] == action_type_order].copy()
    if orders.empty:
        return None

    # 官方 Action 表 user_id/sku_id 为浮点("1.0"),归一化为 int 才能与 User 表 join
    for col in ("user_id", "sku_id"):
        orders[col] = pd.to_numeric(orders[col], errors="coerce")
        orders = orders[orders[col].notna()].copy()
        orders[col] = orders[col].astype(int)

    # 按时间排序后分配自增 order_id(同一用户的时间顺序即订单顺序)
    orders = orders.sort_values("time").reset_index(drop=True)
    orders["order_id"] = orders.index + 1

    # 用户采样:按出现顺序取前 N 个去重用户(确定性)
    if max_users and orders["user_id"].nunique() > max_users:
        keep = orders["user_id"].drop_duplicates().head(max_users)
        orders = orders[orders["user_id"].isin(keep)]

    # 价格:JData 无价格字段 → 确定性合成;一行下单行为 = 一条订单明细
    price_col = orders.apply(lambda r: _synth_price(r.get("cate"), r.get("brand")), axis=1)
    orders["unit_price"] = price_col
    orders["price"] = price_col
    orders["quantity"] = 1
    orders["total_amount"] = price_col
    orders["order_date"] = pd.to_datetime(orders["time"], errors="coerce")
    orders["product_id"] = orders["sku_id"]
    orders["product_name"] = "SKU-" + orders["sku_id"].astype(str)
    orders["category"] = orders["cate"].astype(str) if "cate" in orders.columns else "未知"

    # 用户维:age / 性别(sex 0/1/2)/ 注册时间(user_reg_tm,空值保留 NaT)
    u = users[["user_id", "age"]].copy()
    u["user_id"] = pd.to_numeric(u["user_id"], errors="coerce")
    u = u[u["user_id"].notna()]
    u["user_id"] = u["user_id"].astype(int)
    if "sex" in users.columns:
        # 官方数据 sex 读入为 float(2.0/0.0/1.0),显式归一化为 int 再查字典
        sex = pd.to_numeric(users["sex"], errors="coerce").astype("Int64")
        u["gender"] = sex.map(JDATA_SEX_MAP).fillna("保密")
    else:
        u["gender"] = "保密"
    u["reg_date"] = pd.to_datetime(
        users["user_reg_tm"] if "user_reg_tm" in users.columns else pd.NaT,
        errors="coerce",
    )
    orders = orders.merge(u, on="user_id", how="left")
    # 官方口径:age 是分桶字符串("26-35岁"/"56岁以上"),-1 = 未知(不是数值!)
    orders["age"] = orders["age"].fillna("未知").astype(str).replace("-1", "未知")
    orders["city"] = "未知"

    keep_cols = ["order_id", "user_id", "product_id", "product_name", "category",
                 "price", "unit_price", "quantity", "total_amount", "order_date",
                 "reg_date", "age", "gender", "city"]
    return orders[[c for c in keep_cols if c in orders.columns]]


def load_tianchi_orders(data_dir: str | None = None, since_date: str | None = None) -> pd.DataFrame | None:
    """加载京东 JData CSV 并映射为 canonical orders。

    Returns None when files missing (caller falls through to mock tier).
    """
    settings = get_settings()
    data_dir = data_dir or settings.TIANCHI_DATA_DIR
    found = _find_jdata_files(data_dir)
    if not (found["user"] and found["action"]):
        logger.debug("tianchi_files_missing", extra={"dir": data_dir})
        return None

    try:
        max_actions = settings.TIANCHI_MAX_ACTIONS
        users = _read_jdata_csvs(data_dir, found["user"])
        actions = _read_jdata_csvs(data_dir, found["action"], nrows=max_actions)
        df = _build_orders_from_jdata(users, actions, max_users=settings.TIANCHI_MAX_USERS)
        if df is None:
            return None
        if since_date:
            df = df[pd.to_datetime(df["order_date"], errors="coerce") > pd.Timestamp(since_date)]
        logger.info("tier_tianchi_success", extra={
            "dir": data_dir, "rows": len(df), "users": df["user_id"].nunique(),
        })
        return df
    except (OSError, ValueError, pd.errors.ParserError) as e:
        logger.warning("tianchi_load_failed", extra={"dir": data_dir, "error": str(e)})
        return None


def _try_tianchi(since_date: str | None = None) -> pd.DataFrame | None:
    """Tier 2.5: 京东 JData 数据源(仅 DATA_SOURCE=tianchi 时进降级链)。"""
    return load_tianchi_orders(since_date=since_date)


# ── Tier 3: mock data ───────────────────────────────────────────


def generate_mock_orders() -> pd.DataFrame:
    """Generate synthetic orders for development / demo.

    Phase 3: includes product_id, category, and unit_price for
    extended profile computation.
    """
    # ── Random seed ──
    # MOCK_SEED 固定时数据可复现(默认 42):Watcher 事件反映真实业务变化而非随机噪声;
    # 设为 None 时每次生成不同数据(演示随机性用)。
    settings = get_settings()
    np.random.seed(settings.MOCK_SEED if settings.MOCK_SEED is not None else None)

    n_users = np.random.randint(120, 350)
    n_orders = np.random.randint(600, 2000)

    # Products with categories
    products = pd.DataFrame({
        'product_id': range(1, 21),
        'product_name': [
            '笔记本电脑', '鼠标', '图书', '耳机', '键盘',
            '显示器', '台灯', '办公椅', '笔记本', '钢笔套装',
            '背包', '水杯', 'U盘', '摄像头', '音箱',
            '平板电脑', '手机壳', '充电器', '运动鞋', 'T恤',
        ],
        'category': [
            '电子产品', '电子产品', '教育用品', '电子产品', '电子产品',
            '电子产品', '家具家居', '家具家居', '教育用品', '教育用品',
            '时尚服饰', '生活用品', '电子产品', '电子产品', '电子产品',
            '电子产品', '电子产品', '电子产品', '时尚服饰', '时尚服饰',
        ],
        'price': [
            5999, 99, 59, 299, 199,
            1299, 89, 899, 29, 49,
            159, 39, 79, 249, 399,
            2599, 49, 69, 329, 79,
        ],
    })

    users = pd.DataFrame({
        'user_id': range(1, n_users + 1),
        'reg_date': pd.date_range('2020-01-01', periods=n_users, freq='D'),
        'age': np.random.randint(18, 65, n_users),
        'city': np.random.choice(
            ['北京', '上海', '广州', '深圳', '杭州', '成都'], n_users
        ),
    })

    # Orders with realistic inter-order intervals per user
    orders = pd.DataFrame({
        'order_id': range(1, n_orders + 1),
        'user_id': np.random.choice(users['user_id'], n_orders),
        'order_date': pd.date_range('2021-01-01', periods=n_orders, freq=f'{np.random.randint(2,12)}h'),
        'total_amount': np.random.uniform(30, 5000, n_orders),
        'quantity': np.random.randint(1, 5, n_orders),
    })

    # Assign random products
    orders['product_id'] = np.random.choice(products['product_id'], n_orders)

    # Merge products and users
    result = orders.merge(products, on='product_id').merge(users, on='user_id')
    return result


def _try_mock_data() -> pd.DataFrame:
    """Tier 3: Built-in mock dataset, enriched with shared-store orders.

    Merges orders from the in-memory data store (created via debug console
    or Vue ecommerce) into the mock dataset so charts reflect real user activity.
    """
    logger.info("tier3_fallback")
    base = generate_mock_orders()

    # ── Merge shared-store orders so charts reflect actual activity ──
    try:
        from api.data_store import list_orders as ds_orders
        store_orders = ds_orders()
        if store_orders:
            extra_rows = []
            for o in store_orders:
                extra_rows.append({
                    "order_id": o.get("order_id", 0),
                    "user_id": o.get("user_id", 1),
                    "product_id": o.get("product_id", o.get("items", [{}])[0].get("product_id", 1) if o.get("items") else 1),
                    "product_name": o.get("product_name", "未知"),
                    "category": "电子产品",
                    "price": o.get("total_amount", 99),
                    "order_date": pd.Timestamp(o.get("created_at", pd.Timestamp.now())),
                    "total_amount": o.get("total_amount", 99),
                    "quantity": o.get("quantity", 1),
                    "unit_price": o.get("total_amount", 99),
                    "reg_date": pd.Timestamp.now() - pd.Timedelta(days=365),
                    "age": 30,
                    "city": "北京",
                })
            if extra_rows:
                extra_df = pd.DataFrame(extra_rows)
                base = pd.concat([base, extra_df], ignore_index=True)
                logger.info("mock_enriched_with_store_orders", extra={
                    "store_orders": len(store_orders),
                    "total_rows": len(base),
                })
    except Exception as e:
        logger.debug("store_merge_skipped", extra={"error": str(e)})

    return base


# ── Public API ──────────────────────────────────────────────────


def load_orders_with_join() -> pd.DataFrame:
    """降级链:MySQL → TTL 缓存 → (京东 JData,可选) → mock。"""
    settings = get_settings()
    tiers = [
        ("MySQL", _try_mysql_joined),
        ("cache", _try_cache_joined),
    ]
    if settings.DATA_SOURCE == "tianchi":
        tiers.append(("tianchi", _try_tianchi))
    tiers.append(("mock", _try_mock_data))
    for name, loader in tiers:
        try:
            result = loader()
            if result is None:
                # Tier 2 can return None on cache miss → skip to next tier
                logger.debug("tier_skipped", extra={"tier": name})
                continue
            return result
        except (DatabaseError, ComputationError) as e:
            logger.warning("tier_failed", extra={"tier": name, "error": str(e)})
            if name == "mock":
                raise
    raise DatabaseError("All three data tiers failed")


def load_new_orders_since(since_date: str) -> pd.DataFrame:
    """Load only orders newer than ``since_date`` (ISO format).

    Three-tier fallback; Tier 2 cache is bypassed for delta loads.
    """
    settings = get_settings()
    tiers = [
        ("MySQL", lambda: _try_mysql_joined(since_date=since_date)),
    ]
    if settings.DATA_SOURCE == "tianchi":
        tiers.append(("tianchi", lambda: _try_tianchi(since_date=since_date)))
    tiers.append(("mock", _try_mock_data))
    for name, loader in tiers:
        try:
            df = loader()
            logger.info("delta_loaded", extra={"since": since_date, "rows": len(df)})
            return df
        except (DatabaseError, ComputationError) as e:
            logger.warning("tier_failed", extra={"tier": name, "error": str(e)})
            if name == "mock":
                raise
    raise DatabaseError("All data tiers failed for delta load")


def load_rfm_from_db():
    """Compute RFM directly in MySQL via GROUP BY (server-side aggregation)."""
    engine = get_db_engine()
    try:
        query = text("""
            SELECT
                user_id,
                MAX(order_date) AS last_order_date,
                DATEDIFF(CURDATE(), MAX(order_date)) AS recency,
                COUNT(DISTINCT order_id) AS frequency,
                SUM(total_amount) AS monetary
            FROM orders
            GROUP BY user_id
        """)
        df = pd.read_sql(query, engine)
        engine.dispose()
        return df
    except (SQLAlchemyError, OSError) as e:
        engine.dispose()
        raise DatabaseError(
            f"RFM SQL query failed: {e}", context={"query": "load_rfm_from_db"}
        ) from e
