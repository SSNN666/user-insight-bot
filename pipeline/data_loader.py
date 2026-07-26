"""Data loading with pagination and three-tier fallback.

Tier 1: MySQL paginated query
Tier 2: Local TTL file cache
Tier 3: Built-in mock dataset

Phase 3: incremental loading + expanded mock data with product/category.
"""

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


# ── Tier 3: mock data ───────────────────────────────────────────


def generate_mock_orders() -> pd.DataFrame:
    """Generate synthetic orders for development / demo.

    Phase 3: includes product_id, category, and unit_price for
    extended profile computation.
    """
    # ── Random seed: different data every time ──
    np.random.seed(None)  # OS entropy — truly random each call

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
    """Three-tier fallback: MySQL → TTL cache → mock data."""
    tiers = [
        ("MySQL", _try_mysql_joined),
        ("cache", _try_cache_joined),
        ("mock", _try_mock_data),
    ]
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
    tiers = [
        ("MySQL", lambda: _try_mysql_joined(since_date=since_date)),
        ("mock", _try_mock_data),
    ]
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
