"""Shared in-memory data store — used by both ecommerce API and debug console.

All CRUD operations from either the Vue frontend or Gradio debug panel
read/write the same store, so changes are immediately visible everywhere.

Thread-safe: all mutable-state operations are protected by a reentrant lock
since the watcher engine (ThreadPoolExecutor) and API (asyncio event loop)
may access the store concurrently.

Design note: reads (get_*, list_*) return direct references for simplicity.
Callers MUST NOT mutate returned objects. If mutation is needed, use the
dedicated mutation helpers which acquire the lock.
"""

import threading
import pandas as pd

from pipeline.data_loader import generate_mock_orders

_lock = threading.RLock()

# ── Seed(懒播种:首次访问时按数据源模式建索引)──────────────────

_df: pd.DataFrame | None = None          # 播种数据源(纯 mock 或 JData 降级链)
_seeded: bool = False                    # 首次访问时播种,避免 import 即全量加载

_users_by_id: dict[int, dict] = {}
_products_by_id: dict[int, dict] = {}
_carts: dict[int, list[dict]] = {}     # user_id → [{product_id, product_name, price, quantity}]
_orders: list[dict] = []                # all orders (ordered)
_orders_by_id: dict[int, dict] = {}     # order_id → order (O(1) lookup)
_orders_by_user: dict[int, list[dict]] = {}  # user_id → list[order] (O(1) filter)
_next_order_id: int = 10000
_next_product_id: int = 1000


def _seed_from_df() -> None:
    """(Re-)build indexes from the seed DataFrame."""
    global _users_by_id, _products_by_id
    _users_by_id = {}
    for u in _df[["user_id", "age", "city"]].drop_duplicates().to_dict(orient="records"):
        _users_by_id[u["user_id"]] = u
    _products_by_id = {}
    for p in _df[["product_id", "product_name", "category", "price"]].drop_duplicates().to_dict(orient="records"):
        _products_by_id[p["product_id"]] = p


def _reseed() -> None:
    """按数据源模式重播种子(调用方需持锁):
    DATA_SOURCE == "tianchi" → JData 真实数据(与分群分析同数据面);
    否则 → mock 合成数据。JData 加载失败回退 mock。
    """
    global _df, _next_order_id, _next_product_id
    from config.settings import get_settings
    settings = get_settings()
    if settings.DATA_SOURCE == "tianchi":
        try:
            from pipeline.data_loader import load_orders_with_join
            _df = load_orders_with_join()
        except Exception:
            from log.logger import get_logger
            get_logger(__name__).warning("seed_tianchi_failed_fallback_mock")
            _df = generate_mock_orders()
    else:
        _df = generate_mock_orders()
    _seed_from_df()
    _next_order_id = 10000
    _next_product_id = 1000


def _ensure_seeded() -> None:
    """懒播种:首次访问时按数据源模式建用户/商品索引(线程安全)。

    优先从 SQLite 恢复上次会话数据(重启不丢:用户修改/新增商品/运行时
    订单/购物车);库为空才重新播种并全量落库。
    """
    global _seeded
    if _seeded:
        return
    with _lock:
        if _seeded:
            return
        if _load_from_db():
            _seeded = True
            return
        _reseed()
        _persist_all()
        _seeded = True


def _load_from_db() -> bool:
    """从 SQLite 恢复内存态(用户/商品/订单/购物车/ID 计数器)。

    Returns:
        True = 恢复成功(库中有已持久化数据);False = 库为空,走播种。
    """
    global _users_by_id, _products_by_id, _carts, _orders, \
        _orders_by_id, _orders_by_user, _next_order_id, _next_product_id
    from api import store_db
    try:
        if not store_db.has_data():
            return False
        _users_by_id = {u["user_id"]: u for u in store_db.load_users()}
        _products_by_id = {p["product_id"]: p for p in store_db.load_products()}
        _carts = store_db.load_carts()
        _orders = store_db.load_orders()
        _rebuild_order_indexes()
        # ID 计数器恢复:取 (种子初始值, 表内 max+1) 较大者,防 delete 后退档冲突
        _next_order_id = max(10000, max((o["order_id"] for o in _orders), default=0) + 1)
        _next_product_id = max(1000, max(_products_by_id.keys(), default=0) + 1)
        from log.logger import get_logger
        get_logger(__name__).info("store_restored_from_db", extra={
            "users": len(_users_by_id), "products": len(_products_by_id),
            "orders": len(_orders),
        })
        return True
    except Exception:
        from log.logger import get_logger
        get_logger(__name__).warning("store_db_load_failed")
        return False


def _persist_all() -> None:
    """全量落库(播种/重置后调用):用户/商品/订单/购物车 + seeded 标志。"""
    from api import store_db
    try:
        store_db.init_db()
        store_db.save_users_all(_users_by_id)
        store_db.save_products_all(_products_by_id)
        store_db.save_orders(_orders)
        store_db.save_carts(_carts)
        store_db.save_meta("seeded", "1")
    except Exception:
        from log.logger import get_logger
        get_logger(__name__).warning("store_db_persist_failed")


def _persist_orders() -> None:
    """订单变更后写穿透(订单量小,全量重写)。"""
    from api import store_db
    try:
        store_db.save_orders(_orders)
    except Exception:
        from log.logger import get_logger
        get_logger(__name__).warning("store_db_orders_save_failed")


def _rebuild_order_indexes() -> None:
    """Rebuild order lookup indexes after mutations."""
    global _orders_by_id, _orders_by_user
    _orders_by_id = {o["order_id"]: o for o in _orders}
    _orders_by_user = {}
    for o in _orders:
        uid = o["user_id"]
        if uid not in _orders_by_user:
            _orders_by_user[uid] = []
        _orders_by_user[uid].append(o)


# ── ID generators (must be called under lock) ────────────────────

def _next_oid() -> int:
    global _next_order_id
    oid = _next_order_id
    _next_order_id += 1
    return oid


def _next_pid() -> int:
    global _next_product_id
    pid = _next_product_id
    _next_product_id += 1
    return pid


# ── Product helpers ──────────────────────────────────────────────

def get_product(pid: int) -> dict | None:
    _ensure_seeded()
    return _products_by_id.get(pid)


def add_product(name: str, category: str, price: float) -> dict:
    _ensure_seeded()
    with _lock:
        pid = _next_pid()
        product = {
            "product_id": pid,
            "product_name": name,
            "category": category,
            "price": price,
        }
        _products_by_id[pid] = product
    try:
        from api import store_db
        store_db.upsert_product(product)
    except Exception:
        from log.logger import get_logger
        get_logger(__name__).warning("store_db_product_save_failed")
    return product


def delete_product(pid: int) -> bool:
    _ensure_seeded()
    with _lock:
        if pid in _products_by_id:
            del _products_by_id[pid]
            try:
                from api import store_db
                store_db.delete_product_row(pid)
            except Exception:
                pass
            return True
        return False


def list_products() -> list[dict]:
    _ensure_seeded()
    return list(_products_by_id.values())


# ── Order helpers ────────────────────────────────────────────────

def _add_order_index(order: dict) -> None:
    """Update order indexes with a new order (caller must hold _lock)."""
    _orders_by_id[order["order_id"]] = order
    uid = order["user_id"]
    if uid not in _orders_by_user:
        _orders_by_user[uid] = []
    _orders_by_user[uid].append(order)


def add_order(user_id: int, product_id: int, quantity: int, total_amount: float) -> dict:
    product = get_product(product_id)
    with _lock:
        oid = _next_oid()
        order = {
            "order_id": oid,
            "user_id": user_id,
            "product_id": product_id,
            "product_name": product["product_name"] if product else "未知商品",
            "quantity": quantity,
            "total_amount": total_amount,
            "status": "已确认",
            "created_at": pd.Timestamp.now().isoformat(),
        }
        _orders.append(order)
        _add_order_index(order)
    _persist_orders()
    return order


def list_orders(user_id: int | None = None) -> list[dict]:
    _ensure_seeded()
    # 返回副本,防调用方原地修改污染共享索引(watcher/API 线程池并发读写)
    if user_id is not None:
        return list(_orders_by_user.get(user_id, []))
    return list(_orders)


# ── User helpers ─────────────────────────────────────────────────

def get_user(uid: int) -> dict | None:
    _ensure_seeded()
    return _users_by_id.get(uid)


def update_user(uid: int, city: str | None = None, age: int | None = None) -> dict | None:
    _ensure_seeded()
    with _lock:
        u = _users_by_id.get(uid)
        if u is None:
            return None
        if city is not None:
            u["city"] = city
        if age is not None:
            u["age"] = age
    try:
        from api import store_db
        store_db.upsert_user(u)
    except Exception:
        from log.logger import get_logger
        get_logger(__name__).warning("store_db_user_save_failed")
    return u


def add_user_preference(uid: int, tags: dict) -> dict | None:
    """把商品图像解析标签并入用户偏好(支撑个性化推荐)。

    tags: {"category": str, "appearance": str, "tags": [str], ...}
    列表类字段按去重追加,标量字段覆盖。
    """
    _ensure_seeded()
    u = _users_by_id.get(uid)
    if u is None:
        return None
    with _lock:
        prefs = u.setdefault("preferences", {})
        for k, v in (tags or {}).items():
            if isinstance(v, list):
                cur = prefs.setdefault(k, [])
                for item in v:
                    if item not in cur:
                        cur.append(item)
            else:
                prefs[k] = v
    try:
        from api import store_db
        store_db.upsert_user(u)
    except Exception:
        pass
    return u


def list_users() -> list[dict]:
    _ensure_seeded()
    return list(_users_by_id.values())


# ── Cart helpers ─────────────────────────────────────────────────

def get_cart(user_id: int) -> dict:
    _ensure_seeded()
    with _lock:
        items = list(_carts.get(user_id, []))
    total = sum(i["price"] * i["quantity"] for i in items)
    return {
        "user_id": user_id,
        "items": items,
        "item_count": len(items),
        "total": round(total, 2),
    }


def _persist_carts() -> None:
    """购物车变更后写穿透(全量重写)。"""
    from api import store_db
    try:
        store_db.save_carts(_carts)
    except Exception:
        from log.logger import get_logger
        get_logger(__name__).warning("store_db_carts_save_failed")


def add_to_cart(user_id: int, product_id: int, quantity: int = 1) -> dict:
    _ensure_seeded()
    p = get_product(product_id)
    if not p:
        return get_cart(user_id)

    with _lock:
        if user_id not in _carts:
            _carts[user_id] = []

        for ci in _carts[user_id]:
            if ci["product_id"] == product_id:
                ci["quantity"] += quantity
                break
        else:
            _carts[user_id].append({
                "product_id": product_id,
                "product_name": p["product_name"],
                "price": p["price"],
                "quantity": quantity,
            })
    _persist_carts()
    return get_cart(user_id)


def remove_from_cart(user_id: int, product_id: int) -> dict:
    _ensure_seeded()
    with _lock:
        if user_id in _carts:
            _carts[user_id] = [
                i for i in _carts[user_id] if i["product_id"] != product_id
            ]
    _persist_carts()
    return get_cart(user_id)


def clear_cart(user_id: int) -> None:
    _ensure_seeded()
    with _lock:
        _carts[user_id] = []
    _persist_carts()


# ── Checkout ─────────────────────────────────────────────────────

def checkout(user_id: int) -> dict:
    _ensure_seeded()
    with _lock:
        items = _carts.get(user_id, [])
        if not items:
            raise ValueError("购物车为空")

        total = sum(i["price"] * i["quantity"] for i in items)
        oid = _next_oid()

        order_items = [
            {
                "product_id": i["product_id"],
                "product_name": i["product_name"],
                "price": i["price"],
                "quantity": i["quantity"],
            }
            for i in items
        ]
        order = {
            "order_id": oid,
            "user_id": user_id,
            "items": order_items,
            "total_amount": round(total, 2),
            "status": "已确认",
            "created_at": pd.Timestamp.now().isoformat(),
        }
        _orders.append(order)
        _add_order_index(order)
        _carts[user_id] = []
    _persist_orders()
    _persist_carts()
    return order


# ── Direct order creation (for ecommerce API with specific items) ─


def create_order_from_items(user_id: int, items: list[dict]) -> dict:
    """Create an order from pre-built item list. Used by ecommerce API.

    Each item dict must have: product_id, product_name, price, quantity.
    """
    _ensure_seeded()
    if not items:
        raise ValueError("购物车为空")
    total = sum(i["price"] * i["quantity"] for i in items)
    with _lock:
        oid = _next_oid()
        order = {
            "order_id": oid,
            "user_id": user_id,
            "items": items,
            "total_amount": round(total, 2),
            "status": "已确认",
            "created_at": pd.Timestamp.now().isoformat(),
        }
        _orders.append(order)
        _add_order_index(order)
        _carts.pop(user_id, None)  # clear cart after order
    _persist_orders()
    _persist_carts()
    return order


# ── Payment ──────────────────────────────────────────────────────

def pay_order(order_id: int) -> dict | None:
    _ensure_seeded()
    with _lock:
        order = _orders_by_id.get(order_id)
        if order is not None:
            order["status"] = "已支付"
            _persist_orders()
    return order


# ── Reset ────────────────────────────────────────────────────────

def reset_all() -> dict:
    """清空持久化 → 按数据源模式重播种子(与懒播种同口径)并清空 carts/orders。"""
    global _carts, _orders, _seeded
    from api import store_db
    try:
        store_db.clear_all()
    except Exception:
        pass
    with _lock:
        _reseed()
        _carts = {}
        _orders = []
        _rebuild_order_indexes()
        _seeded = True
        _persist_all()
    return {
        "users": len(_users_by_id),
        "products": len(_products_by_id),
    }
