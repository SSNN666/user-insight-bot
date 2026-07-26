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

# ── Seed from mock data ──────────────────────────────────────────

_df = generate_mock_orders()

_users_by_id: dict[int, dict] = {}
_products_by_id: dict[int, dict] = {}
_carts: dict[int, list[dict]] = {}     # user_id → [{product_id, product_name, price, quantity}]
_orders: list[dict] = []                # all orders (ordered)
_orders_by_id: dict[int, dict] = {}     # order_id → order (O(1) lookup)
_orders_by_user: dict[int, list[dict]] = {}  # user_id → list[order] (O(1) filter)
_next_order_id: int = 10000
_next_product_id: int = 1000


def _seed_from_df() -> None:
    """(Re-)build indexes from the mock DataFrame."""
    global _users_by_id, _products_by_id
    _users_by_id = {}
    for u in _df[["user_id", "age", "city"]].drop_duplicates().to_dict(orient="records"):
        _users_by_id[u["user_id"]] = u
    _products_by_id = {}
    for p in _df[["product_id", "product_name", "category", "price"]].drop_duplicates().to_dict(orient="records"):
        _products_by_id[p["product_id"]] = p


_seed_from_df()


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
    return _products_by_id.get(pid)


def add_product(name: str, category: str, price: float) -> dict:
    with _lock:
        pid = _next_pid()
        product = {
            "product_id": pid,
            "product_name": name,
            "category": category,
            "price": price,
        }
        _products_by_id[pid] = product
    return product


def delete_product(pid: int) -> bool:
    with _lock:
        if pid in _products_by_id:
            del _products_by_id[pid]
            return True
        return False


def list_products() -> list[dict]:
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
    return order


def list_orders(user_id: int | None = None) -> list[dict]:
    if user_id is not None:
        return _orders_by_user.get(user_id, [])
    return _orders


# ── User helpers ─────────────────────────────────────────────────

def get_user(uid: int) -> dict | None:
    return _users_by_id.get(uid)


def update_user(uid: int, city: str | None = None, age: int | None = None) -> dict | None:
    u = _users_by_id.get(uid)
    if u is None:
        return None
    if city is not None:
        u["city"] = city
    if age is not None:
        u["age"] = age
    return u


def list_users() -> list[dict]:
    return list(_users_by_id.values())


# ── Cart helpers ─────────────────────────────────────────────────

def get_cart(user_id: int) -> dict:
    with _lock:
        items = list(_carts.get(user_id, []))
    total = sum(i["price"] * i["quantity"] for i in items)
    return {
        "user_id": user_id,
        "items": items,
        "item_count": len(items),
        "total": round(total, 2),
    }


def add_to_cart(user_id: int, product_id: int, quantity: int = 1) -> dict:
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
    return get_cart(user_id)


def remove_from_cart(user_id: int, product_id: int) -> dict:
    with _lock:
        if user_id in _carts:
            _carts[user_id] = [
                i for i in _carts[user_id] if i["product_id"] != product_id
            ]
    return get_cart(user_id)


def clear_cart(user_id: int) -> None:
    with _lock:
        _carts[user_id] = []


# ── Checkout ─────────────────────────────────────────────────────

def checkout(user_id: int) -> dict:
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
    return order


# ── Direct order creation (for ecommerce API with specific items) ─


def create_order_from_items(user_id: int, items: list[dict]) -> dict:
    """Create an order from pre-built item list. Used by ecommerce API.

    Each item dict must have: product_id, product_name, price, quantity.
    """
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
    return order


# ── Payment ──────────────────────────────────────────────────────

def pay_order(order_id: int) -> dict | None:
    order = _orders_by_id.get(order_id)
    if order is not None:
        order["status"] = "已支付"
    return order


# ── Reset ────────────────────────────────────────────────────────

def reset_all() -> dict:
    """Re-seed all stores from mock data and clear carts/orders."""
    global _df, _carts, _orders, _next_order_id, _next_product_id
    with _lock:
        _df = generate_mock_orders()
        _seed_from_df()
        _carts = {}
        _orders = []
        _rebuild_order_indexes()
        _next_order_id = 10000
        _next_product_id = 1000
    return {
        "users": len(_users_by_id),
        "products": len(_products_by_id),
    }
