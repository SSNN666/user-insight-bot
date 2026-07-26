"""E-commerce REST API — products, cart, orders, user center.

Phase 10: powers the Vue frontend with full e-commerce flow.
All stores live in api.data_store so debug CRUD and ecommerce share the same data.

User identification:
    All cart/order endpoints accept ``X-User-ID`` header (defaults to 1 for demo).
    In production this should come from an auth middleware that sets
    ``request.state.user_id``.
"""

from typing import Annotated

from fastapi import APIRouter, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from log.logger import get_logger
from api.data_store import (
    get_product,
    list_products as ds_list_products,
    get_cart,
    add_to_cart as ds_add_to_cart,
    remove_from_cart as ds_remove_from_cart,
    clear_cart as ds_clear_cart,
    checkout as ds_checkout,
    create_order_from_items,
    list_orders as ds_list_orders,
    pay_order as ds_pay_order,
    get_user,
)

logger = get_logger(__name__, log_type="user_chat")
router = APIRouter(prefix="/api", tags=["ecommerce"])

# ── Helpers ─────────────────────────────────────────────────────


def _resolve_user_id(
    explicit: int | None = None,
    header: Annotated[str | None, Header(alias="X-User-ID")] = None,
) -> int:
    """Resolve user ID from explicit param → header → demo default."""
    if explicit is not None:
        return explicit
    if header is not None:
        try:
            return int(header)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail="X-User-ID 必须为整数")
    return 1  # demo default


# ── Pydantic models ────────────────────────────────────────────


class CartItemAdd(BaseModel):
    product_id: int
    quantity: int = Field(default=1, ge=1)


class OrderCreate(BaseModel):
    user_id: int
    cart_items: list[CartItemAdd] = Field(default_factory=list)


# ── Products ───────────────────────────────────────────────────


@router.get("/products")
async def list_products(
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
    category: Annotated[str | None, Query()] = None,
    q: Annotated[str | None, Query()] = None,
):
    """Paginated product listing with optional filter + search."""
    results = ds_list_products()

    if category:
        results = [p for p in results if p["category"] == category]
    if q:
        ql = q.lower()
        results = [p for p in results if ql in p["product_name"].lower()]

    total = len(results)
    start = (page - 1) * page_size
    end = start + page_size
    return {
        "products": results[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


@router.get("/products/categories")
async def list_categories():
    cats = sorted({p["category"] for p in ds_list_products()})
    return {"categories": cats}


@router.get("/products/{product_id}")
async def get_product_endpoint(product_id: int):
    p = get_product(product_id)
    if not p:
        raise HTTPException(status_code=404, detail="商品不存在")
    return p


# ── Cart ───────────────────────────────────────────────────────


@router.get("/cart")
async def view_cart(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
):
    uid = _resolve_user_id(header=x_user_id)
    return get_cart(uid)


@router.post("/cart/items")
async def add_to_cart(
    item: CartItemAdd,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
):
    uid = _resolve_user_id(header=x_user_id)
    p = get_product(item.product_id)
    if not p:
        raise HTTPException(status_code=404, detail="商品不存在")
    return ds_add_to_cart(uid, item.product_id, item.quantity)


@router.delete("/cart/items/{product_id}")
async def remove_from_cart(
    product_id: int,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
):
    uid = _resolve_user_id(header=x_user_id)
    return ds_remove_from_cart(uid, product_id)


@router.delete("/cart")
async def clear_cart(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
):
    uid = _resolve_user_id(header=x_user_id)
    ds_clear_cart(uid)
    return {"status": "已清空"}


# ── Orders ─────────────────────────────────────────────────────


@router.post("/orders")
async def create_order(
    req: OrderCreate | None = None,
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
):
    """Create an order from cart items."""
    uid = req.user_id if req else _resolve_user_id(header=x_user_id)

    if req and req.cart_items:
        items = []
        for ci in req.cart_items:
            p = get_product(ci.product_id)
            if p:
                items.append({
                    "product_id": ci.product_id,
                    "product_name": p["product_name"],
                    "price": p["price"],
                    "quantity": ci.quantity,
                })
        if not items:
            raise HTTPException(status_code=400, detail="购物车为空")
        try:
            order = create_order_from_items(uid, items)
            logger.info("order_created", extra={
                "order_id": order["order_id"], "user_id": uid,
                "total": order["total_amount"],
            })
            return order
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    try:
        order = ds_checkout(uid)
        logger.info("order_created", extra={
            "order_id": order["order_id"], "user_id": uid,
            "total": order["total_amount"],
        })
        return order
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/orders")
async def list_orders(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
):
    uid = _resolve_user_id(header=x_user_id)
    results = ds_list_orders(uid)
    total = len(results)
    start = (page - 1) * page_size
    return {
        "orders": results[start:start + page_size],
        "total": total, "page": page, "page_size": page_size,
        "total_pages": max(1, (total + page_size - 1) // page_size),
    }


@router.get("/orders/{order_id}")
async def get_order(order_id: int):
    for o in ds_list_orders():
        if o["order_id"] == order_id:
            return o
    raise HTTPException(status_code=404, detail="订单不存在")


@router.post("/orders/{order_id}/pay")
async def pay_order(order_id: int):
    result = ds_pay_order(order_id)
    if not result:
        raise HTTPException(status_code=404, detail="订单不存在")
    return {"status": "已支付", "order_id": order_id}


# ── User Center ────────────────────────────────────────────────


@router.get("/users/me")
async def user_profile(
    x_user_id: Annotated[str | None, Header(alias="X-User-ID")] = None,
):
    uid = _resolve_user_id(header=x_user_id)
    u = get_user(uid)
    if u is None:
        return {"user_id": uid, "age": 25, "city": "北京"}
    return u
