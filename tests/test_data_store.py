"""Unit tests for api/data_store.py — CRUD, indexes, thread safety."""

import threading
import pytest
from api.data_store import (
    get_product, add_product, delete_product, list_products,
    get_user, update_user, list_users,
    add_order, list_orders,
    get_cart, add_to_cart, remove_from_cart, clear_cart,
    checkout, create_order_from_items, pay_order,
    reset_all,
)


# ══════════════════════════════════════════════════════════════════
# Products
# ══════════════════════════════════════════════════════════════════

class TestProducts:
    def test_add_and_get(self):
        p = add_product("测试商品", "电子产品", 99.0)
        assert p["product_id"] > 0
        assert p["product_name"] == "测试商品"

        found = get_product(p["product_id"])
        assert found is not None
        assert found["price"] == 99.0

    def test_get_nonexistent(self):
        assert get_product(999999) is None

    def test_delete(self):
        p = add_product("待删除", "测试", 1.0)
        assert delete_product(p["product_id"]) is True
        assert get_product(p["product_id"]) is None

    def test_delete_nonexistent(self):
        assert delete_product(999999) is False

    def test_list_all(self):
        reset_all()
        p1 = add_product("A", "cat1", 10)
        p2 = add_product("B", "cat2", 20)
        all_p = list_products()
        ids = {p["product_id"] for p in all_p}
        assert p1["product_id"] in ids
        assert p2["product_id"] in ids


# ══════════════════════════════════════════════════════════════════
# Users
# ══════════════════════════════════════════════════════════════════

class TestUsers:
    def test_get_existing(self):
        reset_all()
        users = list_users()
        assert len(users) > 0
        uid = users[0]["user_id"]
        u = get_user(uid)
        assert u is not None
        assert u["user_id"] == uid

    def test_get_nonexistent(self):
        assert get_user(999999) is None

    def test_update(self):
        reset_all()
        uid = list_users()[0]["user_id"]
        result = update_user(uid, city="深圳", age=30)
        assert result is not None
        assert result["city"] == "深圳"
        assert result["age"] == 30

    def test_update_nonexistent(self):
        assert update_user(999999, city="X") is None


# ══════════════════════════════════════════════════════════════════
# Orders
# ══════════════════════════════════════════════════════════════════

class TestOrders:
    def test_add_and_list(self):
        reset_all()
        o = add_order(1, 1, 2, 198.0)
        assert o["order_id"] > 0
        assert o["status"] == "已确认"

        orders = list_orders(user_id=1)
        assert len(orders) >= 1
        assert orders[-1]["order_id"] == o["order_id"]

    def test_list_by_user(self):
        reset_all()
        add_order(10, 1, 1, 50)
        add_order(20, 1, 1, 50)
        assert len(list_orders(user_id=10)) >= 1
        assert len(list_orders(user_id=20)) >= 1

    def test_pay_order(self):
        reset_all()
        o = add_order(1, 1, 1, 100)
        result = pay_order(o["order_id"])
        assert result is not None
        assert result["status"] == "已支付"

    def test_pay_nonexistent(self):
        assert pay_order(999999) is None


# ══════════════════════════════════════════════════════════════════
# Cart + Checkout
# ══════════════════════════════════════════════════════════════════

class TestCart:
    def test_empty_cart(self):
        c = get_cart(9999)
        assert c["item_count"] == 0
        assert c["total"] == 0

    def test_add_to_cart(self):
        reset_all()
        c = add_to_cart(1, 1, quantity=2)
        assert c["item_count"] >= 1

    def test_remove_from_cart(self):
        reset_all()
        add_to_cart(1, 1, quantity=1)
        c = remove_from_cart(1, 1)
        assert c["item_count"] == 0

    def test_clear_cart(self):
        reset_all()
        add_to_cart(1, 1, quantity=1)
        clear_cart(1)
        c = get_cart(1)
        assert c["item_count"] == 0

    def test_checkout_creates_order(self):
        reset_all()
        add_to_cart(1, 1, quantity=1)
        order = checkout(1)
        assert order["order_id"] > 0
        assert order["total_amount"] > 0
        # Cart should be cleared
        assert get_cart(1)["item_count"] == 0

    def test_checkout_empty_cart_raises(self):
        with pytest.raises(ValueError, match="购物车为空"):
            checkout(9999)

    def test_create_order_from_items(self):
        reset_all()
        items = [{"product_id": 1, "product_name": "测试", "price": 10, "quantity": 2}]
        order = create_order_from_items(5, items)
        assert order["total_amount"] == 20
        assert order["user_id"] == 5

    def test_create_order_from_empty_items_raises(self):
        with pytest.raises(ValueError, match="购物车为空"):
            create_order_from_items(1, [])


# ══════════════════════════════════════════════════════════════════
# Reset
# ══════════════════════════════════════════════════════════════════

class TestReset:
    def test_reset_re_seeds(self):
        info = reset_all()
        assert info["users"] > 0
        assert info["products"] > 0

    def test_reset_clears_orders(self):
        add_order(1, 1, 1, 100)
        reset_all()
        assert len(list_orders()) == 0


# ══════════════════════════════════════════════════════════════════
# Thread Safety
# ══════════════════════════════════════════════════════════════════

class TestThreadSafety:
    def test_concurrent_add_products(self):
        """10 threads adding products simultaneously — no dupes, no crashes."""
        reset_all()
        results = []
        errors = []

        def worker(i):
            try:
                p = add_product(f"thread-{i}", "test", float(i))
                results.append(p["product_id"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # All IDs should be unique
        assert len(results) == len(set(results))

    def test_concurrent_add_orders(self):
        """10 threads adding orders — IDs should be unique."""
        reset_all()
        oids = []
        errors = []

        def worker(u):
            try:
                o = add_order(u, 1, 1, 10.0)
                oids.append(o["order_id"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(oids) == len(set(oids))


# ══════════════════════════════════════════════════════════════════
# 播种数据源(懒播种:JData 模式 → JData 数据面,与分群分析统一)
# ══════════════════════════════════════════════════════════════════

class TestSeedDataSource:
    def _jdata_df(self):
        import pandas as pd
        return pd.DataFrame({
            "order_id": [1, 2],
            "user_id": [200001, 200002],
            "product_id": [9001, 9002],
            "product_name": ["SKU-9001", "SKU-9002"],
            "category": ["8", "8"],
            "price": [100.0, 200.0],
            "unit_price": [100.0, 200.0],
            "quantity": [1, 1],
            "total_amount": [100.0, 200.0],
            "order_date": pd.to_datetime(["2016-02-01", "2016-02-02"]),
            "reg_date": pd.to_datetime(["2016-01-01", "2016-01-02"]),
            "age": ["26-35岁", "36-45岁"],
            "gender": ["男", "女"],
            "city": ["未知", "未知"],
        })

    def test_tianchi_mode_seeds_from_jdata(self, monkeypatch):
        """DATA_SOURCE=tianchi → 用户/商品来自 JData(与分群同数据面)。"""
        from config.settings import get_settings
        from api import data_store as ds
        monkeypatch.setenv("DATA_SOURCE", "tianchi")
        get_settings.cache_clear()
        monkeypatch.setattr(
            "pipeline.data_loader.load_orders_with_join", lambda: self._jdata_df())
        ds.reset_all()
        try:
            users = ds.list_users()
            products = ds.list_products()
            assert {u["user_id"] for u in users} == {200001, 200002}
            assert {p["product_id"] for p in products} == {9001, 9002}
            assert users[0]["city"] == "未知"        # JData 无城市字段
        finally:
            # 显式恢复 mock 种子,防污染其他测试(不依赖 fixture 清理顺序)
            monkeypatch.setenv("DATA_SOURCE", "auto")
            get_settings.cache_clear()
            ds.reset_all()

    def test_jdata_load_failure_falls_back_mock(self, monkeypatch):
        """JData 加载失败 → 回退 mock,不阻塞服务。"""
        from config.settings import get_settings
        from api import data_store as ds
        monkeypatch.setenv("DATA_SOURCE", "tianchi")
        get_settings.cache_clear()

        def _boom():
            raise RuntimeError("jdata unavailable")
        monkeypatch.setattr("pipeline.data_loader.load_orders_with_join", _boom)
        ds.reset_all()
        try:
            assert len(ds.list_users()) > 0          # mock 兜底
        finally:
            monkeypatch.setenv("DATA_SOURCE", "auto")
            get_settings.cache_clear()
            ds.reset_all()

    def test_default_mock_seed_still_works(self):
        """默认(auto)模式行为不变:mock 种子。"""
        from api import data_store as ds
        ds.reset_all()
        assert len(ds.list_users()) > 0
        assert len(ds.list_products()) > 0
        ds.reset_all()   # 幂等
