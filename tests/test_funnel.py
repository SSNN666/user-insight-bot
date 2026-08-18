"""转化漏斗:行为加载(JData / 订单派生)+ 漏斗计算 + Skill。

全部离线:JData 用合成 CSV 模拟(含官方浮点 ID 口径),不读真实 2.2GB 文件。
"""
import os

import pandas as pd
import pytest

import pipeline.funnel as pf
import skills.funnel as sf_mod
from skills.funnel import FunnelAnalysisSkill
from skills.base import SkillStatus
from config.settings import get_settings


def _actions_df() -> pd.DataFrame:
    """窗口 [2026-08-09, 2026-08-15]:浏览4人 加购2人 下单2人。"""
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 2, 3, 3, 4],
        "action_type": [1, 2, 1, 2, 4, 1, 4, 1],
        "date": pd.to_datetime([
            "2026-08-10 10:00", "2026-08-10 11:00", "2026-08-11 10:00",
            "2026-08-11 11:00", "2026-08-11 12:00", "2026-08-12 10:00",
            "2026-08-12 11:00", "2026-08-15 10:00",
        ]),
    })


def _orders_df(n: int = 100) -> pd.DataFrame:
    """n 个用户各 1 笔订单(同一日期)。"""
    return pd.DataFrame({
        "user_id": list(range(1, n + 1)),
        "order_date": pd.to_datetime(["2026-08-10"] * n),
    })


# ══════════════════════════════════════════════════════════════════
# compute_funnel
# ══════════════════════════════════════════════════════════════════

class TestComputeFunnel:
    def test_steps_and_conversion_rates(self):
        rows = pf.compute_funnel(_actions_df(), days=7)
        assert [r["step"] for r in rows] == ["浏览", "加购", "下单"]
        assert rows[0]["user_count"] == 4 and rows[0]["conversion_rate"] == 1.0
        assert rows[1]["user_count"] == 2 and rows[1]["conversion_rate"] == 0.5
        assert rows[2]["user_count"] == 2 and rows[2]["conversion_rate"] == 1.0

    def test_window_filters_outside_events(self):
        # days=1 → 窗口仅 2026-08-15 → 只有 1 人浏览
        rows = pf.compute_funnel(_actions_df(), days=1)
        assert rows[0]["user_count"] == 1
        assert rows[1]["user_count"] == 0
        assert rows[1]["conversion_rate"] == 0.0    # 上步为 0 → 0.0
        assert rows[2]["conversion_rate"] == 0.0

    def test_empty_window_returns_empty(self):
        assert pf.compute_funnel(None, 7) == []
        assert pf.compute_funnel(
            pd.DataFrame(columns=["user_id", "action_type", "date"]), 7
        ) == []

    def test_duplicate_users_deduped_per_step(self):
        actions = pd.DataFrame({
            "user_id": [1, 1, 1, 2, 2],
            "action_type": [1, 1, 4, 4, 1],
            "date": pd.to_datetime(["2026-08-10"] * 5),
        })
        rows = pf.compute_funnel(actions, 7)
        assert rows[0]["user_count"] == 2           # 用户1 浏览两次只算 1
        assert rows[2]["user_count"] == 2


# ══════════════════════════════════════════════════════════════════
# _derive_mock_actions(订单同源派生)
# ══════════════════════════════════════════════════════════════════

class TestDeriveMockActions:
    def test_deterministic_same_seed(self):
        a1 = pf._derive_mock_actions(_orders_df())
        a2 = pf._derive_mock_actions(_orders_df())
        assert a1.equals(a2)

    def test_one_order_action_per_order(self):
        orders = _orders_df()
        a = pf._derive_mock_actions(orders)
        assert (a["action_type"] == 4).sum() == len(orders)
        # 下单行为与订单同源(同 user / 同日期)
        merged = a[a["action_type"] == 4].merge(
            orders[["user_id", "order_date"]], on="user_id")
        assert (merged["date"].dt.date == merged["order_date"].dt.date).all()

    def test_action_types_limited_to_funnel(self):
        a = pf._derive_mock_actions(_orders_df())
        assert set(a["action_type"].unique()) <= {1, 2, 4}

    def test_funnel_gradient_on_large_sample(self):
        """虚拟流失用户保证 浏览 > 加购 > 下单 的漏斗梯度。"""
        a = pf._derive_mock_actions(_orders_df(500))
        v = a[a["action_type"] == 1]["user_id"].nunique()
        c = a[a["action_type"] == 2]["user_id"].nunique()
        o = a[a["action_type"] == 4]["user_id"].nunique()
        assert v > c > o

    def test_cart_happens_before_order(self):
        orders = _orders_df()
        a = pf._derive_mock_actions(orders)
        m = a[a["action_type"] == 2].merge(
            orders[["user_id", "order_date"]], on="user_id")
        assert (m["date"] < m["order_date"]).all()

    def test_empty_orders_returns_empty(self):
        empty = pd.DataFrame(columns=["user_id", "order_date"])
        assert pf._derive_mock_actions(empty).empty
        assert pf._derive_mock_actions(None).empty


# ══════════════════════════════════════════════════════════════════
# load_funnel_actions(JData 合成 CSV / 降级)
# ══════════════════════════════════════════════════════════════════

class TestLoadFunnelActions:
    def _write_action_csv(self, tmp_path) -> str:
        data_dir = os.path.join(str(tmp_path), "tianchi")
        os.makedirs(data_dir, exist_ok=True)
        path = os.path.join(data_dir, "JData_Action_201602.csv")
        # 官方口径:user_id 为浮点;type 1=浏览 2=加购 4=下单 6=点击(应被忽略)
        content = (
            "user_id,sku_id,time,model_id,type,cate,brand\n"
            "1.0,9001.0,2016-02-01 10:00:00,0,1,8,100\n"
            "2.0,9002.0,2016-02-01 11:00:00,0,2,8,100\n"
            "3.0,9003.0,2016-02-01 12:00:00,0,4,8,100\n"
            "4.0,9004.0,2016-02-01 13:00:00,0,6,8,100\n"
            "5.0,9005.0,2016-02-01 14:00:00,0,1,8,100\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return data_dir

    def test_tianchi_actions_keep_only_funnel_types(self, monkeypatch, tmp_path):
        data_dir = self._write_action_csv(tmp_path)
        monkeypatch.setenv("DATA_SOURCE", "tianchi")
        monkeypatch.setenv("TIANCHI_DATA_DIR", data_dir)
        get_settings.cache_clear()
        pf.invalidate_funnel_cache()
        try:
            actions = pf.load_funnel_actions(force_refresh=True)
        finally:
            pf.invalidate_funnel_cache()
            get_settings.cache_clear()
        assert len(actions) == 4                     # 点击行为被过滤
        assert set(actions["action_type"].unique()) == {1, 2, 4}
        assert pd.api.types.is_integer_dtype(actions["user_id"])   # 浮点归一 int
        assert pd.api.types.is_datetime64_dtype(actions["date"])

    def test_files_missing_falls_back_to_derived(self, monkeypatch, tmp_path):
        """DATA_SOURCE=tianchi 但 Action 文件缺失 → 回退订单派生(与降级链一致)。"""
        monkeypatch.setenv("DATA_SOURCE", "tianchi")
        monkeypatch.setenv("TIANCHI_DATA_DIR", str(tmp_path))   # 空目录
        get_settings.cache_clear()
        pf.invalidate_funnel_cache()
        try:
            actions = pf.load_funnel_actions(force_refresh=True)
        finally:
            pf.invalidate_funnel_cache()
            get_settings.cache_clear()
        assert not actions.empty
        assert set(actions["action_type"].unique()) <= {1, 2, 4}


# ══════════════════════════════════════════════════════════════════
# FunnelAnalysisSkill
# ══════════════════════════════════════════════════════════════════

class TestFunnelSkill:
    def test_success_with_data_and_analysis(self, monkeypatch):
        monkeypatch.setattr(
            sf_mod, "load_funnel_actions", lambda force_refresh=False: _actions_df())
        res = FunnelAnalysisSkill().execute(days=7)
        assert res.status == SkillStatus.SUCCESS
        assert [r["step"] for r in res.data] == ["浏览", "加购", "下单"]
        assert res.data[1]["conversion_rate"] == 0.5
        assert "最大流失环节" in res.summary
        assert "数据来源" in res.summary            # 数据新鲜度标注
        assert "去重用户数" in res.summary

    def test_partial_on_empty_window(self, monkeypatch):
        monkeypatch.setattr(
            sf_mod, "load_funnel_actions",
            lambda force_refresh=False: pd.DataFrame(columns=["user_id", "action_type", "date"]))
        res = FunnelAnalysisSkill().execute(days=7)
        assert res.status == SkillStatus.PARTIAL
        assert res.confidence < 0.5

    def test_days_parameter_passed(self, monkeypatch):
        calls = {}
        def fake_load(force_refresh=False):
            calls["days_check"] = True
            return _actions_df()
        monkeypatch.setattr(sf_mod, "load_funnel_actions", fake_load)
        # days 影响窗口:days=1 → 只有 1 人浏览
        res = FunnelAnalysisSkill().execute(days=1)
        assert res.status == SkillStatus.SUCCESS
        assert res.data[0]["user_count"] == 1
