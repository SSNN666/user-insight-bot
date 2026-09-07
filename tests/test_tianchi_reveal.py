"""滚动揭晓(tianchi reveal):状态机单测 + 加载集成 + 指纹 + 漏斗一致性。

全部离线:合成 10 天窗口的 JData Action CSV(每天 1 浏览 + 1 下单),
reveal 开关默认关闭 → 本文件不启用时行为与旧路径逐字节一致(回归盾)。
"""
import datetime
import os

import pandas as pd
import pytest

from pipeline import data_loader as dl


@pytest.fixture(autouse=True)
def _no_store_orders(monkeypatch):
    """隔离共享存储订单(同 test_tianchi_source.py:JData 加载会合并运行时订单)。"""
    monkeypatch.setattr("api.data_store._orders", [])
    yield


def _users_df() -> pd.DataFrame:
    return pd.DataFrame({
        "user_id": [101, 102],
        "age": ["26-35岁", "36-45岁"],
        "sex": [0, 1],
        "user_lv_cd": [1, 3],
        "user_reg_tm": ["2016-01-01", "2016-01-15"],
    })


def _write_daily_jdata(data_dir: str, days: int = 10) -> None:
    """写入 days 天窗口的合成 JData:每天 user101 浏览 1 次、user102 下单 1 次。"""
    os.makedirs(data_dir, exist_ok=True)
    start = datetime.date(2016, 2, 1)
    rows = []
    for i in range(days):
        d = start + datetime.timedelta(days=i)
        rows.append({"user_id": 101.0, "sku_id": 9001.0, "model_id": 0, "type": 1,
                     "cate": 8, "brand": 100, "time": f"{d} 10:00:00"})
        rows.append({"user_id": 102.0, "sku_id": 9002.0, "model_id": 0, "type": 4,
                     "cate": 8, "brand": 200, "time": f"{d} 11:00:00"})
    pd.DataFrame(rows).to_csv(os.path.join(data_dir, "JData_Action_201602.csv"),
                              index=False)
    _users_df().to_csv(os.path.join(data_dir, "JData_User.csv"), index=False)


def _enable_reveal(monkeypatch, tmp_path, days: int = 10, preheat: int = 3,
                   data_source: str = "tianchi") -> str:
    """打开 reveal(tianchi 档)并指向合成 CSV 目录,返回 data_dir。"""
    data_dir = str(tmp_path / "tianchi")
    _write_daily_jdata(data_dir, days=days)
    monkeypatch.setenv("DATA_SOURCE", data_source)
    monkeypatch.setenv("TIANCHI_DATA_DIR", data_dir)
    monkeypatch.setenv("TIANCHI_REVEAL_ENABLED", "true")
    monkeypatch.setenv("TIANCHI_REVEAL_PREHEAT_DAYS", str(preheat))
    monkeypatch.setenv("TIANCHI_REVEAL_STEP", "1")
    from config.settings import get_settings
    get_settings.cache_clear()
    return data_dir


def _state_path() -> str:
    from config.settings import get_settings
    return os.path.join(get_settings().CACHE_DIR, "tianchi_reveal.json")


def _backdate_last_advanced(days_back: int = 1) -> None:
    """把状态文件 last_advanced 改写为 N 天前(模拟"新的一天")。"""
    from pipeline import reveal_state
    st = reveal_state.load_state()
    yesterday = (datetime.date.today()
                 - datetime.timedelta(days=days_back)).isoformat()
    st["last_advanced"] = yesterday
    reveal_state._save(st)


class TestRevealDisabledRegression:
    def test_off_returns_full_window_and_no_state(self, tmp_path, monkeypatch):
        """回归盾:reveal 关闭时行数与旧路径一致,且不写状态文件。"""
        data_dir = str(tmp_path / "tianchi")
        _write_daily_jdata(data_dir, days=10)
        monkeypatch.setenv("DATA_SOURCE", "tianchi")
        monkeypatch.setenv("TIANCHI_DATA_DIR", data_dir)
        # 不设 TIANCHI_REVEAL_ENABLED → 默认 False
        from config.settings import get_settings
        get_settings.cache_clear()

        df = dl.load_tianchi_orders()
        assert df is not None and len(df) == 10      # 全窗口
        assert not os.path.exists(_state_path())      # 零状态写入

    def test_disabled_skips_cut_even_if_state_exists(self, tmp_path, monkeypatch):
        """开关关闭时即使残留状态文件也不截断(半开状态清理的对称回归)。"""
        data_dir = _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        assert len(dl.load_tianchi_orders()) == 3     # 先开一次,状态已建
        monkeypatch.setenv("TIANCHI_REVEAL_ENABLED", "false")
        from config.settings import get_settings
        get_settings.cache_clear()
        df = dl.load_tianchi_orders()                 # 关闭 → 全窗口
        assert df is not None and len(df) == 10


class TestRevealStateUnit:
    @pytest.fixture(autouse=True)
    def _preheat3_env(self, monkeypatch):
        """状态机单测直接调 ensure_state,须注入 preheat=3 + 启用开关。"""
        monkeypatch.setenv("TIANCHI_REVEAL_ENABLED", "true")
        monkeypatch.setenv("TIANCHI_REVEAL_PREHEAT_DAYS", "3")
        monkeypatch.setenv("TIANCHI_REVEAL_STEP", "1")
        from config.settings import get_settings
        get_settings.cache_clear()
        yield

    def test_init_preheat(self):
        from pipeline import reveal_state
        st = reveal_state.ensure_state(10)
        assert st["revealed_days"] == 3
        assert st["total_days"] == 10
        assert st["last_advanced"] == datetime.date.today().isoformat()

    def test_same_day_idempotent(self):
        from pipeline import reveal_state
        reveal_state.ensure_state(10)
        st = reveal_state.ensure_state(10)
        assert st["revealed_days"] == 3              # 同日不再推进
        assert reveal_state.maybe_advance() is False  # watcher 侧同日 no-op

    def test_advance_once_per_new_day(self):
        from pipeline import reveal_state
        reveal_state.ensure_state(10)
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        st = reveal_state.ensure_state(10, today=tomorrow)
        assert st["revealed_days"] == 4              # +STEP=1
        st = reveal_state.ensure_state(10, today=tomorrow)
        assert st["revealed_days"] == 4              # 同日第二次不推进
        assert reveal_state.maybe_advance(today=tomorrow) is False

    def test_maybe_advance_true_on_new_day(self):
        from pipeline import reveal_state
        reveal_state.ensure_state(10)
        tomorrow = datetime.date.today() + datetime.timedelta(days=1)
        assert reveal_state.maybe_advance(today=tomorrow) is True
        assert reveal_state.load_state()["revealed_days"] == 4

    def test_cap_at_total(self):
        from pipeline import reveal_state
        st = reveal_state.ensure_state(10)
        st = reveal_state.force_set(99)              # 超量钳制
        assert st["revealed_days"] == 10
        assert reveal_state.maybe_advance() is False  # 到顶不再推进
        st = reveal_state.ensure_state(10)
        assert st["revealed_days"] == 10

    def test_corrupt_state_defaults_no_raise(self):
        from pipeline import reveal_state
        os.makedirs(os.path.dirname(_state_path()), exist_ok=True)
        with open(_state_path(), "w", encoding="utf-8") as f:
            f.write("{not json!!")
        assert reveal_state.load_state()["revealed_days"] == 0   # 不抛
        st = reveal_state.ensure_state(10)                       # 重建
        assert st["revealed_days"] == 3

    def test_span_shrink_clamps(self):
        from pipeline import reveal_state
        reveal_state.ensure_state(10)
        st = reveal_state.ensure_state(5)            # CSV 缩短
        assert st["total_days"] == 5
        assert st["revealed_days"] == 3              # 钳制不超新跨度

    def test_first_load_does_not_double_advance(self):
        """首启(preheat 3/10):初始化当天即 last=today,不再 +STEP。"""
        from pipeline import reveal_state
        st = reveal_state.ensure_state(10)
        assert st["revealed_days"] == 3              # 不是 3+1=4


class TestRevealLoadIntegration:
    def test_init_cut_keeps_tail_only(self, tmp_path, monkeypatch):
        _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        df = dl.load_tianchi_orders()
        assert df is not None and len(df) == 3        # 只有最后 3 天(每日 1 单)

    def test_same_day_reload_stable(self, tmp_path, monkeypatch):
        _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        first = dl.load_tianchi_orders()
        second = dl.load_tianchi_orders()
        assert len(first) == len(second) == 3         # 同日幂等
        assert sorted(first["order_id"]) == sorted(second["order_id"])

    def test_advance_grows_window(self, tmp_path, monkeypatch):
        _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        assert len(dl.load_tianchi_orders()) == 3
        _backdate_last_advanced(1)                    # 模拟"新的一天"
        df = dl.load_tianchi_orders()
        assert len(df) == 4                           # 窗口 +1 天
        df = dl.load_tianchi_orders()
        assert len(df) == 4                           # 同日再载不推进

    def test_cap_equals_full_window(self, tmp_path, monkeypatch):
        _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        from pipeline import reveal_state
        reveal_state.force_set(10)                    # 直接到顶
        df = dl.load_tianchi_orders()
        assert df is not None and len(df) == 10       # = 全窗口
        assert reveal_state.maybe_advance() is False

    def test_span_shrink_on_reload(self, tmp_path, monkeypatch):
        data_dir = _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        assert len(dl.load_tianchi_orders()) == 3
        _write_daily_jdata(data_dir, days=5)          # CSV 缩短为 5 天
        df = dl.load_tianchi_orders()
        assert len(df) == 3                           # 钳到 3/5,不超新跨度
        from pipeline import reveal_state
        st = reveal_state.load_state()
        assert st["total_days"] == 5 and st["revealed_days"] == 3

    def test_corrupt_state_reinitializes_on_load(self, tmp_path, monkeypatch):
        _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        os.makedirs(os.path.dirname(_state_path()), exist_ok=True)
        with open(_state_path(), "w", encoding="utf-8") as f:
            f.write("{corrupt")
        df = dl.load_tianchi_orders()                 # 不抛,重建状态
        assert df is not None and len(df) == 3


class TestRevealFingerprint:
    def test_fp_stable_when_disabled(self, tmp_path, monkeypatch):
        """关闭:指纹不含 reveal,两次调用稳定。"""
        from config.settings import get_settings
        data_dir = str(tmp_path / "tianchi")
        _write_daily_jdata(data_dir, days=10)
        monkeypatch.setenv("DATA_SOURCE", "tianchi")
        monkeypatch.setenv("TIANCHI_DATA_DIR", data_dir)
        get_settings.cache_clear()
        assert dl.data_fingerprint() == dl.data_fingerprint()

    def test_fp_changes_when_revealed_days_changes(self, tmp_path, monkeypatch):
        _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        fp1 = dl.data_fingerprint()
        assert fp1 == dl.data_fingerprint()           # 同日稳定
        _backdate_last_advanced(1)
        dl.load_tianchi_orders()                      # 指纹只读,须由加载推进
        fp3 = dl.data_fingerprint()
        assert fp3 != fp1                             # 窗口增长 → 指纹变化

    def test_fp_stable_at_steady_state(self, tmp_path, monkeypatch):
        """到顶后 last_advanced 照走但不进指纹 → 稳态不触发每日全量重算。"""
        _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        from pipeline import reveal_state
        reveal_state.force_set(10)
        fp1 = dl.data_fingerprint()
        _backdate_last_advanced(1)                    # 日期照走
        fp2 = dl.data_fingerprint()                   # 指纹不因日期变化
        assert fp1 == fp2


class TestRevealFunnelConsistency:
    def test_funnel_respects_cut(self, tmp_path, monkeypatch):
        """漏斗(独立读 Action CSV)遵守同一揭晓窗口:初始 3 天 → 6 行行为。"""
        _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        dl.load_tianchi_orders()                      # 先初始化状态(订单口径)
        from pipeline import funnel
        actions = funnel._load_jdata_actions()
        assert actions is not None and len(actions) == 6   # 3 天 × 2 行为

    def test_funnel_skips_when_state_uninitialized(self, tmp_path, monkeypatch):
        """漏斗先于订单加载被调用:状态缺失 → 跳过截断返回全量,不崩不建状态。"""
        _enable_reveal(monkeypatch, tmp_path, days=10, preheat=3)
        assert not os.path.exists(_state_path())
        from pipeline import funnel
        actions = funnel._load_jdata_actions()
        assert actions is not None and len(actions) == 20   # 全量 10 天 × 2
        assert not os.path.exists(_state_path())            # 不初始化状态
