"""数据指纹 + Watcher 增量重算(纯函数 + engine poll spy)。

全部离线:不触 LLM、不读真实 JData。
"""
import os

import pandas as pd
import pytest

from pipeline.data_loader import data_fingerprint
from config.settings import get_settings


def _orders(n: int = 0) -> list[dict]:
    return [
        {"order_id": 10000 + i, "user_id": 1, "created_at": f"2026-08-1{i}T10:00:00"}
        for i in range(n)
    ]


class TestDataFingerprint:
    def test_same_state_same_fingerprint(self, monkeypatch):
        monkeypatch.setattr("api.data_store._orders", _orders(3))
        assert data_fingerprint() == data_fingerprint()

    def test_new_order_changes_fingerprint(self, monkeypatch):
        monkeypatch.setattr("api.data_store._orders", _orders(0))
        fp0 = data_fingerprint()
        monkeypatch.setattr("api.data_store._orders", _orders(1))
        fp1 = data_fingerprint()
        assert fp0 != fp1

    def test_latest_order_time_changes_fingerprint(self, monkeypatch):
        monkeypatch.setattr("api.data_store._orders", _orders(2))
        fp0 = data_fingerprint()
        monkeypatch.setattr("api.data_store._orders", [
            {"order_id": 10000, "user_id": 1, "created_at": "2026-08-20T10:00:00"},
            {"order_id": 10001, "user_id": 1, "created_at": "2026-08-21T10:00:00"},
        ])
        fp1 = data_fingerprint()
        assert fp0 != fp1

    def test_data_source_change_changes_fingerprint(self, monkeypatch):
        monkeypatch.setattr("api.data_store._orders", [])
        get_settings.cache_clear()
        try:
            monkeypatch.setenv("DATA_SOURCE", "auto")
            fp_auto = data_fingerprint()
            monkeypatch.setenv("DATA_SOURCE", "tianchi")
            get_settings.cache_clear()
            fp_tianchi = data_fingerprint()
            assert fp_auto != fp_tianchi
        finally:
            get_settings.cache_clear()

    def test_csv_mtime_change_changes_fingerprint(self, monkeypatch, tmp_path):
        """JData 文件更新(mtime)→ 指纹变化(CSV 更新触发重算)。"""
        monkeypatch.setattr("api.data_store._orders", [])
        data_dir = tmp_path / "tianchi"
        data_dir.mkdir()
        f = data_dir / "JData_Action_201602.csv"
        f.write_text("user_id,sku_id,time,model_id,type,cate,brand\n", encoding="utf-8")
        monkeypatch.setenv("DATA_SOURCE", "tianchi")
        monkeypatch.setenv("TIANCHI_DATA_DIR", str(data_dir))
        get_settings.cache_clear()
        try:
            fp0 = data_fingerprint()
            # touch 文件(mtime +1s)
            import time as _time
            _time.sleep(1.1)
            f.write_text("user_id,sku_id,time,model_id,type,cate,brand\nx\n", encoding="utf-8")
            fp1 = data_fingerprint()
            assert fp0 != fp1
        finally:
            get_settings.cache_clear()


class TestPollIncrementalRefresh:
    """engine.poll_once:指纹不变 → 不强制重算;订单变化 → force_refresh。"""

    def _seed_snapshots(self):
        from pipeline.user_segmentation import save_snapshot
        import time as _time
        rfm = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 6],
            "recency": [3, 10, 20, 30, 45, 60],
            "frequency": [5.0, 3.0, 2.0, 1.5, 1.0, 1.0],
            "monetary": [800.0, 500.0, 300.0, 200.0, 100.0, 80.0],
            "segment": [2, 2, 1, 1, 0, 0],
            "flow_tag": ["active", "active", "potential", "dormant", "churned", "churned"],
        })
        save_snapshot(rfm, k_value=3, silhouette=0.6)
        _time.sleep(0.05)
        save_snapshot(rfm, k_value=3, silhouette=0.6)

    def test_fingerprint_hit_skips_full_refresh(self, monkeypatch):
        import asyncio
        self._seed_snapshots()
        calls: list[bool] = []
        rfm = pd.DataFrame({
            "user_id": [1, 2, 3, 4, 5, 6],
            "recency": [3, 3, 3, 3, 3, 3],          # 全部 < 30 → dormant 不触发
            "frequency": [1.0] * 6,
            "monetary": [100.0] * 6,
            "segment": [2, 2, 1, 1, 0, 0],
            "flow_tag": ["active"] * 6,
        })

        def fake_load(force_refresh=False):
            calls.append(force_refresh)
            return rfm, rfm, {}

        monkeypatch.setattr("skills.user_segment._load_and_process", fake_load)
        monkeypatch.setattr("api.data_store._orders", [])
        import watcher.task_manager as tm_mod
        tm_mod._task_manager = None

        async def _run():
            from watcher.engine import WatcherEngine
            engine = WatcherEngine()
            await engine.poll_once()
            await engine.poll_once()                # 指纹不变
            assert calls == [True, False]           # 首次全量 → 指纹命中跳过
            monkeypatch.setattr("api.data_store._orders",
                                [{"order_id": 1, "user_id": 1, "created_at": "x"}])
            await engine.poll_once()                # 订单变化 → 重新全量
            assert calls == [True, False, True]

        asyncio.run(_run())
