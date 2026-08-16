"""自动选 K(Gap statistic + 1-SE 规则 + 最小簇约束)单元测试。

纯离线:合成高斯团数据,不发网络请求、不读真实数据集。
"""
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from pipeline.user_segmentation import find_optimal_k


def _three_blobs(n_per_blob: int = 150) -> np.ndarray:
    """3 个分离良好的高斯团(3 维,模拟标准化后的 RFM 空间)。"""
    rng = np.random.default_rng(0)
    centers = [(-4.0, -4.0, -4.0), (0.0, 0.0, 0.0), (4.0, 4.0, 4.0)]
    parts = [rng.normal(c, 0.5, size=(n_per_blob, 3)) for c in centers]
    X = np.vstack(parts)
    return StandardScaler().fit_transform(X)


class TestGapStatistic:
    def test_three_blobs_pick_three(self):
        X = _three_blobs()
        best_k, diag = find_optimal_k(X)
        assert best_k == 3
        assert "gap" in diag[3] and "gap_se" in diag[3]   # 诊断字段存在

    def test_deterministic(self):
        X = _three_blobs()
        assert find_optimal_k(X)[0] == find_optimal_k(X)[0]

    def test_k_min_respected(self):
        """搜索下界生效:强制 k_min=3 时不会选回 2。"""
        X = _three_blobs()
        best_k, _ = find_optimal_k(X, k_min=3, k_max=6)
        assert best_k >= 3

    def test_single_blob_never_exceeds_data_size(self):
        """退化数据:单团 + 极小样本,不崩且 K 不超过数据规模。"""
        rng = np.random.default_rng(1)
        X = StandardScaler().fit_transform(rng.normal(size=(8, 3)))
        best_k, _ = find_optimal_k(X, k_min=2, k_max=10)
        assert 2 <= best_k <= 7      # k_max 被 min(k_max, n-1) 钳制


class TestMinClusterConstraint:
    def test_constraint_prunes_all_then_falls_back(self, monkeypatch):
        """最小簇约束:设到不可满足 → 全部淘汰 → 退回 k_max(有日志可查)。"""
        monkeypatch.setenv("AUTO_K_MIN_CLUSTER_SIZE", "100000")
        from config.settings import get_settings
        get_settings.cache_clear()

        X = _three_blobs(n_per_blob=100)
        best_k, diag = find_optimal_k(X, k_min=2, k_max=6)
        assert best_k == 6
        assert all(diag[k]["valid"] is False for k in range(2, 7))

    def test_normal_data_marks_valid(self):
        X = _three_blobs()
        _, diag = find_optimal_k(X, k_min=2, k_max=5)
        assert diag[3]["valid"] is True
