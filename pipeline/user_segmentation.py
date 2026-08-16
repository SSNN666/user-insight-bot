"""User segmentation via K-Means, interpretable decision-tree rules,
flow classification, and cluster snapshot persistence.

Phase 3 upgrades:
- ``find_optimal_k()`` — Gap statistic + 1-SE 简约规则自动选 K
- ``perform_clustering()`` — uses auto-K when ``n_clusters=None``
- ``extract_rules()`` — returns dict with pruned tree + flow summary
- ``classify_user_flow()`` — rule-based flow tagging
- ``ClusterSnapshot`` — save / load / compare across time
"""

import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import StandardScaler

from config.settings import get_settings
from errors.exceptions import ComputationError
from log.logger import get_logger

logger = get_logger(__name__)


# ── Optimal K Selection ─────────────────────────────────────────


def find_optimal_k(
    X_scaled: np.ndarray,
    k_min: int | None = None,
    k_max: int | None = None,
) -> tuple[int, dict]:
    """Gap statistic + 1-SE 简约规则自动选 K。

    选 K 依据(顺序):
      1. Gap statistic(Tibshirani 2001):真实数据 inertia 对比均匀参考分布,
         1-SE 规则取"再增加 K 收益不再显著"的最小 K——不会像肘部+轮廓的
         组合分那样单调爬升永远选到上限;
      2. 最小簇约束:任一簇人数 < max(AUTO_K_MIN_CLUSTER_SIZE, n×RATIO) 的 K
         直接淘汰(业务分群需要最低规模,真实数据实测 k≥6 出现 18 人迷你簇);
      3. 全部淘汰时退回满足约束的最小 K,再不行用 k_max。
    大样本(>8000 人)评估时固定种子下采样,聚类本身仍用全量数据。
    肘部/silhouette 保留在 diagnostics 里作为可观测信息。

    Args:
        X_scaled: Standardized feature matrix (n_samples, n_features).
        k_min, k_max: Search range. Defaults from settings.

    Returns:
        (best_k, diagnostics_dict) where diagnostics maps k → metrics.
    """
    settings = get_settings()
    k_min = k_min or settings.AUTO_K_MIN
    k_max = k_max or settings.AUTO_K_MAX
    k_max = min(k_max, max(2, len(X_scaled) - 1))
    k_min = min(k_min, k_max)
    rng = np.random.default_rng(42)

    if len(X_scaled) > 8000:
        idx = rng.choice(len(X_scaled), 8000, replace=False)
        X = X_scaled[idx]
    else:
        X = X_scaled
    n = len(X)

    ks = list(range(1, k_max + 1))
    diagnostics: dict[int, dict] = {}
    inertias: dict[int, float] = {}
    silhouettes: dict[int, float] = {}

    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        inertias[k] = km.inertia_
        silhouettes[k] = silhouette_score(X, labels) if k >= 2 else 0.0

    # Gap statistic:B 组均匀参考分布(覆盖观测范围,与标准化量纲无关)
    B = settings.AUTO_K_GAP_B
    ref = rng.uniform(X.min(axis=0), X.max(axis=0), size=(B,) + X.shape)
    log_w_ref = np.empty((B, len(ks)))
    for b in range(B):
        for j, k in enumerate(ks):
            log_w_ref[b, j] = np.log(
                KMeans(n_clusters=k, random_state=42, n_init=10).fit(ref[b]).inertia_
            )
    gap = log_w_ref.mean(axis=0) - np.array([np.log(inertias[k]) for k in ks])
    sk = log_w_ref.std(axis=0, ddof=1) * np.sqrt(1 + 1 / B)

    def _valid(k: int) -> bool:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        min_size = max(settings.AUTO_K_MIN_CLUSTER_SIZE,
                       int(n * settings.AUTO_K_MIN_CLUSTER_RATIO))
        return int(np.bincount(labels).min()) >= min_size

    # 1-SE 规则:最小的 k 满足 gap(k) ≥ gap(k+1) − s(k+1)
    best_k = None
    for k in range(k_min, k_max):
        if not _valid(k):
            continue
        if gap[k - 1] >= gap[k] - sk[k]:
            best_k = k
            break
    if best_k is None:
        valid = [k for k in range(k_min, k_max + 1) if _valid(k)]
        best_k = valid[0] if valid else k_max

    for j, k in enumerate(ks):
        diagnostics[k] = {
            "inertia": round(inertias[k], 2),
            "silhouette": round(silhouettes[k], 4),
            "gap": round(float(gap[j]), 4),
            "gap_se": round(float(sk[j]), 4),
            "valid": bool(k < k_min or _valid(k)),
        }

    logger.info("optimal_k_found", extra={
        "best_k": best_k,
        "method": "gap_statistic",
        "silhouette": round(silhouettes[best_k], 4),
        "gap": round(float(gap[best_k - 1]), 4),
        "search_range": f"[{k_min},{k_max}]",
    })
    return best_k, diagnostics


# ── Clustering ──────────────────────────────────────────────────


def perform_clustering(
    rfm_df: pd.DataFrame, n_clusters: int | None = None
) -> tuple[pd.DataFrame, KMeans, StandardScaler, int]:
    """Cluster users by standardized RFM using K-Means.

    When ``n_clusters=None`` and ``AUTO_K_ENABLED=True``, automatically
    selects the best K via elbow + silhouette.

    Returns:
        (rfm_df_with_segment, kmeans_model, scaler, actual_k)
    """
    settings = get_settings()

    features = ['recency', 'frequency', 'monetary']
    X = rfm_df[features].copy()

    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        if n_clusters is None and settings.AUTO_K_ENABLED:
            n_clusters, _diag = find_optimal_k(X_scaled)
        elif n_clusters is None:
            n_clusters = settings.DATA_N_CLUSTERS

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        rfm_df = rfm_df.copy()
        rfm_df['segment'] = kmeans.fit_predict(X_scaled)

        # Re-order clusters by monetary mean for interpretability
        cluster_avg = (
            rfm_df.groupby('segment')['monetary'].mean().sort_values()
        )
        mapping = {old: new for new, old in enumerate(cluster_avg.index)}
        rfm_df['segment'] = rfm_df['segment'].map(mapping)
        # 同步模型 labels_(旧编号 → 新分群),避免调用方拿到与 rfm_df 不一致的模型
        kmeans.labels_ = np.array([mapping[l] for l in kmeans.labels_])

        logger.info("clustering_done", extra={
            "n_clusters": n_clusters, "users": len(rfm_df),
        })
        return rfm_df, kmeans, scaler, n_clusters
    except Exception as e:
        raise ComputationError(
            f"Clustering failed: {e}", context={"n_users": len(rfm_df)}
        ) from e


# ── Decision Tree Rules ─────────────────────────────────────────


def extract_rules(rfm_df: pd.DataFrame) -> dict:
    """Fit a pruned decision tree and extract rules + flow summary.

    Uses ccp_alpha cost-complexity pruning to remove redundant splits.
    Incorporates flow features (recency_change_rate, frequency_trend) if available.

    Returns:
        ``{"text": str, "tree": DecisionTreeClassifier, "flow_summary": dict}``
    """
    feature_cols = ['recency', 'frequency', 'monetary']
    # Add flow features if present
    if 'recency_change_rate' in rfm_df.columns:
        feature_cols.append('recency_change_rate')
    if 'frequency_trend' in rfm_df.columns:
        feature_cols.append('frequency_trend')

    X = rfm_df[feature_cols].copy()
    y = rfm_df['segment']

    try:
        # Find best ccp_alpha via cost-complexity pruning path
        dt_raw = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt_raw.fit(X, y)
        path = dt_raw.cost_complexity_pruning_path(X, y)
        # Use the alpha that gives the best balance (second-to-last = most pruned but not trivial)
        alphas = path.ccp_alphas
        if len(alphas) >= 2:
            best_alpha = alphas[-2]  # most aggressive prune before root-only
        else:
            best_alpha = 0.0

        dt = DecisionTreeClassifier(
            max_depth=3, random_state=42, ccp_alpha=best_alpha,
        )
        dt.fit(X, y)
        rules_text = export_text(dt, feature_names=feature_cols)

        # Compute flow summary: per-segment distribution
        flow_summary: dict = {}
        if 'flow_tag' in rfm_df.columns:
            for seg in sorted(rfm_df['segment'].unique()):
                seg_data = rfm_df[rfm_df['segment'] == seg]
                flows = seg_data['flow_tag'].value_counts().to_dict()
                flow_summary[int(seg)] = flows

        logger.info("rules_extracted", extra={
            "tree_depth": dt.get_depth(),
            "ccp_alpha": round(float(best_alpha), 6),
            "features": feature_cols,
        })
        return {"text": rules_text, "tree": dt, "flow_summary": flow_summary}
    except Exception as e:
        raise ComputationError(
            f"Rule extraction failed: {e}", context={"n_users": len(rfm_df)}
        ) from e


# ── Flow Classification ─────────────────────────────────────────


def classify_user_flow(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Tag each user with a flow status based on RFM thresholds.

    Tags:
        - ``active``: recency ≤ 30 and frequency ≥ 3
        - ``potential``: recency ≤ 90, monetary ≥ median, frequency < 3
        - ``dormant``: 90 < recency ≤ 180
        - ``churned``: recency > 180

    Args:
        rfm_df: Must contain ``[user_id, recency, frequency, monetary]``.

    Returns:
        Same DataFrame with added ``flow_tag`` column.
    """
    try:
        df = rfm_df.copy()
        median_monetary = df['monetary'].median()

        conditions = [
            (df['recency'] <= 30) & (df['frequency'] >= 3),
            (df['recency'] <= 90) & (df['monetary'] >= median_monetary) & (df['frequency'] < 3),
            (df['recency'] > 90) & (df['recency'] <= 180),
            (df['recency'] > 180),
        ]
        tags = ['active', 'potential', 'dormant', 'churned']

        df['flow_tag'] = 'stable'  # default
        for cond, tag in zip(conditions, tags):
            df.loc[cond, 'flow_tag'] = tag

        # Add derived flow features for the decision tree
        df['recency_change_rate'] = df['recency'] / (df['recency'].mean() + 1e-10)
        df['frequency_trend'] = df['frequency'] / (df['frequency'].mean() + 1e-10)

        tag_counts = df['flow_tag'].value_counts().to_dict()
        logger.info("flow_classified", extra={"users": len(df), "tags": tag_counts})
        return df
    except Exception as e:
        raise ComputationError(
            f"Flow classification failed: {e}", context={"n_users": len(rfm_df)}
        ) from e


# ── Cluster Snapshot ────────────────────────────────────────────


@dataclass
class ClusterSnapshot:
    timestamp: str          # ISO 8601
    k_value: int
    silhouette: float
    segment_stats: dict     # {seg_id: {"user_count": N, "avg_recency": R, ...}}
    user_assignments: dict  # {user_id: segment}
    flow_distribution: dict # {seg_id: {"active": N, "potential": N, ...}}

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "k_value": self.k_value,
            "silhouette": self.silhouette,
            "segment_stats": self.segment_stats,
            "flow_distribution": self.flow_distribution,
            "user_count": len(self.user_assignments),
        }


def save_snapshot(
    rfm_df: pd.DataFrame,
    k_value: int,
    silhouette: float,
    snapshot_dir: str | None = None,
) -> ClusterSnapshot:
    """Persist current clustering state as a snapshot."""
    settings = get_settings()
    snap_dir = snapshot_dir or settings.SNAPSHOT_DIR
    os.makedirs(snap_dir, exist_ok=True)

    timestamp = pd.Timestamp.now().isoformat()

    # Per-segment stats
    seg_stats: dict = {}
    for seg in sorted(rfm_df['segment'].unique()):
        seg_df = rfm_df[rfm_df['segment'] == seg]
        seg_stats[int(seg)] = {
            "user_count": len(seg_df),
            "avg_recency": round(float(seg_df['recency'].mean()), 1),
            "avg_frequency": round(float(seg_df['frequency'].mean()), 2),
            "avg_monetary": round(float(seg_df['monetary'].mean()), 2),
        }

    # User assignments
    user_assignments = dict(zip(
        rfm_df['user_id'].astype(int),
        rfm_df['segment'].astype(int),
    ))

    # Flow distribution
    flow_dist: dict = {}
    if 'flow_tag' in rfm_df.columns:
        for seg in sorted(rfm_df['segment'].unique()):
            seg_df = rfm_df[rfm_df['segment'] == seg]
            flow_dist[int(seg)] = seg_df['flow_tag'].value_counts().to_dict()

    snapshot = ClusterSnapshot(
        timestamp=timestamp,
        k_value=k_value,
        silhouette=silhouette,
        segment_stats=seg_stats,
        user_assignments=user_assignments,
        flow_distribution=flow_dist,
    )

    # Save to file
    fname = f"{timestamp[:19].replace(':', '')}.pkl"
    fpath = os.path.join(snap_dir, fname)
    with open(fpath, "wb") as f:
        pickle.dump(snapshot, f)

    # ── 保留策略:超出 SNAPSHOT_KEEP 的最旧快照自动清理(防无限累积) ──
    keep = settings.SNAPSHOT_KEEP
    if keep and keep > 0:
        existing = sorted(
            fn for fn in os.listdir(snap_dir) if fn.endswith(".pkl")
        )
        for old in existing[:-keep]:
            try:
                os.remove(os.path.join(snap_dir, old))
            except OSError:
                pass

    logger.info("snapshot_saved", extra={
        "path": fpath, "k": k_value, "users": len(user_assignments),
    })
    return snapshot


def load_snapshots(
    since: str | None = None,
    until: str | None = None,
    snapshot_dir: str | None = None,
) -> list[ClusterSnapshot]:
    """Load snapshots within a time range.

    Args:
        since: ISO timestamp lower bound (inclusive).
        until: ISO timestamp upper bound (inclusive).
        snapshot_dir: Directory to scan.

    Returns:
        List of ClusterSnapshot sorted by timestamp.
    """
    settings = get_settings()
    snap_dir = snapshot_dir or settings.SNAPSHOT_DIR
    if not os.path.isdir(snap_dir):
        return []

    snapshots: list[ClusterSnapshot] = []
    for fname in sorted(os.listdir(snap_dir)):
        if not fname.endswith('.pkl'):
            continue
        fpath = os.path.join(snap_dir, fname)
        try:
            with open(fpath, "rb") as f:
                snap: ClusterSnapshot = pickle.load(f)
            ts = snap.timestamp
            if since and ts < since:
                continue
            if until and ts > until:
                continue
            snapshots.append(snap)
        except Exception as e:
            logger.warning("snapshot_load_error", extra={"path": fpath, "error": str(e)})

    return snapshots


def compare_snapshots(
    snap1: ClusterSnapshot,
    snap2: ClusterSnapshot,
) -> dict:
    """Compute per-segment deltas between two snapshots.

    Returns:
        ``{seg_id: {"user_count_delta": N, "avg_monetary_delta": M, ...}}``
    """
    deltas: dict = {}
    all_segs = set(snap1.segment_stats.keys()) | set(snap2.segment_stats.keys())

    for seg in sorted(all_segs):
        s1 = snap1.segment_stats.get(seg, {})
        s2 = snap2.segment_stats.get(seg, {})
        deltas[seg] = {}
        for metric in ["user_count", "avg_recency", "avg_frequency", "avg_monetary"]:
            v1 = s1.get(metric, 0)
            v2 = s2.get(metric, 0)
            deltas[seg][f"{metric}_delta"] = round(v2 - v1, 2)

    logger.info("snapshots_compared", extra={
        "t1": snap1.timestamp, "t2": snap2.timestamp, "segs": len(deltas),
    })
    return deltas


def segment_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-segment statistics (user count, mean R/F/M)."""
    try:
        agg_cols = ['user_id', 'recency', 'frequency', 'monetary']
        if 'flow_tag' in rfm_df.columns:
            # Include flow distribution counts
            pass  # keep simple for now; flow available via snapshot
        summary = rfm_df.groupby('segment').agg(
            用户数=('user_id', 'nunique'),
            平均近度=('recency', 'mean'),
            平均频次=('frequency', 'mean'),
            平均消费=('monetary', 'mean'),
        ).reset_index()
        return summary
    except Exception as e:
        raise ComputationError(
            f"Segment summary failed: {e}", context={"n_users": len(rfm_df)}
        ) from e
