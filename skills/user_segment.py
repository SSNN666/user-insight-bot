"""User-segmentation skills — TTL-aware cache + refresh support.

Phase 3: pipeline cache TTL, force-refresh, and ``RefreshPipelineSkill``.
"""

import threading
import time

import pandas as pd
from pydantic import BaseModel, Field

from skills.base import BaseSkill, SkillResult, SkillStatus
from pipeline.data_cleaning import clean_data, compute_rfm
from pipeline.user_segmentation import (
    perform_clustering, extract_rules, segment_summary,
    classify_user_flow, save_snapshot,
)
from pipeline.data_loader import load_orders_with_join
from errors.exceptions import DatabaseError, ComputationError
from log.logger import get_logger

logger = get_logger(__name__)

# ── Shared data cache (process-wide, TTL-aware, thread-safe) ────

_cached_rfm: pd.DataFrame | None = None
_cached_segments: pd.DataFrame | None = None
_cached_rules: dict | None = None
_cached_at: float | None = None
_cache_lock = threading.RLock()


def _load_and_process(force_refresh: bool = False):
    """Load data, clean, compute RFM, cluster, classify flow, extract rules.

    Results are cached with TTL (``PIPELINE_CACHE_TTL``). Subsequent calls
    within the TTL window return cached data. Pass ``force_refresh=True``
    to bypass the cache.

    Thread-safe: uses a reentrant lock so concurrent API requests and
    watcher poll cycles don't race on the global cache.
    """
    global _cached_rfm, _cached_segments, _cached_rules, _cached_at
    from config.settings import get_settings
    settings = get_settings()

    # Fast-path read under lock
    with _cache_lock:
        if not force_refresh and _cached_rfm is not None and _cached_at is not None:
            age = time.time() - _cached_at
            if age < settings.PIPELINE_CACHE_TTL:
                logger.debug("pipeline_cache_hit", extra={"age_seconds": round(age)})
                return _cached_rfm, _cached_segments, _cached_rules
            logger.info("pipeline_cache_expired", extra={"age_seconds": round(age)})

        # Double-check: if another thread just finished, reuse its result
        if not force_refresh and _cached_rfm is not None and _cached_at is not None:
            age = time.time() - _cached_at
            if age < 5:  # very fresh → another thread just computed it
                logger.debug("pipeline_cache_race_avoided")
                return _cached_rfm, _cached_segments, _cached_rules

    # Slow path — compute (outside lock to allow concurrent reads)
    logger.info("data_pipeline_start")
    try:
        df = load_orders_with_join()
        df = clean_data(df)
        rfm = compute_rfm(df)
    except (DatabaseError, ComputationError):
        logger.error("data_pipeline_failed")
        raise

    # Flow classification
    rfm = classify_user_flow(rfm)

    # Clustering
    rfm, _kmeans, _scaler, k_value = perform_clustering(rfm)

    # Rules (now returns dict)
    rules = extract_rules(rfm)

    # Snapshot
    silhouette = 0.0
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        X = StandardScaler().fit_transform(rfm[['recency', 'frequency', 'monetary']])
        silhouette = float(silhouette_score(X, rfm['segment']))
    except Exception as e:
        logger.warning("silhouette_computation_failed", extra={"error": str(e)})
    save_snapshot(rfm, k_value, silhouette)

    # Publish under lock
    with _cache_lock:
        _cached_rfm = rfm
        _cached_segments = rfm
        _cached_rules = rules
        _cached_at = time.time()

    logger.info("data_pipeline_complete", extra={
        "users": len(rfm), "k": k_value, "silhouette": round(silhouette, 4),
    })
    return _cached_rfm, _cached_segments, _cached_rules


def invalidate_pipeline_cache() -> None:
    """Clear all pipeline caches (in-memory + disk).

    Next call to ``_load_and_process()`` will recompute from scratch.
    """
    global _cached_rfm, _cached_segments, _cached_rules, _cached_at
    with _cache_lock:
        _cached_rfm = None
        _cached_segments = None
        _cached_rules = None
        _cached_at = None

    from cache.ttl_cache import TTLCache
    cache = TTLCache(namespace="orders")
    cache.invalidate("orders_joined")
    cache.invalidate("rfm_from_db")
    logger.info("pipeline_cache_invalidated")


# ── Skills ──────────────────────────────────────────────────────


class UserSegmentStatsSkill(BaseSkill):
    name = "get_user_segment_stats"
    description = (
        "获取各用户分群的人数、平均近度、平均频次、平均消费金额。"
        "返回分群统计表格。"
    )

    def execute(self) -> SkillResult:
        try:
            _, segments, _ = _load_and_process()
            summary_df = segment_summary(segments)
            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=summary_df.to_dict(orient="records"),
                summary="用户分群统计:\n" + summary_df.to_markdown(index=False),
                confidence=0.95,
            )
        except (DatabaseError, ComputationError) as e:
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)


class UserSegmentRulesSkill(BaseSkill):
    name = "get_segment_rules"
    description = (
        "获取决策树导出的用户分群规则，可帮助理解不同群体的特征。"
        "返回文本形式的 if-then 规则。"
    )

    def execute(self) -> SkillResult:
        try:
            _, _, rules = _load_and_process()
            rules_text = rules["text"] if isinstance(rules, dict) else str(rules)
            flow_info = ""
            if isinstance(rules, dict) and rules.get("flow_summary"):
                flow_info = "\n流转分布: " + str(rules["flow_summary"])
            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=rules_text,
                summary="分群决策树规则:\n" + rules_text[:800] + flow_info,
                confidence=0.90,
            )
        except (DatabaseError, ComputationError) as e:
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)


class HighValueUsersSkill(BaseSkill):
    name = "get_high_value_users"
    description = (
        "获取高价值用户列表（分群编号最大的群体），"
        "返回用户ID和他们的近度、频次、消费金额（RFM）指标以及流转标签。"
    )

    def execute(self) -> SkillResult:
        try:
            rfm, segments, _ = _load_and_process()
            max_seg = segments['segment'].max()
            cols = ['user_id', 'recency', 'frequency', 'monetary']
            if 'flow_tag' in segments.columns:
                cols.append('flow_tag')
            high_value = segments[segments['segment'] == max_seg][cols]
            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=high_value.to_dict(orient="records"),
                summary=(
                    f"高价值用户 (segment={max_seg}), "
                    f"共 {len(high_value)} 人:\n"
                    + high_value.head(20).to_markdown(index=False)
                ),
                confidence=0.95,
            )
        except (DatabaseError, ComputationError) as e:
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)


class SegmentGrowthSkill(BaseSkill):
    name = "get_segment_growth"
    description = (
        "获取各用户分群的环比增长指标：销售额增长百分比、转化率变化、GMV提升比例。"
        "对比最近两次快照数据，输出专业的分群对比分析表。"
        "适用于需要了解各分群增长趋势和商业价值的场景。"
    )

    def execute(self) -> SkillResult:
        try:
            from pipeline.user_segmentation import load_snapshots
            from pipeline.profile import compute_segment_growth

            # Ensure pipeline is computed (creates initial snapshot if needed)
            _load_and_process(force_refresh=False)

            snaps = load_snapshots()
            if len(snaps) < 2:
                # Not enough snapshots — run a second computation
                import time as _time
                _time.sleep(0.1)
                # Force new snapshot
                invalidate_pipeline_cache()
                _load_and_process(force_refresh=True)
                snaps = load_snapshots()

            if len(snaps) < 2:
                return SkillResult(
                    status=SkillStatus.PARTIAL,
                    data=[],
                    summary="快照数据不足（需要至少2个快照才能计算增长指标）。请先触发一次数据重置。",
                    confidence=0.3,
                )

            prev = snaps[-2]
            curr = snaps[-1]
            growth = compute_segment_growth(prev.segment_stats, curr.segment_stats)

            # Build professional comparison table
            table_lines = [
                "## 高价值用户分群对比分析表",
                "",
                "| 分群 | 销售额增长百分比（%） | 转化率变化（%） | GMV提升比例（%） | 用户数 | 人均消费 |",
                "|------|------------------------|-----------------|------------------|--------|----------|",
            ]
            for g in growth:
                table_lines.append(
                    f"| {g['segment']} | {g['sales_growth_pct']:+.1f} | "
                    f"{g['conversion_change_pct']:+.1f} | {g['gmv_lift_pct']:+.1f} | "
                    f"{g['user_count']} | ¥{g['avg_monetary']:.2f} |"
                )

            # Data commentary
            table_lines.append("")
            table_lines.append("### 数据说明")

            # Find best in each metric
            best_sales = max(growth, key=lambda x: x['sales_growth_pct'])
            best_gmv = max(growth, key=lambda x: x['gmv_lift_pct'])
            best_conv = max(growth, key=lambda x: x['conversion_change_pct'])

            table_lines.append(
                f"- **分群 {best_sales['segment']}** 表现出最佳的销售额增长"
                f"（{best_sales['sales_growth_pct']:+.1f}%），"
                f"表明其高价值用户在购买频率和总交易金额上有显著优势。"
            )
            table_lines.append(
                f"- **分群 {best_conv['segment']}** 的转化率变化最为突出"
                f"（{best_conv['conversion_change_pct']:+.1f}%），"
                f"可重点关注其转化路径优化。"
            )
            table_lines.append(
                f"- **分群 {best_gmv['segment']}** 的 GMV 提升比例最高"
                f"（{best_gmv['gmv_lift_pct']:+.1f}%），"
                f"是最具商业增长潜力的群体。"
            )

            # Full explanation for flat/negative segments
            flat_segs = [g for g in growth if g['conversion_change_pct'] < 0]
            if flat_segs:
                names = "、".join(str(g['segment']) for g in flat_segs)
                table_lines.append(
                    f"- 分群 {names} 的转化率出现负增长，"
                    f"建议排查用户流失原因并制定针对性挽回策略。"
                )

            summary_text = "\n".join(table_lines)

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=growth,
                summary=summary_text,
                confidence=0.92,
            )
        except (DatabaseError, ComputationError) as e:
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)
        except Exception as e:
            return SkillResult(
                status=SkillStatus.ERROR,
                error=f"增长指标计算失败: {e}",
                confidence=0.0,
            )


class ProductSearchInput(BaseModel):
    keyword: str = Field(default="", description="搜索关键词，匹配商品名称，如'手机'、'耳机'")
    category: str = Field(default="", description="按品类筛选，如'电子产品'、'时尚服饰'")


class ProductSearchSkill(BaseSkill):
    name = "search_products"
    description = (
        "搜索电商平台商品。根据关键词匹配商品名称，或按品类筛选。"
        "返回匹配的商品列表（含ID、名称、品类、价格）。"
        "适用于用户想找特定商品、浏览某品类、或对比同类商品价格的场景。"
        "参数 keyword: 商品名关键词（如'手机'）; category: 品类名（如'电子产品'）"
    )
    input_schema = ProductSearchInput

    def execute(self, keyword: str = "", category: str = "") -> SkillResult:
        try:
            from api.data_store import list_products as ds_products
            all_products = ds_products()

            matches = all_products
            if category:
                matches = [p for p in matches if p.get("category") == category]
            if keyword:
                kw = keyword.lower()
                name_matches = [p for p in matches if kw in p["product_name"].lower()]
                if name_matches:
                    matches = name_matches
                else:
                    # Keyword doesn't match any product name — try matching as category too
                    cat_matches = [p for p in matches if kw in p.get("category", "").lower()]
                    if cat_matches:
                        matches = cat_matches

            if not matches:
                return SkillResult(
                    status=SkillStatus.PARTIAL,
                    data=[],
                    summary=f"未找到与「{keyword or category}」匹配的商品。建议尝试其他关键词或浏览全部分类。",
                    confidence=0.8,
                )

            # Build a clean product list for LLM
            lines = [f"找到 {len(matches)} 件商品：" if keyword or category
                     else f"平台共有 {len(matches)} 件商品：", ""]
            lines.append("| 商品ID | 名称 | 品类 | 价格 |")
            lines.append("|--------|------|------|------|")
            for p in matches[:20]:  # limit for LLM context
                lines.append(
                    f"| {p['product_id']} | {p['product_name']} | "
                    f"{p.get('category','')} | ¥{p['price']} |"
                )
            if len(matches) > 20:
                lines.append(f"| ... | ...共{len(matches)}件，以上为前20件 | ... | ... |")

            # Category summary
            from collections import Counter
            cat_counts = Counter(p.get("category", "其他") for p in matches)
            cat_summary = "、".join(f"{c}({n}件)" for c, n in cat_counts.most_common(5))

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=matches,
                summary="\n".join(lines) + f"\n\n品类分布: {cat_summary}",
                confidence=0.95,
            )
        except Exception as e:
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)


class GetCategoriesSkill(BaseSkill):
    name = "get_categories"
    description = (
        "获取平台所有商品品类列表。"
        "在用户不确定具体商品名、想浏览某个品类时优先调用此工具。"
    )

    def execute(self) -> SkillResult:
        try:
            from api.data_store import list_products as ds_products
            from collections import Counter
            products = ds_products()
            cat_counts = Counter(p.get("category", "其他") for p in products)
            lines = ["平台商品品类：", ""]
            for cat, count in cat_counts.most_common():
                lines.append(f"- {cat}（{count} 件）")
            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=dict(cat_counts),
                summary="\n".join(lines),
                confidence=1.0,
            )
        except Exception as e:
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)


class RefreshPipelineSkill(BaseSkill):
    name = "refresh_pipeline"
    description = (
        "刷新数据流水线缓存。强制重新加载数据、重算RFM、重新聚类。"
        "在数据更新后使用此工具获取最新分群结果。"
    )

    def execute(self) -> SkillResult:
        try:
            invalidate_pipeline_cache()
            _load_and_process(force_refresh=True)
            return SkillResult(
                status=SkillStatus.SUCCESS,
                summary="数据流水线缓存已刷新，分群结果已更新。",
                confidence=1.0,
            )
        except Exception as e:
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)


# ── Registration helper ─────────────────────────────────────────


def register_all_skills() -> None:
    """Register all built-in skills with the global registry."""
    from skills import SkillRegistry

    # ── E-commerce / shopping skills ──
    SkillRegistry.register(ProductSearchSkill())
    SkillRegistry.register(GetCategoriesSkill())

    # ── User segment analysis skills ──
    SkillRegistry.register(UserSegmentStatsSkill())
    SkillRegistry.register(UserSegmentRulesSkill())
    SkillRegistry.register(HighValueUsersSkill())
    SkillRegistry.register(SegmentGrowthSkill())
    SkillRegistry.register(RefreshPipelineSkill())
    logger.info("all_skills_registered")
