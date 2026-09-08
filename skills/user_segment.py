"""User-segmentation skills — TTL-aware cache + refresh support.

Phase 3: pipeline cache TTL, force-refresh, and ``RefreshPipelineSkill``.
"""

import re
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
                return _cached_rfm.copy(), _cached_segments.copy(), dict(_cached_rules)
            logger.info("pipeline_cache_expired", extra={"age_seconds": round(age)})

        # Double-check: if another thread just finished, reuse its result
        if not force_refresh and _cached_rfm is not None and _cached_at is not None:
            age = time.time() - _cached_at
            if age < 5:  # very fresh → another thread just computed it
                logger.debug("pipeline_cache_race_avoided")
                return _cached_rfm.copy(), _cached_segments.copy(), dict(_cached_rules)

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
    # 返回副本:缓存是全局共享的,调用方原地修改会污染后续所有读取方
    return _cached_rfm.copy(), _cached_segments.copy(), dict(_cached_rules)


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

    # 联动清空漏斗行为缓存(否则 refresh 后漏斗仍显示旧数据)
    try:
        from pipeline.funnel import invalidate_funnel_cache
        invalidate_funnel_cache()
    except Exception:
        pass
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
            summary = "用户分群统计:\n" + summary_df.to_markdown(index=False)
            try:
                from pipeline.segment_naming import get_segment_names
                names = get_segment_names(segments)
                if names:
                    summary += "\n分群业务命名: " + ", ".join(
                        f"分群{k}({v})" for k, v in sorted(names.items())
                    )
            except Exception:
                pass
            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=summary_df.to_dict(orient="records"),
                summary=summary,
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


class SegmentGrowthInput(BaseModel):
    period1: str = Field(default="", description="起始月份(YYYY-MM),如'2016-02';空=倒数第二个快照")
    period2: str = Field(default="", description="结束月份(YYYY-MM),如'2016-03';空=最近快照")


def _pick_snapshot_periods(snaps: list, p1: str, p2: str):
    """按月份前缀挑两个快照(取该月最后一个);p 为空时用最近两期。"""
    if not p1 and not p2:
        return snaps[-2], snaps[-1], ""
    if not p1 or not p2:
        return None, None, "对比需要同时指定两个月份(period1/period2),或都不指定(对比最近两期)"

    def _find(prefix: str):
        matched = [s for s in snaps if s.timestamp.startswith(prefix)]
        return matched[-1] if matched else None

    s1, s2 = _find(p1), _find(p2)
    if s1 is None or s2 is None:
        available = sorted({s.timestamp[:7] for s in snaps})
        return None, None, f"指定月份无快照(现有: {available});请换月份或用默认的最近两期对比"
    return s1, s2, ""


class SegmentGrowthSkill(BaseSkill):
    name = "get_segment_growth"
    description = (
        "获取各用户分群的环比增长指标:销售额增长、转化率变化、GMV提升、人数与平均消费变化。"
        "对比两个时间段的快照;默认最近两期,可用 period1/period2 指定月份(如 '2016-02' vs '2016-03')。"
        "适用于'2月比3月怎么样'这类环比对比问题。"
    )
    input_schema = SegmentGrowthInput

    def execute(self, **kwargs) -> SkillResult:
        try:
            from pipeline.user_segmentation import load_snapshots, compare_snapshots
            from pipeline.profile import compute_segment_growth
            from pipeline.segment_naming import get_segment_names

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

            prev, curr, err = _pick_snapshot_periods(
                snaps, (kwargs.get("period1") or "").strip(),
                (kwargs.get("period2") or "").strip(),
            )
            if err:
                return SkillResult(status=SkillStatus.PARTIAL, data=[], summary=err, confidence=0.3)

            growth = compute_segment_growth(prev.segment_stats, curr.segment_stats)
            deltas = compare_snapshots(prev, curr)
            names = get_segment_names()

            for g in growth:
                d = deltas.get(g["segment"], {})
                g["user_count_delta"] = d.get("user_count_delta", 0)
                g["avg_monetary_delta"] = d.get("avg_monetary_delta", 0)

            def _label(seg):
                return names.get(int(seg), f"分群{seg}")

            table_lines = [
                f"## 分群环比对比( {prev.timestamp[:10]} → {curr.timestamp[:10]} )",
                "",
                "| 分群 | 销售额增长(%) | 转化率变化(%) | GMV提升(%) | 用户数 | 人均消费 | 人数变化 |",
                "|------|--------------|--------------|-----------|--------|----------|---------|",
            ]
            for g in growth:
                table_lines.append(
                    f"| {_label(g['segment'])} | {g['sales_growth_pct']:+.1f} | "
                    f"{g['conversion_change_pct']:+.1f} | {g['gmv_lift_pct']:+.1f} | "
                    f"{g['user_count']} | ¥{g['avg_monetary']:.2f} | {g['user_count_delta']:+.0f} |"
                )

            # 环比明细(供 Agent 引用,数字来自 compare_snapshots,可被数值核查)
            table_lines.append("")
            table_lines.append("### 分群明细变化")
            for g in growth:
                table_lines.append(
                    f"- {_label(g['segment'])}: 人数 {g['user_count'] - g['user_count_delta']:.0f} → "
                    f"{g['user_count']:.0f} ({g['user_count_delta']:+.0f}), "
                    f"平均消费 {g['avg_monetary'] - g['avg_monetary_delta']:.2f} → "
                    f"{g['avg_monetary']:.2f} ({g['avg_monetary_delta']:+.2f})"
                )

            # Data commentary
            table_lines.append("")
            table_lines.append("### 数据说明")
            best_sales = max(growth, key=lambda x: x['sales_growth_pct'])
            best_gmv = max(growth, key=lambda x: x['gmv_lift_pct'])
            table_lines.append(
                f"- **{_label(best_sales['segment'])}** 的销售额增长最突出"
                f"（{best_sales['sales_growth_pct']:+.1f}%）。"
            )
            table_lines.append(
                f"- **{_label(best_gmv['segment'])}** 的 GMV 提升比例最高"
                f"（{best_gmv['gmv_lift_pct']:+.1f}%），是最具商业增长潜力的群体。"
            )
            flat_segs = [g for g in growth if g['conversion_change_pct'] < 0]
            if flat_segs:
                table_lines.append(
                    f"- { '、'.join(_label(g['segment']) for g in flat_segs) } 的转化率负增长，"
                    f"建议排查流失原因并制定挽回策略。"
                )

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=growth,
                summary="\n".join(table_lines),
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


def _segment_keyword(kw: str) -> list[str]:
    """关键词切词:jieba 分词(命中 ≥2 字词);非中文串退回分隔符切。"""
    import re as _re
    try:
        import jieba
        words = [w for w in jieba.lcut(kw) if len(w) >= 2]
        if words:
            return words
    except Exception:
        pass
    return [t for t in _re.split(r"[\s\-/、]+", kw) if len(t) >= 2]


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
                cat_matches = [p for p in matches if kw in p.get("category", "").lower()]
                # 中文复合词整串常命不中合成商品名("降噪耳机" vs "耳机-36692")
                # → token 级兜底:jieba 分词,任一词命中即可(全词优先)
                tokens = _segment_keyword(kw)
                token_matches = []
                if not name_matches and not cat_matches and len(tokens) > 1:
                    token_matches = [
                        p for p in matches
                        if any(t in p["product_name"].lower() for t in tokens)
                        or any(t in p.get("category", "").lower() for t in tokens)
                    ]
                if name_matches:
                    matches = name_matches
                elif cat_matches:
                    matches = cat_matches
                elif token_matches:
                    matches = token_matches
                else:
                    # 无任何匹配 → 空结果(否则会带着全量商品返回 success,
                    # LLM 误以为"找到相关商品",从全量列表编推荐 → 幻觉)
                    matches = []

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

class SegmentTrendSkill(BaseSkill):
    name = "get_segment_trend"
    description = (
        "获取各分群人数随时间的变化趋势(历史快照时间序列)。"
        "返回每个快照时间点各分群的人数与平均消费。"
        "适用于'分群人数趋势''最近分群有什么变化'类问题。"
    )

    def execute(self) -> SkillResult:
        try:
            from pipeline.user_segmentation import load_snapshots
            from pipeline.segment_naming import get_segment_names

            _load_and_process(force_refresh=False)
            snaps = load_snapshots()
            if len(snaps) < 2:
                return SkillResult(
                    status=SkillStatus.PARTIAL, data=[],
                    summary="快照不足(需≥2个),等 Watcher 轮询几轮后再问趋势。",
                    confidence=0.3,
                )

            # 结构化时序行(前端折线图直接消费)
            rows = []
            for s in snaps[-10:]:
                ts = s.timestamp[:16].replace("T", " ")
                for sid in sorted(s.segment_stats.keys()):
                    st = s.segment_stats[sid]
                    rows.append({
                        "timestamp": ts,
                        "segment": int(sid),
                        "user_count": st.get("user_count", 0),
                        "avg_monetary": round(st.get("avg_monetary", 0), 2),
                    })

            # 摘要:首末快照的每分群人数变化(业务名标注)
            first, last = snaps[0], snaps[-1]
            lines = [
                f"分群人数趋势(快照 {len(snaps)} 个,"
                f"{first.timestamp[:10]} → {last.timestamp[:10]}):",
            ]
            names = get_segment_names()
            for sid in sorted(last.segment_stats.keys()):
                n0 = first.segment_stats.get(sid, {}).get("user_count", 0)
                n1 = last.segment_stats.get(sid, {}).get("user_count", 0)
                label = names.get(int(sid), f"分群{sid}")
                lines.append(f"- {label}: {n0} → {n1} 人 ({n1 - n0:+})")
            return SkillResult(
                status=SkillStatus.SUCCESS, data=rows,
                summary="\n".join(lines), confidence=0.9,
            )
        except (DatabaseError, ComputationError) as e:
            return SkillResult(status=SkillStatus.ERROR, error=str(e), confidence=0.0)


def register_all_skills() -> None:
    """Register all built-in skills with the global registry.

    Skill engineering 主路径:优先从 ``skills/definitions/*/SKILL.md`` 声明式注册
    (加载器按 frontmatter 覆盖 group/version/tags/description);
    定义文件缺失时回退到程序化注册(向后兼容,行为等同旧版)。
    """
    from skills import SkillRegistry
    from skills.loader import register_from_definitions

    registered = set(register_from_definitions())

    # ── 兜底:未提供 SKILL.md 的 Skill 仍按旧方式注册 ──
    fallback_classes = [
        ProductSearchSkill, GetCategoriesSkill,
        UserSegmentStatsSkill, UserSegmentRulesSkill,
        HighValueUsersSkill, SegmentGrowthSkill, SegmentTrendSkill,
        RefreshPipelineSkill,
    ]
    for cls in fallback_classes:
        if cls.name not in registered:
            SkillRegistry.register(cls())

    # ── 转化漏斗(行为流 → 浏览/加购/下单 逐级转化率)──
    from skills.funnel import FunnelAnalysisSkill
    if FunnelAnalysisSkill.name not in registered:
        SkillRegistry.register(FunnelAnalysisSkill())

    # ── VL 扩展:商品图像解析(group=vision,不进 Agent 工具绑定名单)──
    from skills.product_image import ProductImageSkill
    if ProductImageSkill.name not in registered:
        SkillRegistry.register(ProductImageSkill())

    # ── 个性化推荐(画像驱动,进入分析模式工具集)──
    from skills.recommend import PersonalRecommendationSkill
    if PersonalRecommendationSkill.name not in registered:
        SkillRegistry.register(PersonalRecommendationSkill())
    logger.info("all_skills_registered",
                extra={"from_definitions": len(registered)})
