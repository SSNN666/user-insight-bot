"""分群运营建议:LLM 基于真实分群数据生成建议,再经数值核查。

流程(与 Agent 回答同一条信任链):
  1. 取图表同源数据:分群统计(中文键,与 get_user_segment_stats 同口径)
     + 流转分布 + 最近趋势快照;
  2. LLM(role=suggest)按"只用给定数据、严禁编造"生成分群级运营建议;
  3. 建议文本过 Layer 1 数值核查(复用 fact_checker,零 LLM 调用);
  4. 有违规 → 附修正提示重生成一次;仍有违规 → grounded=False 如实返回。

设计要点:
  - 建议里的数字与图表数据同源且可核查——"建议可信"和"回答可信"用同一套保证;
  - _call_llm 是独立薄包装,测试可直接替换(零真实 LLM 调用)。
"""

import json

from agent.fact_checker import run_fact_check
from llm.bridge import get_fallback_llm
from log.logger import get_logger
from skills.base import SkillResult, SkillStatus

logger = get_logger(__name__)

SUGGEST_SYSTEM_PROMPT = (
    "你是电商运营数据分析师。基于给定的用户分群统计、流转分布与趋势快照数据,"
    "为每个用户分群给出运营建议。\n\n"
    "要求:\n"
    "1. 只使用下方数据中的数字,严禁编造或外推不存在的数据;\n"
    "2. 每个分群一段:一句现状总结 + 1-2 条可执行动作 + 预期效果;\n"
    "3. Markdown 输出,数字与数据一致(元保留 2 位小数);\n"
    "4. 结尾给一条整体运营建议。"
)


def _build_stats_rows(seg) -> list[dict]:
    """分群统计行(中文键,与 get_user_segment_stats Skill 同口径,供核查复用)。"""
    rows = []
    for seg_id in sorted(seg["segment"].unique()):
        sdf = seg[seg["segment"] == seg_id]
        rows.append({
            "segment": int(seg_id),
            "用户数": int(len(sdf)),
            "平均近度": round(float(sdf["recency"].mean()), 1),
            "平均频次": round(float(sdf["frequency"].mean()), 2),
            "平均消费": round(float(sdf["monetary"].mean()), 2),
        })
    return rows


def _build_flow_summary(seg) -> dict[int, dict]:
    flow: dict[int, dict] = {}
    if "flow_tag" in seg.columns:
        for seg_id in sorted(seg["segment"].unique()):
            flow[int(seg_id)] = (
                seg[seg["segment"] == seg_id]["flow_tag"].value_counts().to_dict()
            )
    return flow


def _build_trend_rows(limit: int = 6) -> list[dict]:
    from pipeline.user_segmentation import load_snapshots
    rows = []
    for s in load_snapshots()[-limit:]:
        ts = s.timestamp[:16].replace("T", " ")
        for sid in sorted(s.segment_stats.keys()):
            st = s.segment_stats[sid]
            rows.append({
                "timestamp": ts,
                "segment": int(sid),
                "user_count": st.get("user_count", 0),
                "avg_monetary": round(st.get("avg_monetary", 0), 2),
            })
    return rows


def build_suggestion_context(seg, limit: int = 6) -> str:
    """图表同源数据 → prompt 上下文 JSON(纯函数,可单测)。"""
    context = {
        "分群统计": _build_stats_rows(seg),
        "流转分布": _build_flow_summary(seg),
        "趋势快照(最近)": _build_trend_rows(limit),
    }
    try:
        from pipeline.segment_naming import get_segment_names
        names = get_segment_names(seg)
        if names:
            context["分群业务命名"] = {str(k): v for k, v in sorted(names.items())}
    except Exception:
        pass
    return json.dumps(context, ensure_ascii=False, indent=2)


def _call_llm(messages: list[dict]) -> str:
    """调用 suggest 角色 LLM(独立薄包装,测试可直接替换)。"""
    llm = get_fallback_llm("suggest")
    resp = llm.invoke(messages)
    return (resp.content or "").strip()


def _fact_check_suggestions(text: str, rows: list[dict]):
    """建议文本 → Layer 1 数值核查(与 Agent 回答共用同一套规则)。"""
    stats_sr = SkillResult(
        status=SkillStatus.SUCCESS, data=rows,
        summary="分群统计", confidence=1.0,
    )
    return run_fact_check(text, [stats_sr], "relaxed")


def generate_segment_suggestions(seg=None, max_retries: int = 1) -> dict:
    """生成分群级运营建议并做数值核查。

    Args:
        seg: 分群 DataFrame(含 segment/recency/frequency/monetary);
             None 时走 _load_and_process(与图表同源)。
        max_retries: 核查失败后的重生成次数上限。

    Returns:
        {"suggestions": str, "grounded": bool, "violations": int, "retries": int}
    """
    if seg is None:
        from skills.user_segment import _load_and_process
        _, seg, _ = _load_and_process(force_refresh=False)

    rows = _build_stats_rows(seg)
    context = build_suggestion_context(seg)
    messages = [
        {"role": "system", "content": SUGGEST_SYSTEM_PROMPT},
        {"role": "user", "content": f"数据如下:\n{context}"},
    ]

    text = _call_llm(messages)
    result = None
    for attempt in range(max_retries + 1):
        result = _fact_check_suggestions(text, rows)
        if result.passed or attempt == max_retries:
            break
        logger.info("suggestion_fact_check_failed", extra={
            "violations": len(result.violations),
            "retry": attempt + 1,
        })
        text = _call_llm(messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": result.format_corrections_for_llm()},
        ])

    assert result is not None
    return {
        "suggestions": text,
        "grounded": result.passed,
        "violations": len(result.violations),
        "retries": attempt,
    }
