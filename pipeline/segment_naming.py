"""分群业务命名:运营说"高价值用户",不说"分群2"。

流程:
  1. 拿分群统计(人数/近度/频次/消费 + 流转分布);
  2. LLM(role=suggest)生成业务名,JSON 输出,经 json_repair 容错解析;
  3. LLM 不可用/输出非法 → 确定性启发式兜底(价值分层 × 流转主标签);
  4. 结果按"统计行哈希 + TTL"缓存,与管线缓存生命周期一致。

启发式兜底保证:任何情况下图表/对话都能说人话,不依赖云端。
"""
import hashlib
import json
import threading
import time

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

# ── 缓存(进程级,与 pipeline 缓存同 TTL) ──────────────────────

_names_cache: dict = {}
_names_lock = threading.RLock()

NAMING_SYSTEM_PROMPT = (
    "你是电商运营专家。根据用户分群统计特征,为每个分群取一个简洁的中文业务名"
    "(6-10 个字,如'高价值核心用户''低活潜力用户'),便于运营人员理解。\n"
    "规则:结合消费水平与流转状态,名字要能体现该分群的核心特征,不同分群名字不可重复。\n"
    '只输出 JSON:{"分群编号": "业务名"}。'
)

# 价值分层(按平均消费排名)与流转主标签的确定性组合
_VALUE_TIERS = {0: "低价值", 1: "中坚", 2: "高价值"}
_FLOW_LABELS = {
    "active": "活跃用户", "potential": "潜力用户", "stable": "稳定用户",
    "dormant": "沉睡用户", "churned": "流失用户",
}


def _stats_fingerprint(rows: list[dict]) -> str:
    return hashlib.md5(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]


def heuristic_names(seg) -> dict[int, str]:
    """确定性兜底:平均消费排名 × 流转主标签,零 LLM 依赖。"""
    names: dict[int, str] = {}
    if seg is None or seg.empty or "segment" not in seg.columns:
        return names
    seg_ids = sorted(seg["segment"].unique())
    if not seg_ids:
        return names

    monetary_rank = {
        int(sid): rank
        for rank, sid in enumerate(
            seg.groupby("segment")["monetary"].mean().sort_values().index
        )
    }
    for sid in seg_ids:
        tier = _VALUE_TIERS.get(monetary_rank.get(int(sid), 1), "用户")
        flow = "用户"
        if "flow_tag" in seg.columns:
            dominant = seg[seg["segment"] == sid]["flow_tag"].value_counts().idxmax()
            flow = _FLOW_LABELS.get(str(dominant), str(dominant))
        names[int(sid)] = f"{tier}{flow}"
    return names


def _call_naming_llm(context: str) -> str:
    """薄包装,测试可替换。"""
    from llm.bridge import get_fallback_llm
    llm = get_fallback_llm("suggest")
    resp = llm.invoke([
        {"role": "system", "content": NAMING_SYSTEM_PROMPT},
        {"role": "user", "content": context},
    ])
    return (resp.content or "").strip()


def _llm_names(seg) -> dict[int, str] | None:
    """LLM 命名;失败/解析失败/键不符 → None(走启发式)。"""
    from common.json_repair import extract_json, repair_json
    rows = []
    for sid in sorted(seg["segment"].unique()):
        sdf = seg[seg["segment"] == sid]
        row = {
            "分群": int(sid),
            "用户数": int(len(sdf)),
            "平均近度": round(float(sdf["recency"].mean()), 1),
            "平均频次": round(float(sdf["frequency"].mean()), 2),
            "平均消费": round(float(sdf["monetary"].mean()), 2),
        }
        if "flow_tag" in sdf.columns:
            row["流转分布"] = sdf["flow_tag"].value_counts().to_dict()
        rows.append(row)

    text = _call_naming_llm(
        "分群统计:\n" + json.dumps(rows, ensure_ascii=False, indent=2)
    )
    try:
        repaired = repair_json(extract_json(text) or "")
        obj = json.loads(repaired) if repaired else None
    except Exception:
        obj = None
    if not isinstance(obj, dict) or not obj:
        return None

    # 键必须是真实分群编号,值必须是短字符串
    valid_ids = {int(sid) for sid in seg["segment"].unique()}
    names: dict[int, str] = {}
    for k, v in obj.items():
        try:
            sid = int(k)
        except (TypeError, ValueError):
            continue
        if sid in valid_ids and isinstance(v, str) and 2 <= len(v.strip()) <= 20:
            names[sid] = v.strip()
    if len(names) != len(valid_ids):
        return None
    return names


def get_segment_names(seg=None, force: bool = False) -> dict[int, str]:
    """获取分群业务名(带缓存;seg 为 None 时走管线缓存)。

    Returns:
        {segment_id: 业务名};任何失败路径都返回启发式结果(可能为空 dict)。
    """
    settings = get_settings()
    if seg is None:
        from skills.user_segment import _load_and_process
        _, seg, _ = _load_and_process(force_refresh=False)

    if seg is None or seg.empty:
        return {}

    seg_ids = tuple(sorted(seg["segment"].unique()))
    fp = _stats_fingerprint(
        [{"segment": int(sid),
          "n": int((seg["segment"] == sid).sum()),
          "m": round(float(seg[seg["segment"] == sid]["monetary"].mean()), 2),
          "f": round(float(seg[seg["segment"] == sid]["frequency"].mean()), 2),
          "r": round(float(seg[seg["segment"] == sid]["recency"].mean()), 1),
          } for sid in seg_ids]
    )
    cache_key = (seg_ids, fp)

    with _names_lock:
        cached = _names_cache.get(cache_key)
        if not force and cached and time.time() - cached[1] < settings.PIPELINE_CACHE_TTL:
            return dict(cached[0])

    names = heuristic_names(seg)
    try:
        llm_names = _llm_names(seg)
        if llm_names:
            names = llm_names
            logger.info("segment_names_llm", extra={"names": names})
        else:
            logger.info("segment_names_heuristic", extra={"names": names})
    except Exception as e:
        logger.warning("segment_naming_llm_failed", extra={"error": str(e)[:150]})

    with _names_lock:
        _names_cache[cache_key] = (dict(names), time.time())
    return names


def invalidate_segment_names() -> None:
    with _names_lock:
        _names_cache.clear()
