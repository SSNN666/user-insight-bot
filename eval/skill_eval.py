"""Skill 评测 — 对每个工具型 Skill 跑代表性查询,断言状态/置信度/耗时。

零 LLM 调用:直接调 ``skill.execute()``,验证数据面/快照/缓存/参数校验的
确定性行为。覆盖正例、降级例(如"指定月份无快照"→ PARTIAL)、空结果例。

用法::

    uv run python -m eval.skill_eval            # 控制台表格
    uv run python -m eval.skill_eval --json     # 额外写 eval/results/skill_eval_latest.json

跳过项(如实标注,不静默):refresh_pipeline(全量重算,重且慢)、
analyze_product_image(需图片文件 + VL Key)。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# 确保 Skill 已注册(与 bridge 同款触发模式:tools 模块导入即注册)
from tools import tools as _registration_trigger  # noqa: F401
from skills import SkillRegistry

# ── 探针:每个 Skill 一组代表性查询(execute 参数直接透传)────────
PROBE_QUERIES: dict[str, list[dict]] = {
    "search_products": [
        {"keyword": "手机"},
        {"category": "电子产品"},
        {"keyword": "不存在的商品xyz"},          # 空结果 → PARTIAL
    ],
    "get_categories": [{}],
    "get_user_segment_stats": [{}],
    "get_segment_rules": [{}],
    "get_high_value_users": [{}],
    "get_segment_growth": [
        {},                                     # 默认最近两期
        {"period1": "1900-01", "period2": "1900-02"},   # 无快照 → 诚实降级 PARTIAL
    ],
    "get_segment_trend": [{}],
    "get_funnel_analysis": [{"days": 7}],
    "get_personal_recommendations": [{"user_id": 1}],
}

# 跳过清单与原因(避免误判为失败)
SKIPPED = {
    "refresh_pipeline": "全量重算(RFM+聚类+快照),重且慢,留手动验证",
    "analyze_product_image": "需本地图片文件 + 多模态 Key",
}

ACCEPTED_STATUSES = {"success", "partial"}   # partial=诚实降级,视为通过
MIN_CONFIDENCE = 0.3                         # 成功探针的最低置信度门槛


def run_probes() -> dict:
    """逐 Skill 跑探针,返回结构化结果。"""
    registry = {s.name: s for s in SkillRegistry.list_all()}
    results: dict = {"skills": {}, "summary": {}}
    stats = {"pass": 0, "fail": 0, "skipped": 0}

    for name, probes in sorted(PROBE_QUERIES.items()):
        skill = registry.get(name)
        if skill is None:
            stats["fail"] += 1
            results["skills"][name] = {
                "status": "not_registered", "detail": "注册表中不存在", "probes": [],
            }
            continue
        probe_results = []
        for kwargs in probes:
            t0 = time.monotonic()
            try:
                r = skill.execute(**kwargs)
                elapsed = round((time.monotonic() - t0) * 1000, 1)
                ok = (r.status.value in ACCEPTED_STATUSES
                      and (r.status != "success" or r.confidence >= MIN_CONFIDENCE)
                      and (r.status != "success" or r.data is not None))
                probe_results.append({
                    "kwargs": kwargs,
                    "status": r.status.value,
                    "confidence": r.confidence,
                    "elapsed_ms": elapsed,
                    "pass": ok,
                    "summary": (r.summary or r.error or "")[:80],
                })
            except Exception as e:  # noqa: BLE001 — 探针必须捕获并报告
                probe_results.append({
                    "kwargs": kwargs,
                    "status": "raised",
                    "confidence": 0.0,
                    "elapsed_ms": 0.0,
                    "pass": False,
                    "summary": f"执行抛异常: {str(e)[:120]}",
                })
        ok_all = all(p["pass"] for p in probe_results)
        stats["pass" if ok_all else "fail"] += 1
        results["skills"][name] = {
            "status": "pass" if ok_all else "fail",
            "probes": probe_results,
        }

    for name, reason in sorted(SKIPPED.items()):
        stats["skipped"] += 1
        results["skills"][name] = {"status": "skipped", "detail": reason, "probes": []}

    results["summary"] = stats
    return results


def render_report(results: dict) -> str:
    lines = ["# Skill 评测报告", ""]
    lines.append(f"| Skill | 结果 | 探针 |")
    lines.append(f"|-------|------|------|")
    for name, r in sorted(results["skills"].items()):
        if r["status"] == "skipped":
            lines.append(f"| {name} | [SKIP] | {r.get('detail', '')} |")
            continue
        detail = "; ".join(
            f"{p['kwargs'] or '{}'}->{p['status']}({p['elapsed_ms']}ms)"
            for p in r["probes"]
        ) or r.get("detail", "")
        mark = "[PASS]" if r["status"] == "pass" else "[FAIL]"
        lines.append(f"| {name} | {mark} | {detail} |")
    s = results["summary"]
    lines += [
        "",
        f"**汇总**: {s['pass']} 通过 / {s['fail']} 失败 / {s['skipped']} 跳过",
        "> 说明:partial 视为通过(诚实降级,如'指定月份无快照');所有探针零 LLM 调用。",
    ]
    return "\n".join(lines)


def main() -> int:
    results = run_probes()
    report = render_report(results)
    print(report)

    if "--json" in sys.argv:
        out = Path(__file__).parent / "results" / "skill_eval_latest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"\n已写 {out}")

    return 1 if results["summary"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
