"""FactCheck 对抗基准 —— 量化三层规则反幻觉的检出能力。

为什么需要它
------------
``PORTFOLIO.md`` 曾声称「三层核查下事实准确率接近 100%」，但这是**未经测量**的表述，
且掩盖了两处真实边界：第 2 层只校验阈值「是否存在于决策树」，不校验「是否属于被引用的
分群」；第 3 层仅覆盖「分群存在性 + 高价值方向」两类启发式。本脚本用可复现的实测数字
替代该表述。

方法
----
正样本：按句式族程序化注入**已知错误**，每个样本标注「本应由哪一层检出」。
负样本：同一句式族的**正确表述**（含舍入词与改写变体），用于测量误报率。

统计口径
--------
- 逐层召回率：目标为该层的正样本中，该层确实报出违规的比例
- 总体拦截率：全部正样本中，至少一层报出违规的比例
- 误报率：负样本中被报出违规的比例（干净回答被冤枉的比例）
- 单次核查耗时：纯函数，无 LLM / 无数据库 / 无网络

用法
----
    python -m eval.factcheck_bench              # 打印报告
    python -m eval.factcheck_bench --json       # 同时写 eval/results/factcheck_bench.json
    python eval/factcheck_bench.py              # 直接运行亦可
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agent.fact_checker import CheckLayer, run_fact_check  # noqa: E402
from skills.base import SkillResult, SkillStatus  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"


# ══════════════════════════════════════════════════════════════════
# 基准数据（与真实 Skill 返回同构）
# ══════════════════════════════════════════════════════════════════

STATS_ROWS = [
    {"segment": 0, "用户数": 120, "平均近度": 8.5,  "平均频次": 1.8, "平均消费": 65.0},
    {"segment": 1, "用户数": 340, "平均近度": 21.0, "平均频次": 3.6, "平均消费": 180.5},
    {"segment": 2, "用户数": 95,  "平均近度": 12.0, "平均频次": 6.4, "平均消费": 430.0},
    {"segment": 3, "用户数": 45,  "平均近度": 4.0,  "平均频次": 9.2, "平均消费": 780.0},
]
TOTAL_USERS = sum(r["用户数"] for r in STATS_ROWS)  # 600

# 决策树阈值（供第 2 层校验）：30.50 / 3.00 / 300.00
RULES_TEXT = """|--- recency <= 30.50
|   |--- frequency <= 3.00
|   |   |--- class: 0
|   |--- frequency > 3.00
|   |   |--- monetary <= 300.00
|   |   |   |--- class: 1
|   |   |--- monetary > 300.00
|   |   |   |--- class: 2
|--- recency > 30.50
|   |--- class: 1
"""


def _skill_results() -> list[SkillResult]:
    return [
        SkillResult(status=SkillStatus.SUCCESS, data=STATS_ROWS,
                    summary="segment stats", confidence=0.95),
        SkillResult(status=SkillStatus.SUCCESS, data=RULES_TEXT,
                    summary="decision tree rules", confidence=0.9),
    ]


@dataclass
class Sample:
    text: str
    family: str                      # 句式族（用于逐族报告，暴露覆盖边界）
    target_layer: CheckLayer | None  # None = 负样本（干净回答）
    is_positive: bool


# ══════════════════════════════════════════════════════════════════
# 正样本：注入已知错误
# ══════════════════════════════════════════════════════════════════

def _bad_int(v: int, mult: int = 3) -> int:
    """明显错误的正整数（×mult 后加奇偏移，确保远离任何舍入容差）"""
    return v * mult + 17


def _bad_float(v: float, mult: float = 4.0) -> float:
    return round(v * mult + 9.9, 1)


def gen_positives() -> list[Sample]:
    out: list[Sample] = []

    for row in STATS_ROWS:
        seg = row["segment"]

        # ── 第 1 层：数值 ──────────────────────────────
        # 用户数（两种量级）
        for mult in (3, 5):
            out.append(Sample(f"分群 {seg} 有 {_bad_int(row['用户数'], mult)} 人。",
                              "L1_分群用户数", CheckLayer.NUMERICAL, True))
        # 平均值（三类指标）
        for metric in ("消费", "频次", "近度"):
            actual = row[f"平均{metric}"]
            out.append(Sample(f"分群 {seg} 的平均{metric}为 {_bad_float(actual)}。",
                              f"L1_平均{metric}", CheckLayer.NUMERICAL, True))
        # 占比
        actual_pct = round(row["用户数"] / TOTAL_USERS * 100)
        bad_pct = min(99, actual_pct * 3 + 11)
        out.append(Sample(f"分群 {seg} 占比 {bad_pct}%。",
                          "L1_占比", CheckLayer.NUMERICAL, True))
        # Markdown 表格行（仅计数列错，其余列正确）
        out.append(Sample(
            f"| {seg} | {_bad_int(row['用户数'])} | {row['平均近度']} "
            f"| {row['平均频次']} | {row['平均消费']} |",
            "L1_表格行", CheckLayer.NUMERICAL, True))

    # 全平台总数
    out.append(Sample(f"全平台共 {_bad_int(TOTAL_USERS)} 名用户。",
                      "L1_全平台总数", CheckLayer.NUMERICAL, True))
    out.append(Sample(f"全部用户合计 {_bad_int(TOTAL_USERS, 2)} 人。",
                      "L1_全平台总数", CheckLayer.NUMERICAL, True))
    # 倍数关系（实际 780/65 = 12 倍）
    out.append(Sample("分群 3 是分群 0 的 99 倍。",
                      "L1_倍数关系", CheckLayer.NUMERICAL, True))
    out.append(Sample("分群 2 是分群 0 的 47 倍。",
                      "L1_倍数关系", CheckLayer.NUMERICAL, True))
    # 差额关系（实际 340-120 = 220 人）
    out.append(Sample("分群 1 比 分群 0 多 999 人。",
                      "L1_差额关系", CheckLayer.NUMERICAL, True))
    out.append(Sample("分群 0 比 分群 1 多 500 人。",
                      "L1_差额关系", CheckLayer.NUMERICAL, True))

    # ── 第 2 层：阈值 ──────────────────────────────
    # 树中不存在、且远超 ±1.0 容差的阈值
    for bad_th in ("9999", "888", "0.5", "1500.0"):
        out.append(Sample(f"消费 > {bad_th} 的用户属于高价值群体。",
                          "L2_捏造阈值", CheckLayer.RULE, True))
    for bad_th in ("99.0",):
        out.append(Sample(f"近度 <= {bad_th} 的用户需要召回。",
                          "L2_捏造阈值", CheckLayer.RULE, True))

    # ── 第 3 层：逻辑 ──────────────────────────────
    # 方向矛盾：分群 3（最高段）平均消费 780 > 分群 0 的 65，故「高价值消费低」为假
    out.append(Sample("高价值用户的消费低。",
                      "L3_方向矛盾", CheckLayer.LOGIC, True))
    out.append(Sample("核心用户的频次少。",
                      "L3_方向矛盾", CheckLayer.LOGIC, True))
    # 不存在的分群
    out.append(Sample("分群 9 比 分群 0 消费高。",
                      "L3_不存在分群", CheckLayer.LOGIC, True))
    out.append(Sample("分群 1 比 分群 7 频次高。",
                      "L3_不存在分群", CheckLayer.LOGIC, True))

    return out


# ══════════════════════════════════════════════════════════════════
# 负样本：正确表述（含改写与舍入变体）
# ══════════════════════════════════════════════════════════════════

def gen_negatives() -> list[Sample]:
    out: list[Sample] = []

    for row in STATS_ROWS:
        seg = row["segment"]
        out.append(Sample(f"分群 {seg} 有 {row['用户数']} 人。",
                          "N_分群用户数", None, False))
        for metric in ("消费", "频次", "近度"):
            out.append(Sample(f"分群 {seg} 的平均{metric}为 {row[f'平均{metric}']}。",
                              f"N_平均{metric}", None, False))
        pct = round(row["用户数"] / TOTAL_USERS * 100, 1)
        out.append(Sample(f"分群 {seg} 占比 {pct}%。",
                          "N_占比", None, False))
        out.append(Sample(
            f"| {seg} | {row['用户数']} | {row['平均近度']} "
            f"| {row['平均频次']} | {row['平均消费']} |",
            "N_表格行", None, False))

    # 总数（含舍入词变体）
    out.append(Sample(f"全平台共 {TOTAL_USERS} 名用户。", "N_全平台总数", None, False))
    out.append(Sample(f"全平台共约 {TOTAL_USERS} 名用户。", "N_全平台总数", None, False))
    out.append(Sample(f"全部用户合计 {TOTAL_USERS} 人。", "N_全平台总数", None, False))
    # 倍数（实际 12 倍）、差额（实际 220 人）
    out.append(Sample("分群 3 是分群 0 的 12 倍。", "N_倍数关系", None, False))
    out.append(Sample("分群 1 比 分群 0 多 220 人。", "N_差额关系", None, False))
    # 树中存在的阈值（300.00 在容差内）
    out.append(Sample("消费 > 300 的用户属于高价值群体。", "N_合法阈值", None, False))
    out.append(Sample("近度 <= 30.5 的用户需要召回。", "N_合法阈值", None, False))
    # 方向正确
    out.append(Sample("高价值用户的消费高。", "N_方向正确", None, False))
    out.append(Sample("核心用户的频次多。", "N_方向正确", None, False))
    # 存在的分群
    out.append(Sample("分群 2 比 分群 0 消费高。", "N_存在分群", None, False))

    return out


# ══════════════════════════════════════════════════════════════════
# 执行与统计
# ══════════════════════════════════════════════════════════════════

def evaluate(samples: list[Sample], skill_results: list[SkillResult]) -> list[dict]:
    rows = []
    for s in samples:
        t0 = time.perf_counter()
        res = run_fact_check(s.text, skill_results, check_mode="relaxed")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        fired = {v.layer for v in res.violations}
        rows.append({
            "text": s.text,
            "family": s.family,
            "is_positive": s.is_positive,
            "target_layer": s.target_layer.value if s.target_layer else None,
            "detected": not res.passed,
            "layers_fired": sorted(l.value for l in fired),
            "target_hit": (s.target_layer in fired) if s.target_layer else None,
            "n_violations": len(res.violations),
            "elapsed_ms": round(elapsed_ms, 4),
        })
    return rows


def summarize(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["is_positive"]]
    neg = [r for r in rows if not r["is_positive"]]

    def _rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    # 逐层召回：分母 = 目标为该层的正样本数
    per_layer = {}
    for layer in ("layer_1", "layer_2", "layer_3"):
        targeted = [r for r in pos if r["target_layer"] == layer]
        hit = sum(1 for r in targeted if r["target_hit"])
        per_layer[layer] = {"n": len(targeted), "hit": hit, "recall": _rate(hit, len(targeted))}

    # 逐句式族：暴露覆盖边界（哪类错误检不出）
    per_family = {}
    for r in rows:
        d = per_family.setdefault(r["family"], {"n": 0, "hit": 0, "positive": r["is_positive"]})
        d["n"] += 1
        if r["is_positive"] and r["target_hit"]:
            d["hit"] += 1
        elif not r["is_positive"] and not r["detected"]:
            d["hit"] += 1  # 负样本「未误报」也算命中

    for fam, d in per_family.items():
        d["rate"] = _rate(d["hit"], d["n"])

    elapsed = [r["elapsed_ms"] for r in rows]
    return {
        "n_positive": len(pos),
        "n_negative": len(neg),
        "overall_interception_rate": _rate(sum(1 for r in pos if r["detected"]), len(pos)),
        "false_positive_rate": _rate(sum(1 for r in neg if r["detected"]), len(neg)),
        "per_layer_recall": per_layer,
        "per_family": per_family,
        "latency_ms": {
            "mean": round(statistics.mean(elapsed), 4),
            "p50": round(statistics.median(elapsed), 4),
            "max": round(max(elapsed), 4),
        },
    }


def report(summary: dict, rows: list[dict]) -> None:
    print("\n" + "=" * 68)
    print("        [FactCheck 对抗基准] 三层规则反幻觉检出能力")
    print("=" * 68)
    print(f"\n  正样本（注入已知错误）: {summary['n_positive']}")
    print(f"  负样本（正确表述）    : {summary['n_negative']}")
    print(f"\n  {'总体拦截率':<26} {summary['overall_interception_rate']:>8.2%}")
    print(f"  {'误报率（越低越好）':<26} {summary['false_positive_rate']:>8.2%}")

    print(f"\n{'-' * 68}")
    print(f"  {'逐层召回率':<26} {'命中/目标':>12} {'召回':>10}")
    print(f"{'-' * 68}")
    names = {"layer_1": "第1层 数值", "layer_2": "第2层 阈值", "layer_3": "第3层 逻辑"}
    for layer, d in summary["per_layer_recall"].items():
        print(f"  {names[layer]:<26} {d['hit']:>5}/{d['n']:<6} {d['recall']:>10.2%}")

    print(f"\n{'-' * 68}")
    print(f"  {'逐句式族（暴露覆盖边界）':<40} {'命中/样本':>10} {'比例':>8}")
    print(f"{'-' * 68}")
    for fam in sorted(summary["per_family"]):
        d = summary["per_family"][fam]
        flag = "" if d["rate"] == 1.0 else "   ← 有漏检/误报"
        print(f"  {fam:<40} {d['hit']:>4}/{d['n']:<5} {d['rate']:>8.1%}{flag}")

    lat = summary["latency_ms"]
    print(f"\n{'-' * 68}")
    print(f"  单次核查耗时: mean {lat['mean']}ms | p50 {lat['p50']}ms | max {lat['max']}ms")
    print(f"  （纯函数，零 LLM 调用 / 零数据库 / 零网络）")

    missed = [r for r in rows if r["is_positive"] and not r["target_hit"]]
    if missed:
        print(f"\n{'-' * 68}")
        print(f"  ⚠️ 漏检样本 ({len(missed)} 条，即覆盖边界):")
        for r in missed[:15]:
            print(f"    [{r['target_layer']}] {r['text'][:56]}")
        if len(missed) > 15:
            print(f"    ... 及其他 {len(missed) - 15} 条")

    fps = [r for r in rows if not r["is_positive"] and r["detected"]]
    if fps:
        print(f"\n  ⚠️ 误报样本 ({len(fps)} 条，干净回答被判违规):")
        for r in fps[:10]:
            print(f"    {r['text'][:56]}  -> {r['layers_fired']}")

    print("=" * 68 + "\n")


def main() -> None:
    skill_results = _skill_results()
    samples = gen_positives() + gen_negatives()
    rows = evaluate(samples, skill_results)
    summary = summarize(rows)
    report(summary, rows)

    if "--json" in sys.argv:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "benchmark": "factcheck_bench",
                "n_positive": summary["n_positive"],
                "n_negative": summary["n_negative"],
                "note": "纯确定性基准，零 LLM/数据库/网络；覆盖边界见 missed_samples",
            },
            "summary": summary,
            "samples": rows,
        }
        out = RESULTS_DIR / "factcheck_bench.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVED] {out}")


if __name__ == "__main__":
    main()
