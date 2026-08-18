"""转化漏斗 Skill:浏览 → 加购 → 下单 每步去重用户数与逐级转化率。

数据来源与 pipeline/funnel.py 保持一致(JData 真实行为 / 订单派生),
summary 附最大流失环节定位 + 数据新鲜度标注,图表由 chart_extract 透出。
"""

import pandas as pd
from pydantic import BaseModel, Field

from pipeline.funnel import load_funnel_actions, compute_funnel
from config.settings import get_settings
from skills.base import BaseSkill, SkillResult, SkillStatus
from log.logger import get_logger

logger = get_logger(__name__)


class FunnelAnalysisInput(BaseModel):
    days: int = Field(default=7, ge=1, le=90, description="分析窗口天数(最近N天),默认7")


class FunnelAnalysisSkill(BaseSkill):
    name = "get_funnel_analysis"
    description = (
        "获取最近N天 浏览→加购→下单 转化漏斗:每步去重用户数与逐级转化率。"
        "适用于'加购到下单的流失率''转化率为什么这么低''漏斗'类问题。"
        "参数 days: 窗口天数(1-90),默认7。"
    )
    input_schema = FunnelAnalysisInput

    def execute(self, days: int = 7) -> SkillResult:
        try:
            actions = load_funnel_actions()
            rows = compute_funnel(actions, days)
            if not rows:
                return SkillResult(
                    status=SkillStatus.PARTIAL,
                    data=[],
                    summary=(
                        f"近 {days} 天窗口内无行为数据,建议扩大窗口(days 参数) "
                        f"或先调用 refresh_pipeline 刷新数据。"
                    ),
                    confidence=0.3,
                )

            # ── 最大流失环节:流失率(= 1 - 转化率)最大的台阶 ──
            drops = []
            for prev, cur in zip(rows, rows[1:]):
                drops.append({
                    "from": prev["step"], "to": cur["step"],
                    "loss": round(1 - cur["conversion_rate"], 4),
                })
            worst = max(drops, key=lambda d: d["loss"]) if drops else None

            # ── 数据新鲜度标注(与 RFM recency 基准一致,不写系统今天)──
            settings = get_settings()
            source = ("JData 真实行为数据" if settings.DATA_SOURCE == "tianchi"
                      else "订单派生行为数据(合成)")
            end = actions["date"].max()
            start = end.normalize() - pd.Timedelta(days=days - 1)

            lines = [
                f"## 转化漏斗(近 {days} 天)",
                "",
                "| 步骤 | 去重用户数 | 转化率 | 流失率 |",
                "|------|-----------|--------|--------|",
            ]
            for r in rows:
                rate = f"{r['conversion_rate'] * 100:.1f}%"
                loss = f"{(1 - r['conversion_rate']) * 100:.1f}%"
                lines.append(
                    f"| {r['step']} | {r['user_count']:,} | {rate} | {loss} |"
                )
            if worst:
                lines.append("")
                lines.append(
                    f"> ⚠️ **最大流失环节:{worst['from']} → {worst['to']}**"
                    f"(流失率 {worst['loss'] * 100:.1f}%),建议优先排查该环节的用户体验与转化障碍。"
                )
            lines.append("")
            lines.append(
                f"_数据来源:{source} | 窗口:{start.date()} ~ {end.date()} | "
                f"口径:去重用户数(行为流),与分群 RFM 同基准_"
            )

            return SkillResult(
                status=SkillStatus.SUCCESS,
                data=rows,
                summary="\n".join(lines),
                confidence=0.92,
            )
        except Exception as e:
            return SkillResult(
                status=SkillStatus.ERROR,
                error=f"漏斗分析失败: {e}",
                confidence=0.0,
            )
