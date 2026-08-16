"""简易 Prompt 注入检测(规则式,零 LLM 调用)。

加权规则:每条命中 +2~3 分,总分 ≥ PROMPT_GUARD_BLOCK_SCORE → 拦截。
覆盖中英文"忽略指令 / 角色越狱 / 索取系统提示词 / 绕过限制"等常见模式。
与康养项目 guardrails.py 同一设计(规则独立维护,评分机制一致)。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config.settings import get_settings
from log.logger import get_logger

logger = get_logger(__name__)

# (正则, 权重, 说明)——强攻击模式(覆盖指令/越狱/索取提示词)单发即达拦截阈值 4,
# 弱信号(身份剥离/绕过限制)仅累计加分,单发只告警放行。
_GUARD_RULES: list[tuple[re.Pattern, int, str]] = [
    (re.compile(r"忽略(以上|之前|此前)?(所有)?(的)?(指令|指示|要求|规则|系统提示|约束)", re.I), 4, "覆盖系统指令"),
    (re.compile(r"(ignore|disregard|forget|bypass)\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)", re.I), 4, "忽略先前指令(英文)"),
    (re.compile(r"(你(现在)?是|扮演|角色扮演|进入)\s*(DAN|越狱|开发者模式|developer mode|无限制模式)", re.I), 4, "角色越狱"),
    (re.compile(r"(泄露|泄漏|打印|输出|展示).{0,12}(系统提示词|system prompt|内部指令|隐藏指令|原始指令)", re.I), 3, "索取系统提示词"),
    (re.compile(r"(假装|装作|假设|从现在开始)你(没有|不再|不是|忘记).{0,12}(助手|模型|AI|限制)", re.I), 2, "身份剥离"),
    (re.compile(r"(无视|绕过|跳过)(所有|一切)?(限制|规则|安全|约束|审核)", re.I), 2, "绕过安全限制"),
    (re.compile(r"(用|换成|翻译成).{0,6}(另一种语言|其他语言|英文).{0,12}(重复|输出|复述).{0,12}(系统提示|指令|规则)", re.I), 2, "多语言套取提示词"),
]


@dataclass
class InjectionVerdict:
    score: int = 0
    hits: list[str] = field(default_factory=list)
    blocked: bool = False

    @property
    def warned(self) -> bool:
        """有命中但未达拦截阈值。"""
        return self.score > 0 and not self.blocked


def detect_injection(text: str) -> InjectionVerdict:
    """对用户输入做注入检测(纯规则)。"""
    settings = get_settings()
    if not settings.PROMPT_GUARD_ENABLED or not text:
        return InjectionVerdict()

    score = 0
    hits: list[str] = []
    for pattern, weight, label in _GUARD_RULES:
        if pattern.search(text):
            score += weight
            hits.append(label)

    blocked = score >= settings.PROMPT_GUARD_BLOCK_SCORE
    verdict = InjectionVerdict(score=score, hits=hits, blocked=blocked)
    if blocked:
        logger.warning("injection_blocked", extra={"score": score, "hits": hits})
    elif hits:
        logger.info("injection_flagged", extra={"score": score, "hits": hits})
    return verdict


WARN_NOTICE = "\n\n---\n> ⚠️ 系统提示: 检测到输入包含指令覆盖迹象,已按平台规则处理。如为误报请换一种说法。"
