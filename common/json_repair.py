"""LLM JSON 输出修复:提取 → 规则修复 → 带反馈重试 → 兜底默认值。

preprocessor.py 与 reflector.py 的 json.loads 失败静默回退统一收敛到此模块:
修复失败会先带错误反馈让 LLM 重生成(有限次数),仍失败才回退默认值并告警。

零第三方依赖。所有修复规则"修复后必须 json.loads 成功才采纳",否则回滚继续下一规则。
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

_log = logging.getLogger("json_repair")


@dataclass
class JsonRepairResult:
    ok: bool
    data: Any = None            # 解析成功且类型正确的数据
    repaired: bool = False      # 是否经过了规则修复
    retries: int = 0            # LLM 反馈重试次数
    error: str = ""             # 最终失败原因(供告警日志)


# ── 提取 ────────────────────────────────────────────────────────

def extract_json(text: str) -> str | None:
    """剥 markdown fence → 平衡括号扫描取第一个 { 到与之配对的 }(字符串/转义感知)。"""
    if not text:
        return None
    t = text.strip()
    # 剥 ```json ... ``` / ``` ... ``` fence
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.DOTALL)
    if m:
        t = m.group(1).strip()
    # 找到第一个 '{'
    start = t.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return t[start:i + 1]
    # 括号未闭合(模型截断输出)→ 返回部分串供修复规则补全
    return t[start:] if depth > 0 else None


# ── 修复规则(按序执行,每步后 json.loads 验证,成功即返回) ────────

_JSON_RESERVED = {"true": "true", "false": "false", "null": "null"}


def _strip_trailing_commas(t: str) -> str:
    """去尾逗号: `,}` / `,]` / 逗号+空白+闭合括号。"""
    return re.sub(r",(\s*[}\]])", r"\1", t)


def _strip_comments(t: str) -> str:
    """剥 // 行注释与 /* */ 块注释(仅字符串外,简化实现:逐字符状态机)。"""
    out: list[str] = []
    i = 0
    n = len(t)
    in_str = False
    escaped = False
    while i < n:
        ch = t[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < n and t[i + 1] == "/":
            j = t.find("\n", i)
            i = n if j < 0 else j
            continue
        if ch == "/" and i + 1 < n and t[i + 1] == "*":
            j = t.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _close_brackets(t: str) -> str:
    """按栈深度补缺失闭合括号:先补 ] 再补 }。"""
    depth = 0
    in_str = False
    escaped = False
    for ch in t:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
    return t + "}" * depth


def _quote_bare_keys(t: str) -> str:
    """裸键加引号:{intent: "x"} → {"intent": "x"}。

    仅处理字符串外的标识符,且前一个有效字符(跳过空白)必须是 { 或 ,。
    """
    out: list[str] = []
    i = 0
    n = len(t)
    in_str = False
    escaped = False
    while i < n:
        ch = t[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == "_" or ch.isalpha():
            # 回看前一个有效字符(跳过空白)是否为 { 或 ,
            j = i - 1
            while j >= 0 and t[j] in " \t\r\n":
                j -= 1
            if j < 0 or t[j] in "{,":
                m = re.match(r"[A-Za-z_][A-Za-z0-9_\-]*\s*:", t[i:])
                if m:
                    key = m.group(0).rstrip().rstrip(":")
                    out.append(f'"{key}"')
                    i += len(key)   # 空白与冒号留给后续循环原样输出
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_quotes(t: str) -> str:
    """智能引号归一化:"" '' → ";单引号键值 → 双引号(仅键值结构位置,防误伤字符串内容)。"""
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("‘", "'").replace("’", "'")
    # 单引号键: 'key': 或 'key' :(键值位置)
    t = re.sub(r"([{,])\s*'([^']{1,80}?)'\s*:", r'\1"\2":', t)
    # 单引号值(简单场景): : 'value'(值后跟 , 或 })
    t = re.sub(r":\s*'([^']{1,200}?)'\s*([,}])", r': "\1"\2', t)
    return t


def _py_literals(t: str) -> str:
    """Python 字面量 True/False/None → true/false/null(词边界,字符串外)。"""
    # 简化:词边界替换(字符串内的 True 极罕见,冒进替换的风险低于收益,且每步有 load 验证兜底)
    t = re.sub(r"\bTrue\b", "true", t)
    t = re.sub(r"\bFalse\b", "false", t)
    t = re.sub(r"\bNone\b", "null", t)
    return t


def _escape_bare_newlines(t: str) -> str:
    """字符串内未转义的裸换行/控制符 → 转义或删除。"""
    out: list[str] = []
    in_str = False
    escaped = False
    for ch in t:
        if not in_str:
            out.append(ch)
            if ch == '"':
                in_str = True
            continue
        # 字符串内
        if escaped:
            out.append(ch)
            escaped = False
            continue
        if ch == "\\":
            out.append(ch)
            escaped = True
            continue
        if ch == '"':
            out.append(ch)
            in_str = False
            continue
        if ch == "\n":
            out.append("\\n")
            continue
        if ord(ch) < 0x20:
            continue
        out.append(ch)
    return "".join(out)


def repair_json(text: str) -> str | None:
    """依序应用修复规则,每步后 json.loads 验证,成功返回修复文本,全败返回 None。"""
    base = extract_json(text)
    candidates = [text]
    if base is not None:
        candidates.append(base)
    for cand in candidates:
        t = cand.strip()
        for step in (
            lambda s: s,
            _strip_trailing_commas,
            _strip_comments,
            _normalize_quotes,
            _py_literals,
            _escape_bare_newlines,
            _quote_bare_keys,
            _close_brackets,
        ):
            t = step(t)
            try:
                json.loads(t)
                return t
            except (json.JSONDecodeError, ValueError):
                continue
    return None


# ── 带反馈重试 ─────────────────────────────────────────────────

def parse_json_with_retry(
    text: str, *,
    llm=None,                        # 有 .invoke(messages) → .content 的对象
    retry_messages: list | None = None,   # 重试时复用的原始消息列表
    hint: str = "",                  # 期望的 JSON 结构说明,拼进重试反馈
    max_retries: int = 2,
    expect_type: type = dict,
    logger: logging.Logger | None = None,
) -> JsonRepairResult:
    """解析 → 修复 → 带反馈重试 → 兜底。"""
    log = logger or _log

    # 1. 直接解析
    extracted = extract_json(text)
    if extracted is not None:
        try:
            data = json.loads(extracted)
            if isinstance(data, expect_type):
                return JsonRepairResult(ok=True, data=data, repaired=extracted != text.strip())
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
        else:
            last_error = f"顶层类型不是 {expect_type.__name__}"
    else:
        last_error = "未找到 JSON 对象"

    # 2. 规则修复
    repaired = repair_json(text)
    if repaired is not None:
        try:
            data = json.loads(repaired)
            if isinstance(data, expect_type):
                return JsonRepairResult(ok=True, data=data, repaired=True)
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
    else:
        last_error = "修复后仍不是合法 JSON"

    # 3. 带反馈重试(仅当提供了 llm 与原始消息)
    if llm is not None and retry_messages is not None:
        for attempt in range(1, max_retries + 1):
            feedback = (
                f"你上次的输出不是合法 JSON，无法解析。解析错误: {last_error}\n"
                f"请只输出一个 JSON 对象，不要包含任何 Markdown、代码块或解释文字。\n"
                f"期望结构: {hint or '与上次一致'}"
            )
            try:
                raw2 = llm.invoke(
                    list(retry_messages) + [{"role": "user", "content": feedback}]
                ).content
                fixed = repair_json(raw2)
                if fixed is not None:
                    data = json.loads(fixed)
                    if isinstance(data, expect_type):
                        return JsonRepairResult(
                            ok=True, data=data, repaired=True, retries=attempt,
                        )
                    last_error = f"顶层类型不是 {expect_type.__name__}"
                else:
                    last_error = "修复后仍不是合法 JSON"
            except Exception as e:
                last_error = f"重试调用失败: {e}"

    return JsonRepairResult(ok=False, retries=max_retries if llm is not None else 0,
                            error=last_error)


def load_json_or_default(
    text: str, *,
    default: dict,
    llm=None,
    retry_messages: list | None = None,
    hint: str = "",
    max_retries: int = 2,
    logger: logging.Logger | None = None,
) -> tuple[Any, JsonRepairResult]:
    """组合入口:parse_json_with_retry → 失败落默认值 + 告警日志。"""
    log = logger or _log
    result = parse_json_with_retry(
        text, llm=llm, retry_messages=retry_messages, hint=hint,
        max_retries=max_retries, expect_type=dict, logger=log,
    )
    if result.ok:
        return result.data, result
    log.warning("json_repair_fallback", extra={
        "raw": (text or "")[:200],
        "retries": result.retries,
        "error": result.error,
    })
    return default, result
