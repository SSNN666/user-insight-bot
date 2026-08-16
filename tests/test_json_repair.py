"""common/json_repair 单测:提取 / 规则修复 / 重试兜底。"""
import pytest

from common.json_repair import (
    extract_json, repair_json, parse_json_with_retry, load_json_or_default,
)


class TestExtractJson:
    def test_plain(self):
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_fence(self):
        text = '```json\n{"intent": "user_segment"}\n```'
        assert extract_json(text) == '{"intent": "user_segment"}'

    def test_leading_noise(self):
        assert extract_json('前导文字 {"x": 1}') == '{"x": 1}'

    def test_unbalanced_returns_partial(self):
        assert extract_json('前导文字 {"x": 1') == '{"x": 1'

    def test_brace_inside_string(self):
        text = '{"b": "中文}内容"}'
        assert extract_json(text) == text

    def test_no_json(self):
        assert extract_json('完全没有括号') is None


class TestRepairJson:
    @pytest.mark.parametrize("text,expected", [
        ('```json\n{"intent":"user_segment",}\n```', {"intent": "user_segment"}),
        ('{intent: "user_segment", entities: {}}', {"intent": "user_segment", "entities": {}}),
        ('{"is_complete": True, "confidence": 0.9,}', {"is_complete": True, "confidence": 0.9}),
        ('前导文字 {"x": 1', {"x": 1}),
        ('{"q": [1, 2], "w": null}', {"q": [1, 2], "w": None}),
        ('{"s": "线1\n线2"}', {"s": "线1\n线2"}),
        ("{'k': 'v'}", {"k": "v"}),
    ])
    def test_repair(self, text, expected):
        import json as _json
        repaired = repair_json(text)
        assert repaired is not None
        assert _json.loads(repaired) == expected

    def test_unrepairable(self):
        assert repair_json('这不是 JSON 也不含括号') is None


class TestParseWithRetry:
    def test_direct_success(self):
        r = parse_json_with_retry('{"a": 1}')
        assert r.ok and r.data == {"a": 1} and r.repaired is False

    def test_repaired_success(self):
        r = parse_json_with_retry('{"a": 1,}')
        assert r.ok and r.repaired is True

    def test_wrong_top_level_type(self):
        r = parse_json_with_retry('[1, 2, 3]')
        assert not r.ok

    def test_no_llm_fallback(self):
        r = parse_json_with_retry('垃圾输出')
        assert not r.ok and r.retries == 0

    def test_retry_with_llm(self):
        class _FakeLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages):
                self.calls += 1
                # 第一次重试给坏 JSON,第二次给好 JSON
                if self.calls == 1:
                    from langchain_core.messages import AIMessage
                    return AIMessage(content='还是不对 {broken')
                from langchain_core.messages import AIMessage
                return AIMessage(content='{"is_complete": true}')

        llm = _FakeLLM()
        r = parse_json_with_retry(
            '{"is_complete":',
            llm=llm,
            retry_messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}],
            hint='{"is_complete": true/false}',
            max_retries=2,
        )
        assert r.ok and r.data == {"is_complete": True} and r.retries == 2

    def test_load_json_or_default(self):
        data, r = load_json_or_default('垃圾', default={"fallback": 1})
        assert data == {"fallback": 1} and not r.ok
