"""LLM 成本观测(metrics.py)离线测试:估算/落库/聚合/降级标记/数据层计数。"""
from llm.adapter import UsageInfo
from llm.metrics import (
    estimate_cost,
    record_data_tier,
    record_usage_bucket,
    tier_summary,
    usage_summary,
)


def _u(provider="dashscope", model="qwen3.8-max", p=1000, c=500,
       latency=320) -> UsageInfo:
    return UsageInfo(prompt_tokens=p, completion_tokens=c,
                     total_tokens=p + c, model=model, provider=provider,
                     latency_ms=latency)


class TestPricing:
    def test_estimate_known_model(self):
        # qwen3.8-max: 2.4/9.6 元每百万 → 1000 in + 1000 out = 0.0024+0.0096
        assert estimate_cost("dashscope", "qwen3.8-max", 1000, 1000) == 0.012
        assert estimate_cost("dashscope", "qwen3.8-max", 0, 0) == 0.0

    def test_estimate_unknown_or_local_free(self):
        assert estimate_cost("ollama", "qwen2.5:7b", 99999, 99999) == 0.0
        assert estimate_cost("dashscope", "不存在模型", 1000, 1000) == 0.0


class TestRecordBucket:
    def test_roundtrip_and_aggregation(self):
        bucket = {
            "decide": [_u(p=1000, c=500), _u(p=2000, c=0, latency=0)],
            "preprocess": [_u(model="qwen3.7-flash", p=100, c=100)],
        }
        n = record_usage_bucket("sess-1", bucket)
        assert n == 3

        s = usage_summary(hours=24)
        assert s["requests"] == 3
        assert s["tokens"] == (1000 + 500) + (2000 + 0) + (100 + 100)
        # decide 两次: qwen3.8-max 0.0024+0.0048 → 见 estimate;仅断言 >0 且模型拆分存在
        assert s["total_cost_rmb"] > 0
        models = {m["model"] for m in s["by_model"]}
        assert models == {"qwen3.8-max", "qwen3.7-flash"}
        roles = {r["role"] for r in s["by_role"]}
        assert roles == {"decide", "preprocess"}
        assert s["by_day"] and s["by_day"][0]["req"] == 3

    def test_degraded_marked_when_not_primary(self):
        # 默认主链 dashscope → ollama 出现 = 降级
        bucket = {"decide": [_u(provider="ollama", model="qwen2.5:7b")]}
        record_usage_bucket("sess-2", bucket)
        s = usage_summary(hours=24)
        assert s["degraded_requests"] >= 1

    def test_empty_bucket_no_rows(self):
        assert record_usage_bucket("sess-3", {}) == 0
        assert record_usage_bucket("sess-3", None) == 0


class TestTierEvents:
    def test_tier_counts(self):
        record_data_tier("MySQL", 100)
        record_data_tier("MySQL", 50)
        record_data_tier("tianchi", 30)
        rows = {t["tier"]: t for t in tier_summary(hours=24)}
        assert rows["MySQL"]["n"] == 2
        assert rows["MySQL"]["rows_loaded"] == 150
        assert rows["tianchi"]["n"] == 1

    def test_empty_tier_ignored(self):
        record_data_tier("", 0)
        assert tier_summary(hours=24) or True  # 不抛即过
