"""Query preprocessor — e-commerce intent classification, entity extraction,
and multi-condition query decomposition.

Uses a lightweight LLM call to enrich the raw user query before it enters
the agent's decision loop.
"""

from enum import Enum

from pydantic import BaseModel, Field

from log.logger import get_logger

logger = get_logger(__name__)

# ── Types ───────────────────────────────────────────────────────


class QueryIntent(str, Enum):
    USER_SEGMENT = "user_segment"        # 用户分群分析
    USER_PROFILE = "user_profile"        # 用户画像 / 个体分析
    ORDER_ANALYSIS = "order_analysis"    # 订单 / 销售分析
    PRODUCT_INSIGHT = "product_insight"  # 商品分析
    GENERAL = "general"                  # 通用问答 / 闲聊


class PreprocessResult(BaseModel):
    intent: QueryIntent = QueryIntent.GENERAL
    sub_queries: list[str] = Field(default_factory=list)
    entities: dict = Field(default_factory=lambda: {
        "time": [], "segments": [], "products": [], "cities": [],
    })
    original_query: str = ""

    def format_for_context(self) -> str:
        """Compact summary for injection into the agent state."""
        parts = [f"intent={self.intent.value}"]
        if self.entities:
            for k, v in self.entities.items():
                if v:
                    parts.append(f"{k}={v}")
        if len(self.sub_queries) > 1:
            parts.append(f"sub_queries({len(self.sub_queries)})")
        return " | ".join(parts)


# ── Few-shot prompt ─────────────────────────────────────────────

PREPROCESS_SYSTEM = """你是一个电商查询分析器。分析用户问题并输出 JSON。

## 意图分类
- product_insight: 商品搜索、购物咨询、找商品、问价格、品类浏览、有什么卖的
- user_segment: 用户分群、用户群体特征、RFM分析、分群人数、高价值用户
- user_profile: 单个用户画像、用户详情
- order_analysis: 订单统计、销售额、订单趋势
- general: 打招呼、闲聊、问候、其他无法归类的问题

## 重要：优先判断 product_insight
如果用户在找商品、问"有没有XX"、"多少钱"、"推荐"、"买什么"、
只输入一个商品名（如"手机"、"耳机"）→ 一律归类为 product_insight。

## 实体抽取
从问题中提取: time(时间), segments(分群), products(商品名), cities(城市)

## 子任务拆分
如果问题包含多个独立条件，拆分为独立的子问题。
单一条件则 sub_queries 只包含原问题本身。

## 输出格式（严格 JSON）
{"intent": "...", "entities": {"time":[],"segments":[],"products":[],"cities":[]}, "sub_queries": ["..."]}

示例:
Q: 手机
A: {"intent":"product_insight","entities":{"time":[],"segments":[],"products":["手机"],"cities":[]},"sub_queries":["搜索手机相关商品"]}

Q: 有没有蓝牙耳机？
A: {"intent":"product_insight","entities":{"time":[],"segments":[],"products":["蓝牙耳机"],"cities":[]},"sub_queries":["搜索蓝牙耳机"]}

Q: 推荐500以内的电子商品
A: {"intent":"product_insight","entities":{"time":[],"segments":[],"products":["电子商品"],"cities":[]},"sub_queries":["搜电子品类中500以内的商品"]}

Q: 各分群的人数是多少？
A: {"intent":"user_segment","entities":{"time":[],"segments":[],"products":[],"cities":[]},"sub_queries":["各分群的人数是多少？"]}

Q: 高价值用户有哪些特征？
A: {"intent":"user_segment","entities":{"time":[],"segments":["高价值"],"products":[],"cities":[]},"sub_queries":["高价值用户的特征分析"]}
"""


# ── Public API ──────────────────────────────────────────────────


def preprocess_query(query: str, llm) -> PreprocessResult:
    """Run LLM-driven query preprocessing.

    Args:
        query: Raw user question.
        llm: A chat model that accepts ``invoke(messages)`` and returns
             an object with ``.content``.

    Returns:
        ``PreprocessResult`` with intent, entities, and sub-queries.
    """
    messages = [
        {"role": "system", "content": PREPROCESS_SYSTEM},
        {"role": "user", "content": query},
    ]

    def _fallback() -> PreprocessResult:
        return PreprocessResult(
            intent=QueryIntent.GENERAL,
            sub_queries=[query],
            original_query=query,
        )

    try:
        response = llm.invoke(messages)
        from common.json_repair import load_json_or_default
        data, repair = load_json_or_default(
            response.content,
            default={"intent": "general", "entities": {}, "sub_queries": [query]},
            llm=llm, retry_messages=messages,
            hint='{"intent":"...","entities":{"time":[],"segments":[],"products":[],"cities":[]},"sub_queries":["..."]}',
            logger=logger,
        )
        try:
            result = PreprocessResult(
                intent=QueryIntent(data.get("intent", "general")),
                entities=data.get("entities", {}),
                sub_queries=data.get("sub_queries", [query]),
                original_query=query,
            )
        except Exception as e:
            # 未知 intent 值等校验失败 → 与解析失败同路径兜底
            logger.warning("preprocess_validate_failed", extra={"error": str(e)})
            result = _fallback()
        logger.info("preprocess_done", extra={
            "intent": result.intent.value,
            "sub_queries": len(result.sub_queries),
            "repaired": repair.repaired,
            "retries": repair.retries,
        })
        return result

    except Exception as e:
        logger.error("preprocess_error", extra={"error": str(e)})
        return _fallback()
