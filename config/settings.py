"""Centralized application configuration — single source of truth.

Uses pydantic-settings to load from .env with sensible defaults.
All modules read config via `get_settings()`, never from os.environ directly.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ──
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str = "root"
    DB_PASSWORD: str = ""
    DB_NAME: str = "ecommerce"
    DB_CHARSET: str = "utf8mb4"

    # ── LLM ──
    LLM_MODEL_NAME: str = "qwen2.5:7b"
    OPENAI_API_KEY: str = "ollama"
    OPENAI_BASE_URL: str = "http://localhost:11434/v1"
    LLM_TEMPERATURE: float = 0.0
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_RETRIES: int = 3

    # ── 统一大模型适配器 (llm/adapter.py,与康养项目共用同一份源码) ──
    # 主链路云端文本模型,降级链末尾 Ollama 仅本地调试/兜底。
    # 未配 Key 的云供应商自动跳过;密钥全部走 .env,勿硬编码。
    LLM_PROVIDER_PRIMARY: str = "dashscope"   # "dashscope" | "deepseek" | "qianfan" | "ollama"
    LLM_FALLBACK_CHAIN: list[str] = ["dashscope", "deepseek", "qianfan", "ollama"]
    LLM_LOCAL_FALLBACK_ENABLED: bool = True   # 云端全部失败时是否落本地 Ollama
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    QIANFAN_API_KEY: str = ""
    QIANFAN_BASE_URL: str = "https://qianfan.baidubce.com/v2"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_NUM_CTX: int = 8192
    LLM_TIMEOUT: float = 60.0                # 云端单次请求超时秒数
    LLM_LOCAL_TIMEOUT: float = 300.0         # 本地 Ollama 超时(CPU 推理慢)
    LLM_RETRY_BACKOFF: tuple = (1.0, 2.0)    # 单供应商内重试退避秒数
    # 角色 → 各供应商模型映射 + 生成参数
    # decide: Agent 主工作流(工具调用),qwen3.8-max 重推理;
    # preprocess/reflect: 轻量 JSON 输出任务,实测 qwen3.7-flash 意图准确率与 max 一致
    # (8/8 典型问题)而成本低一个量级 → 分层调度
    # thinking 默认关(实测:数据查询类任务开思考 27s/轮,关思考快一个量级;
    # 需要更强推理时改 true + thinking_budget)
    LLM_ROLES: dict = {
        "decide": {
            "dashscope": "qwen3.8-max", "deepseek": "deepseek-chat",
            "qianfan": "ernie-4.5-turbo-128k", "ollama": "qwen2.5:7b",
            "temperature": 0.0, "max_tokens": None,
            "thinking": False,
        },
        "preprocess": {
            "dashscope": "qwen3.7-flash", "deepseek": "deepseek-chat",
            "qianfan": "ernie-4.5-turbo-128k", "ollama": "qwen2.5:7b",
            "temperature": 0.0, "max_tokens": 256, "thinking": False,
        },
        "reflect": {
            "dashscope": "qwen3.7-flash", "deepseek": "deepseek-chat",
            "qianfan": "ernie-4.5-turbo-128k", "ollama": "qwen2.5:7b",
            "temperature": 0.0, "max_tokens": 256, "thinking": False,
        },
        "suggest": {
            "dashscope": "qwen3.7-flash", "deepseek": "deepseek-chat",
            "qianfan": "ernie-4.5-turbo-128k", "ollama": "qwen2.5:7b",
            "temperature": 0.3, "max_tokens": 1024, "thinking": False,
        },
        "vision": {
            "dashscope": "qwen3-vl-plus", "ollama": None,
            "temperature": 0.3, "max_tokens": 1024, "thinking": False,
        },
    }

    # ── API (FastAPI) ──
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    API_WORKERS: int = 1

    # ── Cache ──
    CACHE_DIR: str = "cache_data"
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    PIPELINE_CACHE_TTL: int = 3600 # pipeline compute cache (seconds)

    # ── Data Pipeline ──
    DATA_PAGE_SIZE: int = 500       # rows per paginated query
    DATA_N_CLUSTERS: int = 4        # KMeans clusters (fallback when auto-K disabled)
    AUTO_K_ENABLED: bool = True     # auto-select optimal K
    AUTO_K_MIN: int = 2             # K search range lower bound
    AUTO_K_MAX: int = 10            # K search range upper bound
    AUTO_K_GAP_B: int = 10          # Gap statistic 参考分布采样次数
    AUTO_K_MIN_CLUSTER_SIZE: int = 20       # 候选 K 的最小簇人数下限(业务可行性)
    AUTO_K_MIN_CLUSTER_RATIO: float = 0.01  # 同上,按总人数比例(两者取大)
    OUTLIER_METHOD: str = "iqr"     # "iqr" | "zscore"
    FILTER_OUTLIERS: bool = True    # filter extreme values
    EXTENDED_PROFILE_ENABLED: bool = True  # compute repurchase interval, price sensitivity, category preference
    SNAPSHOT_DIR: str = "cache_data/snapshots"  # cluster snapshot storage
    SNAPSHOT_KEEP: int = 2016       # 快照保留数量(默认一周@5分钟轮询;实测 20 个只够 100 分钟,
                                    # 趋势图/环比几乎无历史可看,故放宽;单个快照约 20KB,一周≈40MB)
    MOCK_SEED: int | None = 42      # mock 数据随机种子(固定=演示可复现;None=每次随机)
    # 京东 JData 公开脱敏数据集(官方在 DataFountain,下载渠道见 docs/DATA_COMPLIANCE.md):
    # DATA_SOURCE=tianchi 时在 TTL 缓存之后接入,文件缺失自动回退 mock
    DATA_SOURCE: str = "auto"           # "auto"=MySQL→缓存→mock | "tianchi"=MySQL→缓存→JData CSV→mock
    TIANCHI_DATA_DIR: str = "data/tianchi"      # JData CSV 文件目录
    TIANCHI_MAX_ACTIONS: int | None = 1_000_000 # 行为表读取行数上限(None=全量;演示建议采样)
    TIANCHI_MAX_USERS: int | None = 100_000     # 采样用户数上限(None=全量)

    # ── 防护(演示级) ──
    DEBUG_API_KEY: str = ""         # 演示级鉴权:/debug/* 与 /tasks/* 要求 X-API-Key 头(空=关闭)
    PROMPT_GUARD_ENABLED: bool = True    # Prompt 注入检测(规则式)
    PROMPT_GUARD_BLOCK_SCORE: int = 4    # 加权分 ≥ 此值拦截
    BAIDU_CENSOR_ENABLED: bool = True    # 百度内容审核(fail-open)
    BAIDU_AK: str = ""                   # 百度智能云 AK/SK(与千帆 Key 不同)
    BAIDU_SK: str = ""
    CENSOR_TIMEOUT: float = 3.0          # 审核超时秒数(超时放行 + 日志)

    # ── Log ──
    LOG_LEVEL: str = "INFO"
    LOG_DIR_USER_CHAT: str = "logs/user_chat_log"
    LOG_DIR_AUTO_TASK: str = "logs/auto_task_log"
    LOG_MAX_BYTES: int = 10_485_760    # 10 MB per file
    LOG_BACKUP_COUNT: int = 5

    # ── Skill ──
    SKILL_ENABLED: dict = {
        "search_products": True,
        "get_categories": True,
        "get_user_segment_stats": True,
        "get_segment_rules": True,
        "get_high_value_users": True,
        "refresh_pipeline": True,
        "get_funnel_analysis": True,
    }

    # ── Agent ──
    PREPROCESS_ENABLED: bool = True
    MAX_REFLECTION_ROUNDS: int = 2
    BUSINESS_MEMORY_TTL: int = 3600

    # ── 会话记忆(持久化 + 长对话摘要) ──
    SESSION_DIR: str = "cache_data/sessions"   # 会话 checkpoint 持久化目录(重启不丢多轮上下文)
    SESSION_TTL_DAYS: float = 7.0              # 过期会话文件自动清理
    CONVERSATION_SUMMARY_ENABLED: bool = True  # 消息数超阈值时压缩历史为摘要
    CONVERSATION_SUMMARY_THRESHOLD: int = 10   # 触发摘要的消息数阈值
    CONVERSATION_KEEP_RECENT: int = 4          # 摘要后保留最近 N 条原文
    FACT_CHECK_MODE: str = "relaxed"        # "relaxed" | "strict"
    FACT_CHECK_MAX_RETRIES: int = 2         # max re-generation attempts in strict mode
    OUTPUT_FORMAT: str = "markdown"
    TOOL_CALL_GUARD_ENABLED: bool = True    # inject guard msg when LLM skips tools
    TOOL_CALL_GUARD_KEYWORDS: list[str] = [
        "分群", "用户", "人数", "统计", "数据", "指标", "画像",
        "多少", "几个", "对比", "消费", "RFM", "增长", "转化率",
    ]
    TRACE_HISTORY_CAP: int = 200            # max number of traces kept in memory

    # ── Watcher (Phase 4) ──
    WATCHER_ENABLED: bool = True
    POLL_INTERVAL_SECONDS: int = 300
    WATCHER_DB_PATH: str = "cache_data/watcher.db"
    FEEDBACK_DB_PATH: str = "cache_data/feedback.db"

    # ── Embedding ──
    EMBEDDING_DIM: int = 768                  # vector dimension (nomic-embed-text)

    # ── Flywheel (Phase 7) ──
    FLYWHEEL_ENABLED: bool = True
    FLYWHEEL_DB_PATH: str = "cache_data/flywheel.db"
    FLYWHEEL_UPDATE_INTERVAL: int = 600
    FLYWHEEL_RETRIEVAL_TOP_K: int = 3
    FLYWHEEL_VECTOR_BACKEND: str = "milvus"   # "milvus" | "memory"
    FLYWHEEL_MILVUS_DATA_DIR: str = "cache_data/milvus_vectors"
    # Thresholds
    EVENT_THRESHOLD_PCT: float = 20.0
    SEGMENT_RATIO_THRESHOLD_PCT: float = 10.0
    HIGH_VALUE_CHURN_THRESHOLD: int = 5
    ORDER_CRASH_THRESHOLD_PCT: float = 30.0
    EXTREME_OUTLIER_STD: float = 3.0
    # 流失预警(高价值用户 N 天未下单,用户级规则,独立于快照 diff)
    HIGH_VALUE_DORMANT_DAYS: int = 30
    HIGH_VALUE_DORMANT_MIN_USERS: int = 5
    HIGH_VALUE_DORMANT_COOLDOWN_SECONDS: int = 86400   # 事件级节流(用户集去重之外的次闸)
    # 自动周报(每周一生成 Markdown,纯规则拼接;错过触发时刻自动补做,幂等)
    WEEKLY_REPORT_ENABLED: bool = True
    WEEKLY_REPORT_HOUR: int = 9
    WEEKLY_REPORT_TIMEZONE: str = "Asia/Shanghai"
    # Dedup
    EVENT_COOLDOWN_SECONDS: int = 3600
    EVENT_MERGE_WINDOW: int = 600
    # Safety
    MAX_AUTO_TOOL_ROUNDS: int = 5
    AUTO_TASK_TIMEOUT_SECONDS: int = 120
    ORCHESTRATOR_TASK_TIMEOUT_SECONDS: int = 300   # HIGH 事件三阶段流水线超时(三段 Agent 串行)

    # ── Rate Limiting (Phase 5) ──
    RATE_LIMIT_ENABLED: bool = True
    CHAT_RATE_LIMIT: int = 20
    CHAT_RATE_WINDOW: int = 60
    LLM_CONCURRENCY_LIMIT: int = 3
    AUTO_TASK_POOL_SIZE: int = 2
    SIMILAR_QUERY_DEDUP: bool = True

    # ── CORS ──
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:7860",
    ]

    # ── Scorer (Phase 7 flywheel) ──
    SCORER_FEEDBACK_UP_WEIGHT: float = 0.3
    SCORER_FEEDBACK_DOWN_WEIGHT: float = -0.4
    SCORER_FACT_CHECK_PASSED_WEIGHT: float = 0.3
    SCORER_FACT_VIOLATION_WEIGHT: float = 0.2
    SCORER_EFFICIENT_TOOLS_WEIGHT: float = 0.1
    SCORER_INEFFICIENT_TOOLS_WEIGHT: float = -0.1
    SCORER_TOOL_ERROR_WEIGHT: float = -0.6
    SCORER_INCOMPLETE_WEIGHT: float = -0.3
    SCORER_REFUSAL_WEIGHT: float = -0.2
    SCORER_EXPERT_ANNOTATED_WEIGHT: float = 0.2

    # ── App ──
    APP_TITLE: str = "电商用户智能画像问答"
    APP_DESCRIPTION: str = (
        "向我询问用户分群信息，"
        "例如：'各分群的人数是多少？'、'高价值用户有哪些特征？'"
    )
    APP_VERSION: str = "0.2.0"


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance, cached per process."""
    return Settings()
