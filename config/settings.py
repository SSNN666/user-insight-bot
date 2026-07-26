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
    OUTLIER_METHOD: str = "iqr"     # "iqr" | "zscore"
    FILTER_OUTLIERS: bool = True    # filter extreme values
    EXTENDED_PROFILE_ENABLED: bool = True  # compute repurchase interval, price sensitivity, category preference
    SNAPSHOT_DIR: str = "cache_data/snapshots"  # cluster snapshot storage

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
    }

    # ── Agent ──
    PREPROCESS_ENABLED: bool = True
    MAX_REFLECTION_ROUNDS: int = 2
    BUSINESS_MEMORY_TTL: int = 3600
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
    # Dedup
    EVENT_COOLDOWN_SECONDS: int = 3600
    EVENT_MERGE_WINDOW: int = 600
    # Safety
    MAX_AUTO_TOOL_ROUNDS: int = 5
    AUTO_TASK_TIMEOUT_SECONDS: int = 120

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
