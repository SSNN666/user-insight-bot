import os
from dotenv import load_dotenv

load_dotenv()

# 数据库配置（请根据实际环境修改）
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "123456"),
    "database": os.getenv("DB_NAME", "ecommerce"),
    "charset": "utf8mb4"
}

# LLM 配置（OpenAI 兼容接口）
LLM_MODEL_NAME = os.getenv("LLM_MODEL", "deepseek-r1:7b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # 可选，用于代理或本地模型