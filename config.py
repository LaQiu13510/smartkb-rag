"""
SmartKB 全局配置管理
=====================
统一管理所有环境变量和配置项。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
ROOT_DIR = Path(__file__).parent
DOCUMENTS_DIR = ROOT_DIR / "documents"
DATA_DIR = ROOT_DIR / "data"

# 确保目录存在
DOCUMENTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# 加载 .env 文件。显式传入的系统环境变量优先，便于测试隔离和部署覆盖。
load_dotenv(ROOT_DIR / ".env", override=False)


# ============================================================
# DeepSeek LLM 配置
# ============================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

# ============================================================
# Google Embedding 配置
# ============================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GOOGLE_EMBEDDING_MODEL = os.getenv(
    "GOOGLE_EMBEDDING_MODEL", "models/text-embedding-004"
)

# ============================================================
# PostgreSQL 配置
# ============================================================
DB_URL = os.getenv("DB_URL", "")

# ============================================================
# Milvus 配置
# ============================================================
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "smartkb_collection")

# ============================================================
# RAG 参数
# ============================================================
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))       # 文档分块大小
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))  # 分块重叠大小
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))  # 检索返回数量
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))      # 向量维度 (智谱 embedding-2=1024)

# ============================================================
# 缓存配置
# ============================================================
REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_BACKEND = os.getenv("CACHE_BACKEND", "auto").lower()
QUERY_CACHE_ENABLED = os.getenv("QUERY_CACHE_ENABLED", "true").lower() == "true"
QUERY_CACHE_TTL_SECONDS = int(os.getenv("QUERY_CACHE_TTL_SECONDS", "600"))

# ============================================================
# LangSmith 追踪 (可选)
# ============================================================
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "smartkb")

# 初始化 LangSmith
if LANGSMITH_TRACING and LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = LANGSMITH_PROJECT
