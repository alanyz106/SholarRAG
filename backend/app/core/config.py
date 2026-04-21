from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

# Find .env file - check project root first, fallback for Docker
_candidate = Path(__file__).resolve().parent.parent.parent.parent / ".env"
ENV_FILE = str(_candidate) if _candidate.exists() else ".env"


class Settings(BaseSettings):
    # === App ===
    APP_NAME: str = "NexusRAG"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # === 数据库 ===
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5433/nexusrag")

    # === 向量存储 (Qdrant) ===
    VECTOR_STORE_PROVIDER: str = "qdrant"  # 固定为 qdrant
    QDRANT_URL: str = Field(default="")
    QDRANT_API_KEY: str = Field(default="")

    # === 嵌入服务 (向量检索) ===
    EMBEDDING_PROVIDER: str = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSION: int = 1536
    EMBEDDING_OPENAI_API_KEY: str = Field(default="")
    EMBEDDING_OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")
    EMBEDDING_OPENAI_ORGANIZATION: Optional[str] = Field(default=None)

    # === LLM (对话) ===
    LLM_PROVIDER: str = "openai"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_OPENAI_API_KEY: str = Field(default="")
    LLM_OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")
    LLM_OPENAI_ORGANIZATION: Optional[str] = Field(default=None)
    LLM_MAX_TOKENS: int = 8192

    # === 重排序服务 ===
    RERANKER_PROVIDER: str = "gitee_ai"  # gitee_ai | siliconflow
    GITEE_AI_API_TOKEN: str = Field(default="")
    GITEE_AI_RERANK_MODEL: str = "bge-reranker-v2-m3"
    GITEE_AI_RERANK_TOP_N: int = 10
    SILICONFLOW_API_KEY: str = Field(default="")
    SILICONFLOW_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
    SILICONFLOW_RERANK_TOP_N: int = 10

    # === 知识图谱 ===
    KG_ENABLED: bool = True
    KG_LANGUAGE: str = "Chinese"
    KG_ENTITY_TYPES: list[str] = [
        "Model", "Dataset", "Method", "Task", "Metric", "Framework",
        "Architecture", "Layer", "Optimizer", "Loss", "Organization",
        "Person", "Publication", "Year", "Hardware", "Hyperparameter",
        "Code", "Technology"
    ]
    KG_OPENAI_API_KEY: str = Field(default="")  # 独立 key

    # === NexusRAG Pipeline ===
    NEXUSRAG_ENABLED: bool = False
    NEXUSRAG_ENABLE_KG: bool = True
    NEXUSRAG_DOCLING_DEVICE: str = "auto"
    NEXUSRAG_DOCLING_NUM_THREADS: int = 4
    NEXUSRAG_DOCLING_IMAGES_SCALE: float = 2.0
    NEXUSRAG_CHUNK_MAX_TOKENS: int = 512
    NEXUSRAG_VECTOR_PREFETCH: int = 20
    NEXUSRAG_RERANKER_TOP_K: int = 8
    NEXUSRAG_MIN_RELEVANCE_SCORE: float = 0.15
    NEXUSRAG_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    NEXUSRAG_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    NEXUSRAG_ENABLE_IMAGE_EXTRACTION: bool = True
    NEXUSRAG_ENABLE_IMAGE_CAPTIONING: bool = False
    NEXUSRAG_ENABLE_TABLE_CAPTIONING: bool = False
    NEXUSRAG_ENABLE_FORMULA_ENRICHMENT: bool = True
    NEXUSRAG_MAX_IMAGES_PER_DOC: int = 50
    NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS: int = 2000
    NEXUSRAG_KG_CHUNK_TOKEN_SIZE: int = 256
    NEXUSRAG_KG_QUERY_TIMEOUT: int = 30

    # === KG Embedding (knowledge graph) ===
    KG_EMBEDDING_PROVIDER: str = "sentence_transformers"
    KG_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    KG_EMBEDDING_DIMENSION: int = 1024

    # === 评估 (Judge) ===
    JUDGE_PROVIDER: str = "openai"
    JUDGE_OPENAI_API_KEY: str = Field(default="")
    JUDGE_OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")
    JUDGE_OPENAI_MODEL: str = "gpt-4o"

    # === CORS ===
    CORS_ORIGINS: list[str] = ["http://localhost:5174"]

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
