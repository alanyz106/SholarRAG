from typing import Optional

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

# Find .env file - check project root first, fallback for Docker
_candidate = Path(__file__).resolve().parent.parent.parent.parent / ".env"
ENV_FILE = str(_candidate) if _candidate.exists() else ".env"


class Settings(BaseSettings):
    # App
    APP_NAME: str = "NexusRAG"
    DEBUG: bool = False
    API_V1_PREFIX: str = "/api/v1"

    # Base directory (backend folder)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Database
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5433/nexusrag")

    # LLM Provider: "gemini" | "openai" | "ollama"
    LLM_PROVIDER: str = Field(default="gemini")

    # OpenAI-compatible (OpenAI, OpenRouter, LocalAI, vLLM, etc.)
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")
    OPENAI_ORGANIZATION: Optional[str] = Field(default=None)
    OPENAI_MODEL: str = Field(default="gpt-4o")

    # Google AI (deprecated - migrating to OpenAI)
    GOOGLE_AI_API_KEY: str = Field(default="")

    # Ollama
    OLLAMA_HOST: str = Field(default="http://localhost:11434")
    OLLAMA_MODEL: str = Field(default="gemma3:12b")
    OLLAMA_ENABLE_THINKING: bool = Field(default=False)

    # LLM (fast model for chat + KG extraction — used when provider=gemini)
    LLM_MODEL_FAST: str = Field(default="gemini-2.5-flash")

    # Thinking level for Gemini 3.x+ models: "minimal" | "low" | "medium" | "high"
    # Gemini 2.5 uses thinking_budget_tokens instead (auto-detected)
    LLM_THINKING_LEVEL: str = Field(default="medium")

    # Max output tokens for LLM chat responses (includes thinking tokens)
    # Gemini 3.1 Flash-Lite supports up to 65536
    LLM_MAX_OUTPUT_TOKENS: int = Field(default=8192)

    # KG Embedding provider (can differ from LLM provider)
    KG_EMBEDDING_PROVIDER: str = Field(default="gemini")
    KG_EMBEDDING_MODEL: str = Field(default="gemini-embedding-001")
    KG_EMBEDDING_DIMENSION: int = Field(default=3072)

    # OpenAI-compatible embedding (when KG_EMBEDDING_PROVIDER=openai)
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    OPENAI_EMBEDDING_DIMENSION: int = Field(default=1536)

    # ChromaDB
    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8002)

    # NexusRAG Pipeline
    NEXUSRAG_ENABLED: bool = True
    NEXUSRAG_ENABLE_KG: bool = True
    NEXUSRAG_ENABLE_IMAGE_EXTRACTION: bool = True
    NEXUSRAG_ENABLE_IMAGE_CAPTIONING: bool = True
    NEXUSRAG_ENABLE_TABLE_CAPTIONING: bool = True
    NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS: int = 8000
    NEXUSRAG_CHUNK_MAX_TOKENS: int = 512
    NEXUSRAG_KG_QUERY_TIMEOUT: float = 30.0
    NEXUSRAG_KG_CHUNK_TOKEN_SIZE: int = 1200
    NEXUSRAG_KG_LANGUAGE: str = "Vietnamese"
    NEXUSRAG_KG_ENTITY_TYPES: list[str] = [
        "Organization", "Person", "Product", "Location", "Event",
        "Financial_Metric", "Technology", "Date", "Regulation",
    ]
    NEXUSRAG_DEFAULT_QUERY_MODE: str = "hybrid"
    NEXUSRAG_DOCLING_IMAGES_SCALE: float = 2.0
    NEXUSRAG_MAX_IMAGES_PER_DOC: int = 50
    NEXUSRAG_ENABLE_FORMULA_ENRICHMENT: bool = True

    # ChromaDB Embedding Provider (for vector search)
    # Options: "sentence_transformers" (local), "openai", "gemini", "ollama"
    CHROMA_EMBEDDING_PROVIDER: str = Field(default="sentence_transformers")

    # OpenAI-compatible embedding for ChromaDB (when CHROMA_EMBEDDING_PROVIDER=openai)
    CHROMA_OPENAI_MODEL: str = Field(default="text-embedding-3-small")
    CHROMA_OPENAI_DIMENSION: int = Field(default=1536)
    # Reuses OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_ORGANIZATION

    # Google AI embedding for ChromaDB (when CHROMA_EMBEDDING_PROVIDER=gemini)
    CHROMA_GEMINI_MODEL: str = Field(default="gemini-embedding-001")
    CHROMA_GEMINI_DIMENSION: int = Field(default=3072)
    # Reuses GOOGLE_AI_API_KEY

    # Ollama embedding for ChromaDB (when CHROMA_EMBEDDING_PROVIDER=ollama)
    CHROMA_OLLAMA_MODEL: str = Field(default="nomic-embed-text")
    CHROMA_OLLAMA_DIMENSION: int = Field(default=768)
    # Reuses OLLAMA_HOST

    # NexusRAG Retrieval Quality
    NEXUSRAG_EMBEDDING_MODEL: str = "BAAI/bge-m3"
    NEXUSRAG_RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    NEXUSRAG_VECTOR_PREFETCH: int = 20
    NEXUSRAG_RERANKER_TOP_K: int = 8
    NEXUSRAG_MIN_RELEVANCE_SCORE: float = 0.15

    # Reranker Provider (for cross-encoder reranking)
    # Options: "sentence_transformers" (local), "cohere" (Cohere API),
    #          "jina" (Jina AI API), "modelscope" (ModelScope OpenAI-compatible)
    RERANKER_PROVIDER: str = Field(default="sentence_transformers")

    # Cohere Rerank API (when RERANKER_PROVIDER=cohere)
    COHERE_API_KEY: str = Field(default="")
    COHERE_RERANK_MODEL: str = Field(default="rerank-english-v3.0")
    COHERE_RERANK_TOP_N: int = Field(default=10)

    # Jina AI Rerank API (when RERANKER_PROVIDER=jina)
    JINA_API_KEY: str = Field(default="")
    JINA_RERANK_MODEL: str = Field(default="jina-reranker-v2-base-multilingual")
    JINA_RERANK_TOP_N: int = Field(default=10)

    # ModelScope Rerank API (when RERANKER_PROVIDER=modelscope)
    # ModelScope uses OpenAI-compatible client, but rerank endpoint similar to Cohere
    MODELSCOPE_API_KEY: str = Field(default="")
    MODELSCOPE_BASE_URL: str = Field(default="https://ms-ens-6f01371a-0c58.api-inference.modelscope.cn/v1")
    MODELSCOPE_RERANK_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3")
    MODELSCOPE_RERANK_TOP_N: int = Field(default=10)

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:5174", "http://localhost:3000"]

    model_config = {
        "env_file": str(ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
