"""
Embedding Service
=================
Generates vector embeddings using OpenAI-compatible API.

Default provider: EMBEDDING_* settings (unified config).
"""
from __future__ import annotations

from typing import Sequence, Optional

from app.core.config import settings


class EmbeddingService:
    """Service for generating text embeddings. Only OpenAI-compatible provider."""

    def __init__(self, model_name: Optional[str] = None):
        from app.services.llm import get_embedding_provider
        self._provider = get_embedding_provider()
        self.model_name = model_name or settings.EMBEDDING_MODEL

    @property
    def dimension(self) -> int:
        return self._provider.get_dimension()

    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        embeddings = self._provider.embed_sync([text])
        return embeddings[0].tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")
        embeddings = self._provider.embed_sync(list(valid_texts))
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return self.embed_text(query)

    @property
    def model(self) -> str:
        """Return the model name for compatibility."""
        return self.model_name


# Default service instance (singleton)
_default_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the default embedding service."""
    global _default_service
    if _default_service is None:
        _default_service = EmbeddingService()
    return _default_service


def embed_text(text: str) -> list[float]:
    """Convenience function to embed a single text."""
    return get_embedding_service().embed_text(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    """Convenience function to embed multiple texts."""
    return get_embedding_service().embed_texts(texts)
