"""
Embedding Service
=================
Generates vector embeddings using multiple providers:
- sentence_transformers (local)
- openai (OpenAI-compatible API)
- gemini (Google Gemini API)
- ollama (local Ollama API)

Default provider: CHROMA_EMBEDDING_PROVIDER from settings.
"""
from __future__ import annotations

import logging
from typing import Sequence, Optional
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    Supports multiple providers: sentence_transformers, openai, gemini, ollama.
    """

    # Dimension lookup for local sentence-transformers models
    _KNOWN_DIMS = {
        "BAAI/bge-m3": 1024,
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-multilingual-MiniLM-L12-v2": 384,
        "intfloat/multilingual-e5-large-instruct": 1024,
    }

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize embedding service.

        Args:
            provider: Embedding provider type (sentence_transformers, openai, gemini, ollama)
                     Defaults to settings.CHROMA_EMBEDDING_PROVIDER
            model_name: Model name (provider-specific). Defaults to provider's default.
        """
        self.provider = provider or settings.CHROMA_EMBEDDING_PROVIDER
        self._external_provider = None  # For external API providers
        self._model = None  # For local sentence_transformers

        # Configure based on provider
        if self.provider == "sentence_transformers":
            self.model_name = model_name or settings.NEXUSRAG_EMBEDDING_MODEL
        elif self.provider == "openai":
            from app.services.llm.openai import OpenAIEmbeddingProvider
            self.model_name = model_name or settings.CHROMA_OPENAI_MODEL
            # Use independent ChromaDB config if set, otherwise fall back to shared config
            api_key = settings.CHROMA_OPENAI_API_KEY or settings.OPENAI_API_KEY
            base_url = settings.CHROMA_OPENAI_BASE_URL or settings.OPENAI_BASE_URL
            organization = settings.CHROMA_OPENAI_ORGANIZATION or settings.OPENAI_ORGANIZATION
            self._external_provider = OpenAIEmbeddingProvider(
                api_key=api_key,
                model=self.model_name,
                base_url=base_url,
                organization=organization,
            )
        elif self.provider == "gemini":
            from app.services.llm.gemini import GeminiEmbeddingProvider
            self.model_name = model_name or settings.CHROMA_GEMINI_MODEL
            self._external_provider = GeminiEmbeddingProvider(
                api_key=settings.GOOGLE_AI_API_KEY,
                model=self.model_name,
            )
        elif self.provider == "ollama":
            from app.services.llm.ollama import OllamaEmbeddingProvider
            self.model_name = model_name or settings.CHROMA_OLLAMA_MODEL
            self._external_provider = OllamaEmbeddingProvider(
                host=settings.OLLAMA_HOST,
                model=self.model_name,
            )
        else:
            raise ValueError(
                f"Unknown CHROMA_EMBEDDING_PROVIDER: {self.provider!r}. "
                "Supported: sentence_transformers, openai, gemini, ollama"
            )

    @property
    def dimension(self) -> int:
        """Return the embedding dimension size."""
        if self.provider == "sentence_transformers":
            if self._model is not None:
                return self._model.get_sentence_embedding_dimension()
            return self._KNOWN_DIMS.get(self.model_name, 1024)
        else:
            # DEBUG
            if self._external_provider is None:
                raise RuntimeError(f"_external_provider is None for provider={self.provider}")
            dim = self._external_provider.get_dimension()
            logger.debug(f"EmbeddingService.dimension called, provider={self.provider}, returning {dim}")
            return dim

    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                f"Embedding model loaded: {self.model_name} "
                f"(dim={self._model.get_sentence_embedding_dimension()})"
            )
        return self._model


    def embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        if self.provider == "sentence_transformers":
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embedding.tolist()
        else:
            embeddings = self._external_provider.embed_sync([text])
            return embeddings[0].tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts in batch."""
        if not texts:
            return []
        valid_texts = [t for t in texts if t.strip()]
        if not valid_texts:
            raise ValueError("All texts are empty")
        if self.provider == "sentence_transformers":
            embeddings = self.model.encode(
                valid_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
            )
            return embeddings.tolist()
        else:
            embeddings = self._external_provider.embed_sync(valid_texts)
            return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Generate embedding for a search query."""
        return self.embed_text(query)

    @property
    def model(self):
        """Lazy load the local sentence-transformers model."""
        if self.provider != "sentence_transformers":
            raise AttributeError("model property only available for sentence_transformers provider")
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(
                f"Embedding model loaded: {self.model_name} "
                f"(dim={self._model.get_sentence_embedding_dimension()})"
            )
        return self._model


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
