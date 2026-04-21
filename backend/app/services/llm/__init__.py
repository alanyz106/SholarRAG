"""
LLM Provider Package
=====================
Factory functions to create LLM and embedding providers based on config.

Usage::

    from app.services.llm import get_llm_provider, get_embedding_provider

    llm = get_llm_provider()          # uses LLM_PROVIDER from .env
    emb = get_embedding_provider()    # uses EMBEDDING_PROVIDER from .env
"""
from __future__ import annotations

from functools import lru_cache

from app.services.llm.base import EmbeddingProvider, LLMProvider


@lru_cache
def get_llm_provider() -> LLMProvider:
    """Create (and cache) the LLM provider. Only OpenAI-compatible is supported."""
    from app.core.config import settings

    if settings.LLM_PROVIDER != "openai":
        raise ValueError(f"LLM_PROVIDER must be 'openai', got '{settings.LLM_PROVIDER}'")

    from app.services.llm.openai import OpenAILLMProvider

    if not settings.LLM_OPENAI_API_KEY:
        raise ValueError("LLM_OPENAI_API_KEY is required when LLM_PROVIDER=openai")

    return OpenAILLMProvider(
        api_key=settings.LLM_OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        base_url=settings.LLM_OPENAI_BASE_URL,
        organization=settings.LLM_OPENAI_ORGANIZATION,
    )


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Create (and cache) the embedding provider. Only OpenAI-compatible is supported."""
    from app.core.config import settings

    if settings.EMBEDDING_PROVIDER != "openai":
        raise ValueError(f"EMBEDDING_PROVIDER must be 'openai', got '{settings.EMBEDDING_PROVIDER}'")

    from app.services.llm.openai import OpenAIEmbeddingProvider

    api_key = settings.EMBEDDING_OPENAI_API_KEY or settings.LLM_OPENAI_API_KEY
    base_url = settings.EMBEDDING_OPENAI_BASE_URL or settings.LLM_OPENAI_BASE_URL
    organization = settings.EMBEDDING_OPENAI_ORGANIZATION or settings.LLM_OPENAI_ORGANIZATION

    if not api_key:
        raise ValueError("EMBEDDING_OPENAI_API_KEY or LLM_OPENAI_API_KEY is required")

    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model=settings.EMBEDDING_MODEL,
        base_url=base_url,
        organization=organization,
    )


__all__ = [
    "get_llm_provider",
    "get_embedding_provider",
    "LLMProvider",
    "EmbeddingProvider",
]