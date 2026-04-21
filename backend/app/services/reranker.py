"""
Reranker Service
================
Cross-encoder reranker for improving retrieval precision.

Supports multiple providers:
- gitee_ai (Gitee AI API)
- siliconflow (SiliconFlow API)

Default provider: RERANKER_PROVIDER from settings (default: gitee_ai).

Usage:
    reranker = get_reranker_service()
    ranked = reranker.rerank("user question", ["chunk1", "chunk2", ...], top_k=5)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, List, Dict, Any

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RerankResult:
    """A single reranked item with its original index and relevance score."""
    index: int          # Original position in the input list
    score: float        # Cross-encoder relevance score (higher = more relevant)
    text: str           # The chunk text


class BaseRerankerProvider(ABC):
    """Abstract base class for reranker providers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[RerankResult]:
        """Rerank documents by relevance to the query."""
        ...


class GiteeAIRerankerProvider(BaseRerankerProvider):
    """Gitee AI sentence similarity API reranker."""

    API_URL = "https://ai.gitee.com/v1/sentence-similarity"

    def __init__(
        self,
        api_token: Optional[str] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ):
        """
        Initialize Gitee AI reranker provider.

        Args:
            api_token: Gitee AI API token
            model: Model name (default: bge-reranker-v2-m3)
            top_n: Default top N results to return
        """
        self.api_token = api_token or settings.GITEE_AI_API_TOKEN
        self.model = model or settings.GITEE_AI_RERANK_MODEL
        self.top_n = top_n if top_n is not None else settings.GITEE_AI_RERANK_TOP_N

        if not self.api_token:
            raise ValueError(
                "GITEE_AI_API_TOKEN is required. Set it in .env file or pass it explicitly."
            )

    def _call_api(
        self,
        query: str,
        sentences: Sequence[str],
        model: Optional[str] = None,
    ) -> list[float]:
        """
        Call Gitee AI sentence similarity API.

        Args:
            query: The source/query sentence
            sentences: List of candidate sentences to score
            model: Model name (uses default if not specified)

        Returns:
            List of relevance scores for each sentence

        Raises:
            RuntimeError: If API call fails
        """
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Failover-Enabled": "true"
        }

        payload = {
            "inputs": {
                "source_sentence": query,
                "sentences": list(sentences)
            },
            "model": model or self.model
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                error_msg = f"Gitee AI API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            result = response.json()

            # API returns a list of scores
            if not isinstance(result, list):
                raise RuntimeError(f"Unexpected response format: {result}")

            return result

        except requests.RequestException as e:
            error_msg = f"Failed to call Gitee AI API: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents using Gitee AI API.

        Args:
            query: The user's search query
            documents: List of document texts to rerank
            top_k: Maximum number of results to return (None = use default top_n)
            min_score: Minimum relevance score threshold (None = no filtering)

        Returns:
            List of RerankResult sorted by score (descending).
        """
        if not documents:
            return []

        # Call API to get scores
        scores = self._call_api(query, documents)

        # Build results
        results = [
            RerankResult(index=i, score=s, text=doc)
            for i, (s, doc) in enumerate(zip(scores, documents))
        ]

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply min_score filter
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]
        elif self.top_n is not None:
            results = results[:self.top_n]

        return results


class SiliconFlowRerankerProvider(BaseRerankerProvider):
    """SiliconFlow reranker API provider."""

    API_URL = "https://api.siliconflow.cn/v1/rerank"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_n: Optional[int] = None,
    ):
        """
        Initialize SiliconFlow reranker provider.

        Args:
            api_key: SiliconFlow API key
            model: Model name (default: BAAI/bge-reranker-v2-m3)
            top_n: Default top N results to return
        """
        self.api_key = api_key or settings.SILICONFLOW_API_KEY
        self.model = model or settings.SILICONFLOW_RERANK_MODEL
        self.top_n = top_n if top_n is not None else settings.SILICONFLOW_RERANK_TOP_N

        if not self.api_key:
            raise ValueError(
                "SILICONFLOW_API_KEY is required. Set it in .env file or pass it explicitly."
            )

    def _call_api(
        self,
        query: str,
        documents: Sequence[str],
        model: Optional[str] = None,
    ) -> list[float]:
        """
        Call SiliconFlow rerank API.

        Args:
            query: The query sentence
            documents: List of candidate documents to score
            model: Model name (uses default if not specified)

        Returns:
            List of relevance scores for each document

        Raises:
            RuntimeError: If API call fails
        """
        import requests

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model or self.model,
            "query": query,
            "documents": list(documents)
        }

        try:
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                error_msg = f"SiliconFlow API error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            result = response.json()

            # API returns format: {"results": [{"index": i, "relevance_score": s}, ...]}
            # We need to extract scores in the original order
            if "results" not in result:
                raise RuntimeError(f"Unexpected response format: {result}")

            # Create a list of scores with original indices
            scores_by_index = {item["index"]: item["relevance_score"] for item in result["results"]}

            # Build scores list in the original document order
            scores = [scores_by_index.get(i, 0.0) for i in range(len(documents))]

            return scores

        except requests.RequestException as e:
            error_msg = f"Failed to call SiliconFlow API: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents using SiliconFlow API.

        Args:
            query: The user's search query
            documents: List of document texts to rerank
            top_k: Maximum number of results to return (None = all)
            min_score: Minimum relevance score threshold (None = no filtering)

        Returns:
            List of RerankResult sorted by score (descending).
        """
        if not documents:
            return []

        # Call API to get scores
        scores = self._call_api(query, documents)

        # Build results
        results = [
            RerankResult(index=i, score=s, text=doc)
            for i, (s, doc) in enumerate(zip(scores, documents))
        ]

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        # Apply min_score filter
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]
        elif self.top_n is not None:
            results = results[:self.top_n]

        return results


class RerankerService:
    """
    Facade service that delegates to the configured provider.
    Supports multiple reranker providers through a common interface.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize reranker service with specified provider.

        Args:
            provider: Reranker provider (gitee_ai, siliconflow)
                     Defaults to settings.RERANKER_PROVIDER
            model_name: Model name (provider-specific)
            **kwargs: Additional provider-specific arguments (api_token, top_n, etc.)
        """
        self.provider = provider or settings.RERANKER_PROVIDER

        if self.provider == "gitee_ai":
            self._provider = GiteeAIRerankerProvider(
                api_token=kwargs.get("api_token") or settings.GITEE_AI_API_TOKEN,
                model=model_name or settings.GITEE_AI_RERANK_MODEL,
                top_n=kwargs.get("top_n", settings.GITEE_AI_RERANK_TOP_N)
            )
        elif self.provider == "siliconflow":
            self._provider = SiliconFlowRerankerProvider(
                api_key=kwargs.get("api_key") or settings.SILICONFLOW_API_KEY,
                model=model_name or settings.SILICONFLOW_RERANK_MODEL,
                top_n=kwargs.get("top_n", settings.SILICONFLOW_RERANK_TOP_N)
            )
        else:
            raise ValueError(
                f"Unknown RERANKER_PROVIDER: {self.provider!r}. "
                f"Supported: gitee_ai, siliconflow"
            )

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[RerankResult]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The user's search query
            documents: List of document texts to rerank
            top_k: Maximum number of results to return (None = all)
            min_score: Minimum relevance score threshold (None = no filtering)

        Returns:
            List of RerankResult sorted by score (descending),
            filtered by top_k and min_score.
        """
        return self._provider.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
            min_score=min_score,
        )


# Singleton instance
_default_service: Optional[RerankerService] = None


def get_reranker_service() -> RerankerService:
    """Get or create the default reranker service."""
    global _default_service
    if _default_service is None:
        _default_service = RerankerService()
    return _default_service