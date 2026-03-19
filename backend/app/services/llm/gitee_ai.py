"""
Gitee AI Sentence Similarity Provider
======================================
Provides reranking functionality using Gitee AI's sentence similarity API.

API Documentation: https://ai.gitee.com/v1/sentence-similarity
"""
from __future__ import annotations

import logging
from typing import Sequence, Optional
import requests
import numpy as np

from app.services.llm.base import EmbeddingProvider
from app.core.config import settings

logger = logging.getLogger(__name__)


class GiteeAIFailure(Exception):
    """Exception raised when Gitee AI API call fails."""
    pass


class GiteeAIRerankerProvider:
    """
    Gitee AI Reranker provider using sentence similarity API.

    This is NOT an EmbeddingProvider - it's a separate provider for reranking.
    """

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
        self.top_n = top_n or settings.GITEE_AI_RERANK_TOP_N

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
            GiteeAIFailure: If API call fails
        """
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
                raise GiteeAIFailure(error_msg)

            result = response.json()

            # API returns a list of scores
            if not isinstance(result, list):
                raise GiteeAIFailure(f"Unexpected response format: {result}")

            return result

        except requests.RequestException as e:
            error_msg = f"Failed to call Gitee AI API: {e}"
            logger.error(error_msg)
            raise GiteeAIFailure(error_msg)

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> list[dict]:
        """
        Rerank documents by relevance to the query.

        Args:
            query: The user's search query
            documents: List of document texts to rerank
            top_k: Maximum number of results to return (None = use default top_n)
            min_score: Minimum relevance score threshold (None = no filtering)

        Returns:
            List of dictionaries with keys: index, score, text
            Sorted by score descending.
        """
        if not documents:
            return []

        # Call API to get scores
        scores = self._call_api(query, documents)

        # Build results
        results = [
            {"index": i, "score": s, "text": doc}
            for i, (s, doc) in enumerate(zip(scores, documents))
        ]

        # Sort by score descending
        results.sort(key=lambda r: r["score"], reverse=True)

        # Apply min_score filter
        if min_score is not None:
            results = [r for r in r in r["score"] >= min_score]

        # Apply top_k limit
        if top_k is not None:
            results = results[:top_k]
        elif self.top_n is not None:
            results = results[:self.top_n]

        return results


# For backward compatibility, also provide embedding provider interface
class GiteeAIEmbeddingProvider(EmbeddingProvider):
    """
    Note: Gitee AI currently does NOT provide embedding API.
    This class exists only for interface compatibility if needed in future.
    """

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError(
            "Gitee AI does not currently support text embeddings. "
            "Use sentence_transformers, openai, gemini, or ollama for embeddings."
        )

    def get_dimension(self) -> int:
        raise NotImplementedError(
            "Gitee AI does not currently support text embeddings."
        )
