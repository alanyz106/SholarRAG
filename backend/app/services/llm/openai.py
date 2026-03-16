"""
OpenAI-Compatible LLM & Embedding Providers
===========================================
Concrete implementations using the ``openai`` SDK.

Supports OpenAI, Azure OpenAI, and any OpenAI-compatible API
(OpenRouter, Together AI, LocalAI, vLLM, etc.) via ``base_url``.
"""
from __future__ import annotations

import logging
from typing import AsyncGenerator, Optional

import numpy as np
from openai import AsyncOpenAI, OpenAI

from app.services.llm.base import EmbeddingProvider, LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk

logger = logging.getLogger(__name__)


class OpenAILLMProvider(LLMProvider):
    """OpenAI-compatible text/multimodal generation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None,
    ):
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_openai_messages(messages: list[LLMMessage]) -> list[dict]:
        """Convert LLMMessage list to OpenAI message dicts."""
        result: list[dict] = []

        for msg in messages:
            entry: dict = {"role": msg.role, "content": ""}

            if msg.content:
                entry["content"] = msg.content

            if msg.images:
                # OpenAI vision format: content can be a list of text/image parts
                entry["content"] = []
                if msg.content:
                    entry["content"].append({"type": "text", "text": msg.content})
                for img in msg.images:
                    # OpenAI expects base64 encoded images with data URI
                    import base64

                    b64 = base64.b64encode(img.data).decode("utf-8")
                    entry["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{img.mime_type};base64,{b64}"},
                        }
                    )

            result.append(entry)

        return result

    # ------------------------------------------------------------------
    # LLMProvider interface
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        openai_msgs = self._to_openai_messages(messages)

        if system_prompt:
            openai_msgs.insert(0, {"role": "system", "content": system_prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=openai_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:
            logger.error(f"OpenAI LLM call failed: {e}")
            return LLMResult(content="") if think else ""

    async def acomplete(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
    ) -> str | LLMResult:
        openai_msgs = self._to_openai_messages(messages)

        if system_prompt:
            openai_msgs.insert(0, {"role": "system", "content": system_prompt})

        try:
            response = await self._async_client.chat.completions.create(
                model=self._model,
                messages=openai_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content or ""
            return content
        except Exception as e:
            logger.error(f"OpenAI async LLM call failed: {e}")
            return LLMResult(content="") if think else ""

    async def astream(
        self,
        messages: list[LLMMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        system_prompt: Optional[str] = None,
        think: bool = False,
        tools: list | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        openai_msgs = self._to_openai_messages(messages)

        if system_prompt:
            openai_msgs.insert(0, {"role": "system", "content": system_prompt})

        try:
            stream = await self._async_client.chat.completions.create(
                model=self._model,
                messages=openai_msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield StreamChunk(type="text", text=chunk.choices[0].delta.content)

                # Handle tool calls if present
                if chunk.choices and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.function:
                            yield StreamChunk(
                                type="function_call",
                                function_call={
                                    "name": tool_call.function.name or "",
                                    "args": {},  # Args come in chunks, need accumulation
                                },
                            )
        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            yield StreamChunk(type="text", text="")

    def supports_vision(self) -> bool:
        """OpenAI GPT-4o and later support vision."""
        # Basic check: gpt-4o, gpt-4-turbo support vision
        vision_models = ["gpt-4o", "gpt-4-turbo", "gpt-4-vision"]
        return any(vm in self._model for vm in vision_models)

    def supports_thinking(self) -> bool:
        """OpenAI does not have an explicit thinking mode."""
        return False


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible text embedding."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        organization: Optional[str] = None,
    ):
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._model = model
        self._dimension: Optional[int] = None
        # Common dimensions:
        # text-embedding-3-small: 1536
        # text-embedding-3-large: 3072
        # text-embedding-ada-002: 1536
        self._known_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def _get_dimension(self) -> int:
        """Get embedding dimension for the model."""
        if self._dimension is not None:
            return self._dimension
        # Try known dimensions first
        for model_name, dim in self._known_dimensions.items():
            if model_name in self._model:
                self._dimension = dim
                return dim
        # Default fallback
        return 1536

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            dim = self._get_dimension()
            return np.zeros((len(texts), dim), dtype=np.float32)

    async def embed(self, texts: list[str]) -> np.ndarray:
        try:
            response = await self._async_client.embeddings.create(
                model=self._model,
                input=texts,
            )
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"OpenAI async embedding failed: {e}")
            dim = self._get_dimension()
            return np.zeros((len(texts), dim), dtype=np.float32)

    def get_dimension(self) -> int:
        return self._get_dimension()
