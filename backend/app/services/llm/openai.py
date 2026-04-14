"""
OpenAI-Compatible LLM & Embedding Providers
===========================================
Concrete implementations using the ``openai`` SDK.

Supports OpenAI, Azure OpenAI, and any OpenAI-compatible API
(OpenRouter, Together AI, LocalAI, vLLM, etc.) via ``base_url``.
"""
from __future__ import annotations

import logging
import re
from typing import AsyncGenerator, Optional

import numpy as np
from openai import AsyncOpenAI, OpenAI

from app.services.llm.base import EmbeddingProvider, LLMProvider
from app.services.llm.types import LLMMessage, LLMResult, StreamChunk

logger = logging.getLogger(__name__)

# Regex to match thinking blocks: <think>...</think> (both Chinese and English brackets)
_THINK_RE = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)


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

    def _extract_thinking(self, content: str) -> tuple[str, str]:
        """Extract thinking blocks and clean content.

        Returns (clean_content, thinking_text).
        """
        thinking_parts = _THINK_RE.findall(content)
        thinking = "".join(thinking_parts)
        clean_content = _THINK_RE.sub("", content).strip()
        return clean_content, thinking

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
            logger.debug(f"MiniMax complete raw: {repr(content[:500])}")

            clean_content, thinking = self._extract_thinking(content)
            if think and thinking:
                return LLMResult(content=clean_content, thinking=thinking)
            return clean_content
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
            logger.debug(f"MiniMax acomplete raw: {repr(content[:500])}")

            clean_content, thinking = self._extract_thinking(content)
            if think and thinking:
                return LLMResult(content=clean_content, thinking=thinking)
            return clean_content
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

        buffer = ""  # Buffer to accumulate content

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
                    content = chunk.choices[0].delta.content
                    buffer += content
                    logger.debug(f"MiniMax stream chunk: {repr(content)}")

                    # Process buffer to extract thinking blocks
                    while True:
                        m = _THINK_RE.search(buffer)
                        if m:
                            # Text before the thinking block
                            before = buffer[:m.start()]
                            if before:
                                yield StreamChunk(type="text", text=before)
                            # The thinking content
                            yield StreamChunk(type="thinking", text=m.group(1))
                            # Remove processed content
                            buffer = buffer[m.end():]
                        else:
                            # No complete block found
                            # Check if we're waiting for a partial tag
                            if "<think>" in buffer:
                                # Opening tag incomplete, wait for more
                                break
                            elif buffer.endswith("<") or buffer.endswith("<t") or buffer.endswith("<th"):
                                break
                            else:
                                # No thinking tags, yield as text
                                if buffer:
                                    yield StreamChunk(type="text", text=buffer)
                                    buffer = ""
                                break

                # Handle tool calls if present
                if chunk.choices and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        if tool_call.function:
                            yield StreamChunk(
                                type="function_call",
                                function_call={
                                    "name": tool_call.function.name or "",
                                    "args": {},
                                },
                            )

            # Flush remaining buffer at end of stream
            if buffer:
                yield StreamChunk(type="text", text=buffer)

        except Exception as e:
            logger.error(f"OpenAI streaming failed: {e}")
            yield StreamChunk(type="text", text="")

    def supports_vision(self) -> bool:
        """OpenAI GPT-4o and later support vision."""
        vision_models = ["gpt-4o", "gpt-4-turbo", "gpt-4-vision"]
        return any(vm in self._model for vm in vision_models)

    def supports_thinking(self) -> bool:
        """OpenAI-compatible models (MiniMax, etc.) may support thinking via <think> tags."""
        return True


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
        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
        )
        self._model = model
        self._dimension: Optional[int] = None
        self._known_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        if model in self._known_dimensions:
            self._dimension = self._known_dimensions[model]
        else:
            for name, dim in self._known_dimensions.items():
                if name in model:
                    self._dimension = dim
                    break
            if "bge-m3" in model.lower():
                self._dimension = 1024
            elif "bge-large-zh" in model.lower() or "bge-large-en" in model.lower():
                self._dimension = 1024

    def _get_dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        for model_name, dim in self._known_dimensions.items():
            if model_name in self._model:
                self._dimension = dim
                return dim
        return 1536

    def embed_sync(self, texts: list[str]) -> np.ndarray:
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
            )
            embeddings = [item.embedding for item in response.data]
            arr = np.array(embeddings, dtype=np.float32)
            if self._dimension is None and arr.size > 0:
                self._dimension = arr.shape[1]
                logger.info(f"Detected embedding dimension: {self._dimension}")
            return arr
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
            arr = np.array(embeddings, dtype=np.float32)
            if self._dimension is None and arr.size > 0:
                self._dimension = arr.shape[1]
                logger.info(f"Detected embedding dimension: {self._dimension}")
            return arr
        except Exception as e:
            logger.error(f"OpenAI async embedding failed: {e}")
            dim = self._get_dimension()
            return np.zeros((len(texts), dim), dtype=np.float32)

    def get_dimension(self) -> int:
        return self._get_dimension()
