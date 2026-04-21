"""
聊天代理 —— NexusRAG 半自主 SSE 流式响应
====================================================

提供 SSE 流式端点，LLM 决定是否调用 ``search_documents`` 或直接回答，
实时流式输出思考过程和令牌。

SSE 事件类型:
  - status:         {"step": str, "detail": str}
  - thinking:       {"text": str}
  - sources:        {"sources": [...]}
  - images:         {"image_refs": [...]}
  - token:          {"text": str}
  - token_rollback: {}
  - complete:       {"answer": str, "sources": [...], ...}
  - error:          {"message": str}
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import string
import uuid
from typing import AsyncGenerator, Optional

from fastapi import Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.models.knowledge_base import KnowledgeBase
from app.schemas.rag import (
    ChatRequest,
    ChatSourceChunk,
    ChatImageRef,
)
from app.services.llm.types import LLMMessage, LLMImagePart, StreamChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_AGENT_ITERATIONS = 3
MAX_VISION_IMAGES = 3
SSE_HEARTBEAT_INTERVAL = 15  # seconds

_CITATION_ID_CHARS = string.ascii_lowercase + string.digits


def _generate_citation_id(existing: set[str]) -> str:
    """Generate a unique 4-char alphanumeric citation ID."""
    while True:
        cid = "".join(random.choices(_CITATION_ID_CHARS, k=4))
        if any(c.isalpha() for c in cid) and cid not in existing:
            return cid


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

def _get_openai_tool():
    """Create OpenAI-compatible tool definition."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": (
                    "搜索知识库中相关的文档片段。"
                    "当用户询问文档内容、数据或事实时使用此工具。"
                    "重要：将用户的问题重写为详细、具体的搜索查询"
                    "以获得更好的检索结果。"
                    "不要将此工具用于问候、闲聊或非文档问题。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "基于用户问题重写的详细搜索查询。"
                                "示例：'revenue?' → '总收入数字和财务绩效指标'。"
                                "'AI 是什么？' → '人工智能的定义、历史和应用'"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "要检索的相关片段数量（默认：5，最大：10）",
                        },
                    },
                    "required": ["query"],
                },
            },
        }
    ]


# ---------------------------------------------------------------------------
# SSE Helpers (ported from PageIndex backend/app/api/v1/chat.py)
# ---------------------------------------------------------------------------

def format_sse_event(event: str, data: dict) -> str:
    """Format data as an SSE event string."""
    json_data = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {json_data}\n\n"


async def sse_with_heartbeat(
    source: AsyncGenerator[str, None],
) -> AsyncGenerator[str, None]:
    """Wrap an SSE generator with periodic heartbeat comments.

    SSE spec allows lines starting with ':' as comments — browsers/clients
    silently ignore them but they keep the TCP connection alive, preventing
    timeouts when the upstream LLM takes a long time to respond.
    """
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _pump():
        try:
            async for event in source:
                await queue.put(event)
        except Exception:
            pass
        finally:
            await queue.put(None)  # sentinel

    task = asyncio.create_task(_pump())
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=SSE_HEARTBEAT_INTERVAL
                )
                if event is None:
                    break
                yield event
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


# ---------------------------------------------------------------------------
# Tool executor — retrieval via retrieval_service
# ---------------------------------------------------------------------------

def _build_tool_result_content(
    context: str,
    message: str,
    img_parts: list[dict],
) -> tuple[str, list[LLMImagePart]]:
    """Build tool result content string and image parts from retrieval result."""
    tool_result_parts = [
        "我已为您检索到以下文档来源。\n",
        "=== 文档来源 ===",
        context,
        "=== 来源结束 ===\n",
        "重要提示:\n"
        "- 仔细阅读上述每一个来源。答案通常需要结合多个来源的数据。\n"
        "- 表格数据：来源可能包含表格数据，格式为 '键, 年份 = 值' 对。"
        "示例：'ROE, 2023 = 12,8%' 表示 2023 年的 ROE 为 12.8%。\n"
        "- 如果没有来源包含相关信息，请说："
        "\"文档不包含此信息。\"\n",
    ]
    tool_result_content = "\n".join(tool_result_parts)

    user_images: list[LLMImagePart] = []
    if img_parts:
        for img_data in img_parts:
            tool_result_content += f"\n[IMG-{img_data['img_ref_id']}] (第 {img_data['page_no']} 页):"
            user_images.append(LLMImagePart(
                data=img_data["inline_data"]["data"],
                mime_type=img_data["inline_data"]["mime_type"],
            ))

    tool_result_content += f"\n\n现在回答问题: {message}"
    return tool_result_content, user_images


async def agent_chat_stream(
    workspace_id: int,
    message: str,
    history: list[dict],
    enable_thinking: bool,
    db: AsyncSession,
    system_prompt: str,
    force_search: bool = False,
) -> AsyncGenerator[dict, None]:
    """半智能体聊天流。

    - force_search=True: 在调用 LLM 之前进行预搜索，将来源作为上下文注入。
      无论模型是否具备工具调用能力，都能保证每次查询都进行检索。
    - force_search=False (默认): 智能体工具调用循环。

    生成包含 'event' 和 'data' 键的字典，用于 SSE 格式化。
    """
    from app.services.llm import get_llm_provider
    from app.core.config import settings
    from app.services.retrieval_service import retrieve_documents

    provider = get_llm_provider()

    existing_ids: set[str] = set()
    all_sources: list[ChatSourceChunk] = []
    all_images: list[ChatImageRef] = []
    all_image_parts: list[dict] = []

    # 构建对话消息
    messages: list[LLMMessage] = []
    for msg in history[-10:]:
        role = "user" if msg["role"] == "user" else "assistant"
        messages.append(LLMMessage(role=role, content=msg["content"]))

    # 构建用户消息
    messages.append(LLMMessage(role="user", content=message))

    # 工具 / 提示词设置
    tools = _get_openai_tool()

    if force_search:
        # ── 强制搜索模式：在 LLM 调用前进行预搜索 ──────────────────
        yield {"event": "status", "data": {"step": "retrieving", "detail": f"正在搜索: {message[:80]}..."}}

        result = await retrieve_documents(
            workspace_id, message, 8, db, existing_ids,
        )
        all_sources.extend(result.sources)
        all_images.extend(result.image_refs)
        all_image_parts.extend(result.image_parts)

        if result.sources:
            yield {"event": "sources", "data": {"sources": [s.model_dump() for s in result.sources]}}
        if result.image_refs:
            yield {"event": "images", "data": {"image_refs": [i.model_dump() for i in result.image_refs]}}

        if result.sources:
            tool_result_content, user_images_fs = _build_tool_result_content(
                result.context, message, result.image_parts,
            )
            messages.append(LLMMessage(
                role="user",
                content=tool_result_content,
                images=user_images_fs,
            ))
        # tools 保持为 None — 模型直接使用提供的上下文进行回答

    yield {"event": "status", "data": {"step": "analyzing", "detail": "正在分析您的问题..."}}

    accumulated_text = ""
    thinking_text = ""

    for iteration in range(MAX_AGENT_ITERATIONS):
        iteration_text = ""
        function_calls: list[dict] = []
        tokens_yielded = False

        async for chunk in provider.astream(
            messages,
            temperature=0.1,
            max_tokens=settings.LLM_MAX_TOKENS,
            system_prompt=system_prompt,
            think=enable_thinking,
            tools=tools,
        ):
            if chunk.type == "thinking":
                thinking_text += chunk.text
                yield {"event": "thinking", "data": {"text": chunk.text}}
            elif chunk.type == "function_call":
                function_calls.append(chunk.function_call)
            elif chunk.type == "text":
                iteration_text += chunk.text
                # 推测性流式传输 —— 如果尚未看到工具调用则发送令牌
                if not function_calls:
                    accumulated_text += chunk.text
                    tokens_yielded = True
                    yield {"event": "token", "data": {"text": chunk.text}}

        if function_calls:
            # 回滚推测性令牌
            if tokens_yielded:
                accumulated_text = ""
                yield {"event": "token_rollback", "data": {}}

            fc = function_calls[0]
            fc_name = fc.get("name", "")
            fc_args = fc.get("args", {})

            if fc_name == "search_documents":
                query = fc_args.get("query", message)
                top_k = int(fc_args.get("top_k", 8))

                yield {"event": "status", "data": {
                    "step": "retrieving",
                    "detail": f"正在搜索: {query[:80]}..."
                }}

                result = await retrieve_documents(
                    workspace_id, query, top_k, db, existing_ids,
                )
                all_sources.extend(result.sources)
                all_images.extend(result.image_refs)
                all_image_parts.extend(result.image_parts)

                if result.sources:
                    yield {"event": "sources", "data": {
                        "sources": [s.model_dump() for s in result.sources]
                    }}
                if result.image_refs:
                    yield {"event": "images", "data": {
                        "image_refs": [i.model_dump() for i in result.image_refs]
                    }}

                tool_result_content, user_images = _build_tool_result_content(
                    result.context, message, result.image_parts,
                )

                # 添加助手消息（调用摘要）+ 用户消息（来源上下文）
                messages.append(LLMMessage(
                    role="assistant",
                    content=f"[调用了 search_documents(query=\"{query}\")]",
                ))
                messages.append(LLMMessage(
                    role="user",
                    content=tool_result_content,
                    images=user_images,
                ))

                yield {"event": "status", "data": {
                    "step": "generating",
                    "detail": "正在生成答案..."
                }}
            else:
                # 未知工具 —— 将累积文本视为答案
                logger.warning(f"未知工具调用: {fc_name}")
                break
        else:
            # 模型未进行工具调用 —— 答案在 accumulated_text 中，完成。
            break

    # ── 回退：模型未生成文本且未进行搜索 ──────────────────────────
    if not accumulated_text and not all_sources:
        logger.warning("模型未生成文本且未进行工具调用 —— 回退到自动搜索")
        yield {"event": "status", "data": {
            "step": "retrieving",
            "detail": f"正在搜索: {message[:80]}..."
        }}

        result = await retrieve_documents(
            workspace_id, message, 8, db, existing_ids,
        )
        all_sources.extend(result.sources)
        all_images.extend(result.image_refs)
        all_image_parts.extend(result.image_parts)

        if result.sources:
            yield {"event": "sources", "data": {
                "sources": [s.model_dump() for s in result.sources]
            }}
        if result.image_refs:
            yield {"event": "images", "data": {
                "image_refs": [i.model_dump() for i in result.image_refs]
            }}

        if result.sources:
            fallback_parts = [
                "我已为您检索到以下文档来源。\n",
                "=== 文档来源 ===",
                result.context,
                "=== 来源结束 ===\n",
                "重要提示:\n"
                "- 仔细阅读上述每一个来源。\n"
                "- 如果没有来源包含相关信息，请说："
                "\"文档不包含此信息。\"\n",
            ]
            fallback_content = "\n".join(fallback_parts)
            fallback_content += f"\n\n现在回答问题: {message}"

            fallback_msgs = messages.copy()
            fallback_msgs.append(LLMMessage(role="user", content=fallback_content))

            yield {"event": "status", "data": {
                "step": "generating", "detail": "正在生成答案..."
            }}

            async for chunk in provider.astream(
                fallback_msgs,
                temperature=0.1,
                max_tokens=settings.LLM_MAX_TOKENS,
                system_prompt=system_prompt,
                think=enable_thinking,
                tools=None,
            ):
                if chunk.type == "thinking":
                    thinking_text += chunk.text
                    yield {"event": "thinking", "data": {"text": chunk.text}}
                elif chunk.type == "text":
                    accumulated_text += chunk.text
                    yield {"event": "token", "data": {"text": chunk.text}}

    # 从知识图谱中提取相关实体
    related_entities: list[str] = []
    try:
        from app.api.rag import _get_kg_service
        kg = await _get_kg_service(workspace_id)
        entities = await kg.get_entities(limit=200)
        entity_names = {e["name"].lower(): e["name"] for e in entities}
        text_lower = accumulated_text.lower()
        for lower_name, original_name in entity_names.items():
            if len(lower_name) >= 2 and lower_name in text_lower:
                related_entities.append(original_name)
    except Exception:
        pass

    # 去除伪影
    if accumulated_text:
        accumulated_text = re.sub(r'<unused\d+>:?\s*', '', accumulated_text).strip()

    yield {"event": "complete", "data": {
        "answer": accumulated_text or "无法生成响应。",
        "sources": [s.model_dump() for s in all_sources],
        "image_refs": [i.model_dump() for i in all_images],
        "thinking": thinking_text or None,
        "related_entities": related_entities[:30],
    }}

# ---------------------------------------------------------------------------
# SSE Streaming endpoint
# ---------------------------------------------------------------------------

async def chat_stream_endpoint(
    workspace_id: int,
    request: ChatRequest,
    db: AsyncSession,
):
    """SSE streaming chat endpoint.

    Called from rag.py router — not a standalone router to avoid circular imports.
    """
    # Verify workspace
    result = await db.execute(
        select(KnowledgeBase).where(KnowledgeBase.id == workspace_id)
    )
    kb = result.scalar_one_or_none()
    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found",
        )

    # Build system prompt
    from app.api.chat_prompt import DEFAULT_SYSTEM_PROMPT, HARD_SYSTEM_PROMPT
    system_prompt = (kb.system_prompt or DEFAULT_SYSTEM_PROMPT) + HARD_SYSTEM_PROMPT

    # Build history
    history = [{"role": m.role, "content": m.content} for m in request.history]

    # Persist user message immediately
    try:
        from app.models.chat_message import ChatMessage as ChatMessageModel
        user_row = ChatMessageModel(
            workspace_id=workspace_id,
            message_id=str(uuid.uuid4()),
            role="user",
            content=request.message,
        )
        db.add(user_row)
        await db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist user message: {e}")
        await db.rollback()

    async def event_generator() -> AsyncGenerator[str, None]:
        final_answer = ""
        final_sources = []
        final_images = []
        final_thinking = None
        final_entities = []

        # Collect agent steps for persistence (ThinkingTimeline survives reload)
        collected_steps: list[dict] = []
        step_counter = 0
        # Track sources/images as they arrive so sources_found inserts BEFORE generating
        streaming_sources: list[dict] = []
        streaming_images: list[dict] = []

        try:
            async for event in agent_chat_stream(
                workspace_id=workspace_id,
                message=request.message,
                history=history,
                enable_thinking=request.enable_thinking,
                db=db,
                system_prompt=system_prompt,
                force_search=request.force_search,
            ):
                event_type = event["event"]
                event_data = event["data"]

                # Collect status steps; insert sources_found before "generating"
                if event_type == "status":
                    step_name = event_data.get("step", "analyzing")

                    # When generating starts, insert sources_found first (correct order)
                    if step_name == "generating" and streaming_sources:
                        step_counter += 1
                        badges = list(dict.fromkeys(
                            s.get("index", "") for s in streaming_sources[:6]
                        ))
                        collected_steps.append({
                            "id": f"step-{step_counter}",
                            "step": "sources_found",
                            "detail": f"Found {len(streaming_sources)} source{'s' if len(streaming_sources) != 1 else ''}",
                            "status": "completed",
                            "timestamp": 0,
                            "sourceCount": len(streaming_sources),
                            "imageCount": len(streaming_images),
                            "sourceBadges": badges,
                        })
                        streaming_sources.clear()
                        streaming_images.clear()

                    step_counter += 1
                    collected_steps.append({
                        "id": f"step-{step_counter}",
                        "step": step_name,
                        "detail": event_data.get("detail", ""),
                        "status": "completed",
                        "timestamp": 0,
                    })

                # Track sources/images as they arrive
                elif event_type == "sources":
                    streaming_sources.extend(event_data.get("sources", []))

                elif event_type == "images":
                    streaming_images.extend(event_data.get("image_refs", []))

                # Attach thinking text to the analyzing step
                elif event_type == "thinking":
                    thinking_fragment = event_data.get("text", "")
                    for s in collected_steps:
                        if s["step"] == "analyzing":
                            s["thinkingText"] = (s.get("thinkingText") or "") + thinking_fragment
                            break

                elif event_type == "complete":
                    final_answer = event_data.get("answer", "")
                    final_sources = event_data.get("sources", [])
                    final_images = event_data.get("image_refs", [])
                    final_thinking = event_data.get("thinking")
                    final_entities = event_data.get("related_entities", [])

                    # Fallback: if sources arrived but generating step was never emitted
                    if streaming_sources and not any(s["step"] == "sources_found" for s in collected_steps):
                        step_counter += 1
                        badges = list(dict.fromkeys(
                            s.get("index", "") for s in streaming_sources[:6]
                        ))
                        collected_steps.append({
                            "id": f"step-{step_counter}",
                            "step": "sources_found",
                            "detail": f"Found {len(streaming_sources)} source{'s' if len(streaming_sources) != 1 else ''}",
                            "status": "completed",
                            "timestamp": 0,
                            "sourceCount": len(streaming_sources),
                            "imageCount": len(streaming_images),
                            "sourceBadges": badges,
                        })

                    # Done step
                    step_counter += 1
                    collected_steps.append({
                        "id": f"step-{step_counter}",
                        "step": "done",
                        "detail": "Done",
                        "status": "completed",
                        "timestamp": 0,
                    })

                yield format_sse_event(event_type, event_data)

        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield format_sse_event("error", {"message": str(e)})
        finally:
            # Persist assistant message
            if final_answer:
                try:
                    from app.models.chat_message import ChatMessage as ChatMessageModel
                    assistant_row = ChatMessageModel(
                        workspace_id=workspace_id,
                        message_id=str(uuid.uuid4()),
                        role="assistant",
                        content=final_answer,
                        sources=final_sources if final_sources else None,
                        related_entities=final_entities[:30] if final_entities else None,
                        image_refs=final_images if final_images else None,
                        thinking=final_thinking,
                        agent_steps=collected_steps if collected_steps else None,
                    )
                    db.add(assistant_row)
                    await db.commit()
                except Exception as e:
                    logger.warning(f"Failed to persist assistant message: {e}")
                    await db.rollback()

    return StreamingResponse(
        sse_with_heartbeat(event_generator()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
