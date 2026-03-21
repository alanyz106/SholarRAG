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
from app.models.document import DocumentImage
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

# Gemini native function calling
def _get_gemini_tool():
    """Lazily create Gemini Tool to avoid import at module level."""
    from google.genai import types
    return types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="search_documents",
            description=(
                "搜索知识库中相关的文档片段。"
                "当用户询问文档内容、数据或事实时使用此工具。"
                "重要：将用户的问题重写为详细、具体的搜索查询"
                "以获得更好的检索结果。"
                "不要将此工具用于问候、闲聊或非文档问题。"
            ),
            parameters={
                "type": "OBJECT",
                "properties": {
                    "query": {
                        "type": "STRING",
                        "description": (
                            "基于用户问题重写的详细搜索查询。"
                            "示例：'revenue?' → '总收入数字和财务绩效指标'。"
                            "'AI 是什么？' → '人工智能的定义、历史和应用'"
                        ),
                    },
                    "top_k": {
                        "type": "INTEGER",
                        "description": "Number of relevant chunks to retrieve (default: 5, max: 10)",
                    },
                },
                "required": ["query"],
            },
        ),
    ])


# OpenAI-compatible function calling
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
# Ollama 基于提示词的工具调用 — 回答前必须进行搜索
# ---------------------------------------------------------------------------

OLLAMA_TOOL_SYSTEM = """\
## 工具：search_documents

你拥有一个工具：search_documents。你需要通过输出以下内容来调用它：

<tool_call>{"name": "search_documents", "arguments": {"query": "<改写后的查询>"}}</tool_call>

### 绝对规则（违反将导致严重错误）

1. **除了简单的对话消息外，始终优先调用 search_documents。**
   不需要工具调用的简单对话消息包括：
   - 问候语："hello"、"你好"、"hi"、"hey"、"早上好"等。
   - 确认语："谢谢"、"感谢"、"thanks"、"ok"、"明白了"等。
   - 告别语："bye"、"goodbye"、"再见"、"拜拜"等。
   对于所有其他消息 —— 问题、请求、事实查询、分析 —— 你必须在回答前调用 search_documents。你的知识不可靠；只有文档来源才是可信的。如果你不确定某条消息是否需要搜索，请进行搜索。

2. **你对可搜索查询的首次完整响应必须仅为 <tool_call> 代码块。**
   前面不要有任何文字。后面不要有任何文字。不要解释。只需工具调用。

3. **改写查询**，使其具体且详细。
   "收入" → "净收入、年度总收入、收入增长"
   "AI 模型" → "AI 模型架构、性能基准、训练细节"

4. 收到搜索结果后，仅使用这些来源进行回答并添加引用。
   格式：声明文本[source_id]。示例：收入达到 4850 亿越南盾[a3x9]。
"""

OLLAMA_TOOL_REMINDER = (
    "\n\n[系统提醒] 如果这是一个问题或请求，你必须首先调用 search_documents。"
    "仅输出：<tool_call>{\"name\": \"search_documents\", \"arguments\": {\"query\": \"...\"}}</tool_call> "
    "例外：简单的问候、感谢或告别不需要工具调用 —— 直接回复。"
    "对于其他所有内容，搜索是强制性的。"
)

# ---------------------------------------------------------------------------
# Gemini 系统提示词强化 — 强制要求对问题进行工具调用
# ---------------------------------------------------------------------------

GEMINI_TOOL_SYSTEM = """\

## 工具使用（强制）

你拥有一个名为 `search_documents` 的工具，用于搜索知识库。

### 绝对规则：
1. 对于所有用户问题、请求、事实查询或分析 —— 你必须在回答前首先调用 \
`search_documents`。即使对话历史中包含相关信息，你也必须再次搜索以获取最新、准确的来源。
2. 仅对简单的对话消息跳过工具调用：
   - 问候语："hello"、"你好"、"hi"、"hey"等。
   - 确认语："谢谢"、"感谢"、"thanks"、"ok"等。
   - 告别语："bye"、"goodbye"、"再见"、"拜拜"等。
3. 永远不要在不搜索的情况下使用之前轮次的信息来回答问题。 \
你之前的回答可能包含过时或不完整的信息。
4. 永远不要重复使用之前回答中的引用 ID。每个回答必须有自己的 \
来自新搜索的引用来源。
5. 将用户的查询改写为具体且详细的，以便更好地检索。
"""


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
# Tool executor — retrieval via NexusRAG
# ---------------------------------------------------------------------------

async def _execute_search_documents(
    workspace_id: int,
    query: str,
    top_k: int,
    db: AsyncSession,
    existing_ids: set[str],
) -> tuple[str, list[ChatSourceChunk], list[ChatImageRef], list[dict]]:
    """Execute document search and return formatted context + structured sources.

    Returns:
        (context_text, sources, image_refs, image_parts_for_vision)
    """
    from app.services.rag_service import get_rag_service
    from app.services.nexus_rag_service import NexusRAGService
    from pathlib import Path as _P
    from app.core.config import settings

    rag_service = get_rag_service(db, workspace_id)

    chunks = []
    citations = []
    if isinstance(rag_service, NexusRAGService):
        result = await rag_service.query_deep(
            question=query,
            top_k=min(top_k, 10),
            mode="hybrid",
            include_images=False,
        )
        chunks = result.chunks
        citations = result.citations
    else:
        from types import SimpleNamespace
        legacy = rag_service.query(question=query, top_k=min(top_k, 10))
        for i, c in enumerate(legacy.chunks):
            chunks.append(SimpleNamespace(
                content=c.content,
                document_id=int(c.metadata.get("document_id", 0)),
                chunk_index=i,
                page_no=int(c.metadata.get("page_no", 0)),
                heading_path=str(c.metadata.get("heading_path", "")).split(" > ") if c.metadata.get("heading_path") else [],
                source_file=str(c.metadata.get("source", "")),
                image_refs=[],
            ))

    # Build sources
    sources: list[ChatSourceChunk] = []
    context_parts: list[str] = []
    for i, chunk in enumerate(chunks):
        citation = citations[i] if i < len(citations) else None
        cid = _generate_citation_id(existing_ids)
        existing_ids.add(cid)
        sources.append(ChatSourceChunk(
            index=cid,
            chunk_id=f"doc_{chunk.document_id}_chunk_{chunk.chunk_index}",
            content=chunk.content,
            document_id=chunk.document_id,
            page_no=chunk.page_no,
            heading_path=chunk.heading_path,
            score=0.0,
            source_type="vector",
        ))
        meta_parts = []
        if citation:
            meta_parts.append(citation.source_file)
            if citation.page_no:
                meta_parts.append(f"page {citation.page_no}")
        heading = " > ".join(chunk.heading_path) if chunk.heading_path else ""
        if heading:
            meta_parts.append(heading)
        meta_line = f" ({', '.join(meta_parts)})" if meta_parts else ""
        context_parts.append(f"Source [{cid}]{meta_line}:\n{chunk.content}")

    context = "\n\n---\n\n".join(context_parts)

    # Build image references
    seen_image_ids: set[str] = set()
    chunk_image_ids: list[str] = []
    for c in chunks:
        for iid in getattr(c, "image_refs", []) or []:
            if iid and iid not in seen_image_ids:
                seen_image_ids.add(iid)
                chunk_image_ids.append(iid)

    resolved_images: list[DocumentImage] = []
    if chunk_image_ids:
        img_result = await db.execute(
            select(DocumentImage).where(DocumentImage.image_id.in_(chunk_image_ids))
        )
        resolved_images = list(img_result.scalars().all())

    if not resolved_images:
        source_pages = {
            (getattr(c, "document_id", 0), getattr(c, "page_no", 0))
            for c in chunks if getattr(c, "page_no", 0) > 0
        }
        if source_pages:
            from sqlalchemy import or_, and_
            page_filters = [
                and_(
                    DocumentImage.document_id == doc_id,
                    DocumentImage.page_no == page_no,
                )
                for doc_id, page_no in source_pages
            ]
            img_result = await db.execute(
                select(DocumentImage).where(or_(*page_filters))
            )
            resolved_images = list(img_result.scalars().all())
            seen = set()
            deduped = []
            for img in resolved_images:
                if img.image_id not in seen:
                    seen.add(img.image_id)
                    deduped.append(img)
            resolved_images = deduped

    chat_image_refs: list[ChatImageRef] = []
    image_context_parts: list[str] = []
    image_parts: list[dict] = []

    for img in resolved_images[:MAX_VISION_IMAGES]:
        img_ref_id = _generate_citation_id(existing_ids)
        existing_ids.add(img_ref_id)
        img_url = f"/static/doc-images/kb_{workspace_id}/images/{img.image_id}.png"
        chat_image_refs.append(ChatImageRef(
            ref_id=img_ref_id,
            image_id=img.image_id,
            document_id=img.document_id,
            page_no=img.page_no,
            caption=img.caption or "",
            url=img_url,
            width=img.width,
            height=img.height,
        ))
        cap = f'"{img.caption}"' if img.caption else "no caption"
        image_context_parts.append(f"- [IMG-{img_ref_id}] Page {img.page_no}: {cap}")

        img_path = _P(img.file_path)
        if img_path.exists():
            try:
                img_bytes = img_path.read_bytes()
                mime = img.mime_type or "image/png"
                image_parts.append({
                    "inline_data": {"mime_type": mime, "data": img_bytes},
                    "page_no": img.page_no,
                    "caption": img.caption or "",
                    "img_ref_id": img_ref_id,
                })
            except Exception as e:
                logger.warning(f"Failed to read image {img.image_id}: {e}")

    if image_context_parts:
        context += "\n\nDocument Images:\n" + "\n".join(image_context_parts)

    return context, sources, chat_image_refs, image_parts


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
      Gemini 使用原生函数调用；Ollama 使用基于提示词的工具调用。

    生成包含 'event' 和 'data' 键的字典，用于 SSE 格式化。
    """
    from app.services.llm import get_llm_provider
    from app.core.config import settings

    provider = get_llm_provider()
    provider_name = settings.LLM_PROVIDER.lower()
    is_gemini = provider_name == "gemini"
    is_openai = provider_name == "openai"

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
    tools = None
    effective_system_prompt = system_prompt

    if force_search:
        # ── 强制搜索模式：在 LLM 调用前进行预搜索 ──────────────────
        # 立即检索来源，作为上下文注入。不需要工具调用。
        yield {"event": "status", "data": {"step": "retrieving", "detail": f"正在搜索: {message[:80]}..."}}

        context, sources, images, img_parts = await _execute_search_documents(
            workspace_id, message, 8, db, existing_ids,
        )
        all_sources.extend(sources)
        all_images.extend(images)
        all_image_parts.extend(img_parts)

        if sources:
            yield {"event": "sources", "data": {"sources": [s.model_dump() for s in sources]}}
        if images:
            yield {"event": "images", "data": {"image_refs": [i.model_dump() for i in images]}}

        if sources:
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

            user_images_fs: list[LLMImagePart] = []
            if img_parts:
                for img_data in img_parts:
                    tool_result_content += f"\n[IMG-{img_data['img_ref_id']}] (第 {img_data['page_no']} 页):"
                    user_images_fs.append(LLMImagePart(
                        data=img_data["inline_data"]["data"],
                        mime_type=img_data["inline_data"]["mime_type"],
                    ))

            tool_result_content += f"\n\n现在回答问题: {message}"
            messages.append(LLMMessage(
                role="user",
                content=tool_result_content,
                images=user_images_fs,
            ))
        # tools 保持为 None — 模型直接使用提供的上下文进行回答
    elif is_gemini:
        tools = [_get_gemini_tool()]
        # 在系统提示词中强化 Gemini 的工具调用义务
        effective_system_prompt = system_prompt + GEMINI_TOOL_SYSTEM
    elif is_openai:
        tools = _get_openai_tool()
        # OpenAI：使用与 Gemini 相同的工具强化（两者都支持原生函数调用）
        effective_system_prompt = system_prompt + GEMINI_TOOL_SYSTEM
    else:
        # Ollama：将强制工具提示词附加到系统提示词
        effective_system_prompt = system_prompt + "\n\n" + OLLAMA_TOOL_SYSTEM
        # 同时将提醒直接附加到用户消息，以便模型
        # 在生成前看到 —— 强化工具要求
        messages[-1] = LLMMessage(
            role="user",
            content=messages[-1].content + OLLAMA_TOOL_REMINDER,
        )

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
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            system_prompt=effective_system_prompt,
            think=enable_thinking,
            tools=tools if (is_gemini or is_openai) else None,
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

                context, sources, images, img_parts = await _execute_search_documents(
                    workspace_id, query, top_k, db, existing_ids,
                )
                all_sources.extend(sources)
                all_images.extend(images)
                all_image_parts.extend(img_parts)

                if sources:
                    yield {"event": "sources", "data": {
                        "sources": [s.model_dump() for s in sources]
                    }}
                if images:
                    yield {"event": "images", "data": {
                        "image_refs": [i.model_dump() for i in images]
                    }}

                # 将工具结果构建为带有来源的用户消息
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

                # 为视觉模型添加图像内联引用
                user_images: list[LLMImagePart] = []
                if img_parts:
                    for img_data in img_parts:
                        tool_result_content += f"\n[IMG-{img_data['img_ref_id']}] (第 {img_data['page_no']} 页):"
                        user_images.append(LLMImagePart(
                            data=img_data["inline_data"]["data"],
                            mime_type=img_data["inline_data"]["mime_type"],
                        ))

                tool_result_content += f"\n\n现在回答问题: {message}"

                if is_gemini:
                    # Gemini：使用原生 Content 与 thought_signature
                    # （Gemini 3 多轮推理所需）
                    # 以及原生 FunctionResponse 作为工具结果。
                    from google.genai import types as _gtypes

                    raw_content = getattr(provider, "last_response_content", None)
                    if raw_content:
                        # 保留模型的原始响应（包含 thought_signature）
                        messages.append(LLMMessage(
                            role="assistant",
                            content="",
                            _raw_provider_content=raw_content,
                        ))
                    else:
                        messages.append(LLMMessage(
                            role="assistant",
                            content=f"[调用了 search_documents(query=\"{query}\")]",
                        ))

                    # 使用来源上下文构建原生 FunctionResponse
                    func_resp_parts = [_gtypes.Part.from_function_response(
                        name="search_documents",
                        response={"result": tool_result_content},
                    )]
                    func_resp_content = _gtypes.Content(
                        role="user",
                        parts=func_resp_parts,
                    )
                    messages.append(LLMMessage(
                        role="user",
                        content="",
                        _raw_provider_content=func_resp_content,
                    ))

                    # 将图像作为单独的用户消息发送，用于视觉处理
                    if img_parts:
                        img_llm_parts: list[LLMImagePart] = []
                        img_text = "引用的文档图像:\n"
                        for img_data in img_parts:
                            img_text += f"[IMG-{img_data['img_ref_id']}] (第 {img_data['page_no']} 页)\n"
                            img_llm_parts.append(LLMImagePart(
                                data=img_data["inline_data"]["data"],
                                mime_type=img_data["inline_data"]["mime_type"],
                            ))
                        messages.append(LLMMessage(
                            role="user",
                            content=img_text,
                            images=img_llm_parts,
                        ))

                    # 移除工具调用指令，因为搜索已完成；
                    # 保留工具以便思考和工具感知仍然有效。
                    effective_system_prompt = system_prompt
                else:
                    # Ollama：添加基于文本的助手 + 用户消息
                    # 以保持正确的用户/助手交替
                    # （防止两个连续的用户消息，这会混淆
                    # 像 qwen3.5 这样的小模型）。
                    messages.append(LLMMessage(
                        role="assistant",
                        content=f"[调用了 search_documents(query=\"{query}\")]",
                    ))
                    messages.append(LLMMessage(
                        role="user",
                        content=tool_result_content,
                        images=user_images,
                    ))
                    # 从系统提示词中移除工具提示词，以便模型
                    # 使用来源回答而不是再次调用工具。
                    effective_system_prompt = system_prompt

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

    # ── 回退：模型未生成文本且未进行搜索 ──────────
    # 小型 Ollama 模型（例如 qwen3.5:4b）可能会输出关于
    # 需要搜索的思考，但从不生成 <tool_call> 标签或任何文本。
    # 自动搜索并重试一次，以避免 "无法生成响应。"
    if not accumulated_text and not all_sources and not is_gemini:
        logger.warning(
            "Ollama 未生成文本且未进行工具调用 —— 回退到自动搜索"
        )
        yield {"event": "status", "data": {
            "step": "retrieving",
            "detail": f"正在搜索: {message[:80]}..."
        }}

        context, sources, images, img_parts = await _execute_search_documents(
            workspace_id, message, 8, db, existing_ids,
        )
        all_sources.extend(sources)
        all_images.extend(images)
        all_image_parts.extend(img_parts)

        if sources:
            yield {"event": "sources", "data": {
                "sources": [s.model_dump() for s in sources]
            }}
        if images:
            yield {"event": "images", "data": {
                "image_refs": [i.model_dump() for i in images]
            }}

        if sources:
            fallback_parts = [
                "我已为您检索到以下文档来源。\n",
                "=== 文档来源 ===",
                context,
                "=== 来源结束 ===\n",
                "重要提示:\n"
                "- 仔细阅读上述每一个来源。\n"
                "- 如果没有来源包含相关信息，请说："
                "\"文档不包含此信息。\"\n",
            ]
            fallback_content = "\n".join(fallback_parts)
            fallback_content += f"\n\n现在回答问题: {message}"

            # 移除旧的工具系统提示词，将来源作为上下文添加
            fallback_msgs = messages.copy()
            fallback_msgs.append(LLMMessage(role="user", content=fallback_content))

            yield {"event": "status", "data": {
                "step": "generating", "detail": "正在生成答案..."
            }}

            async for chunk in provider.astream(
                fallback_msgs,
                temperature=0.1,
                max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                system_prompt=system_prompt,  # 原始提示词，不带工具指令
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
