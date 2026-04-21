"""
Chat Service
============
Unified chat logic: prompt building, LLM calls, response processing, message persistence.
"""
import re
import uuid
import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.chat_message import ChatMessage as ChatMessageModel


logger = logging.getLogger(__name__)


@dataclass
class ChatResult:
    """Structured result from a chat operation."""
    answer: str
    sources: list
    related_entities: list = field(default_factory=list)
    kg_summary: Optional[str] = None
    image_refs: list = field(default_factory=list)
    thinking: Optional[str] = None


def build_user_message(
    question: str,
    context: str,
    image_context_parts: list[str],
    history: list,
) -> str:
    """
    Build user message: CONTEXT → RULES → QUESTION structure.

    Structure:
    1. Document sources (the model reads this first)
    2. Image references (if any)
    3. Contextual rules about reading sources, table data extraction
    4. Conversation context recap (if history exists)
    5. The actual question (last = highest attention position)
    """
    user_parts: list[str] = []

    # 1. Document sources
    user_parts.append("I have retrieved the following document sources for you.\n")
    user_parts.append("=== DOCUMENT SOURCES ===")
    user_parts.append(context)
    user_parts.append("=== END SOURCES ===\n")

    # 2. Image references
    if image_context_parts:
        user_parts.append("Document Images:")
        user_parts.extend(image_context_parts)
        user_parts.append("")

    # 3. Contextual rules
    user_parts.append(
        "IMPORTANT:\n"
        "- Read EVERY source above carefully. Answers often require "
        "combining data from MULTIPLE sources.\n"
        "- TABLE DATA: Sources may contain table data as 'Key, Year = Value' pairs. "
        "Example: 'ROE, 2023 = 12,8%' means ROE was 12.8% in 2023. "
        "Extract and report these values.\n"
        "- If no source contains relevant information, say: "
        "\"Tài liệu không chứa thông tin này.\"\n"
    )

    # 4. Conversation context recap
    if history:
        last_exchange = history[-2:]  # last Q+A pair
        recap_parts = []
        for msg in last_exchange:
            prefix = "User" if msg.role == "user" else "Assistant"
            recap_parts.append(f"{prefix}: {msg.content[:300]}")
        user_parts.append(
            "CONVERSATION CONTEXT (previous exchange):\n"
            + "\n".join(recap_parts) + "\n"
        )

    # 5. The actual question (last = highest attention position)
    user_parts.append(f"My question: {question}")

    return "\n".join(user_parts)


async def execute_chat(
    workspace_id: int,
    question: str,
    context: str,
    sources: list,
    image_refs: list,
    image_parts: list,
    history: list,
    enable_thinking: bool,
    system_prompt: str,
    db: AsyncSession,
) -> ChatResult:
    """
    Execute chat with documents as context.

    Returns ChatResult with answer, sources, image_refs, thinking.
    """
    from app.services.llm import get_llm_provider
    from app.services.llm.types import LLMImagePart, LLMMessage, LLMResult

    # Build image context parts for user message
    image_context_parts = []
    for img in image_refs:
        cap = f'"{img.caption}"' if img.caption else "no caption"
        image_context_parts.append(
            f"- [IMG-{img.ref_id}] Page {img.page_no}: {cap}"
        )

    # Build user message
    user_content = build_user_message(
        question=question,
        context=context,
        image_context_parts=image_context_parts,
        history=history,
    )

    # Build messages for LLM (keep last 10 for context)
    messages: list[LLMMessage] = []
    for msg in history[-10:]:
        role = "user" if msg.role == "user" else "assistant"
        messages.append(LLMMessage(role=role, content=msg.content))

    # Attach images to user message (for multimodal models)
    user_images: list[LLMImagePart] = []
    if image_parts:
        for img_data in image_parts:
            user_content += f"\n[IMG-{img_data['img_ref_id']}] (page {img_data['page_no']}):"
            user_images.append(LLMImagePart(
                data=img_data["inline_data"]["data"],
                mime_type=img_data["inline_data"]["mime_type"],
            ))

    messages.append(LLMMessage(role="user", content=user_content, images=user_images))

    # Call LLM
    provider = get_llm_provider()

    thinking_text: str | None = None
    try:
        result = await provider.acomplete(
            messages,
            system_prompt=system_prompt,
            temperature=0.1,
            max_tokens=4096,
            think=enable_thinking,
        )

        if isinstance(result, LLMResult):
            answer = result.content
            thinking_text = result.thinking or None
        else:
            answer = result
            thinking_text = None

        if not answer:
            answer = "Unable to generate a response."

        # Strip Gemini token artifacts (e.g. <unused778>:)
        answer = re.sub(r'<unused\d+>:?\s*', '', answer).strip()

    except Exception as e:
        logger.error(f"LLM chat error: {e}")
        answer = f"Sorry, I encountered an error generating the response: {str(e)}"
        thinking_text = None

    return ChatResult(
        answer=answer,
        sources=sources,
        related_entities=[],  # Will be filled by caller if needed
        kg_summary=None,
        image_refs=image_refs,
        thinking=thinking_text,
    )


async def persist_messages(
    workspace_id: int,
    question: str,
    answer: str,
    sources: list,
    related_entities: list,
    image_refs: list,
    thinking: Optional[str],
    db: AsyncSession,
) -> bool:
    """
    Persist user and assistant messages to the database.

    Returns True on success, False on failure.
    """
    try:
        user_row = ChatMessageModel(
            workspace_id=workspace_id,
            message_id=str(uuid.uuid4()),
            role="user",
            content=question,
        )
        db.add(user_row)

        assistant_row = ChatMessageModel(
            workspace_id=workspace_id,
            message_id=str(uuid.uuid4()),
            role="assistant",
            content=answer,
            sources=[s.model_dump() for s in sources] if sources else None,
            related_entities=related_entities[:30] if related_entities else None,
            image_refs=[img.model_dump() for img in image_refs] if image_refs else None,
            thinking=thinking,
        )
        db.add(assistant_row)
        await db.commit()
        return True
    except Exception as e:
        logger.warning(f"Failed to persist chat messages: {e}")
        await db.rollback()
        return False
