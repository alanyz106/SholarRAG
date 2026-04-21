"""
Retrieval Service
================
Unified retrieval logic for NexusRAG: vector search, KG search, image resolution.
"""
import logging
import random
import string
from dataclasses import dataclass
from pathlib import Path as _P
from typing import Optional

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import DocumentImage
from app.schemas.rag import ChatImageRef, ChatSourceChunk

logger = logging.getLogger(__name__)

# Constants
MAX_VISION_IMAGES = 3
_CITATION_ID_CHARS = string.ascii_lowercase + string.digits


def _generate_citation_id(existing: set[str]) -> str:
    """Generate unique 4-char alphanumeric citation ID."""
    while True:
        cid = "".join(random.choices(_CITATION_ID_CHARS, k=4))
        if any(c.isalpha() for c in cid) and cid not in existing:
            return cid


@dataclass
class RetrievalResult:
    """Result from retrieval service."""
    context: str
    sources: list[ChatSourceChunk]
    image_refs: list[ChatImageRef]
    image_parts: list[dict]  # For vision models


async def retrieve_documents(
    workspace_id: int,
    question: str,
    top_k: int,
    db: AsyncSession,
    existing_ids: Optional[set[str]] = None,
    document_ids: Optional[list[int]] = None,
) -> RetrievalResult:
    """
    Execute hybrid retrieval for a question.
    Returns context, sources, image refs, and image parts for vision.
    """
    from app.services.rag_service import get_rag_service
    from app.services.nexus_rag_service import NexusRAGService
    from types import SimpleNamespace

    if existing_ids is None:
        existing_ids = set()

    rag_service = get_rag_service(db, workspace_id)

    chunks = []
    citations = []

    if isinstance(rag_service, NexusRAGService):
        result = await rag_service.query_deep(
            question=question,
            top_k=min(top_k, 10),
            document_ids=document_ids,
            mode="hybrid",
            include_images=False,
        )
        chunks = result.chunks
        citations = result.citations
    else:
        legacy = rag_service.query(question=question, top_k=min(top_k, 10), document_ids=document_ids)
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

    # Build sources with citation IDs
    sources: list[ChatSourceChunk] = []
    context_parts: list[str] = []

    for i, chunk in enumerate(chunks):
        citation = citations[i] if i < len(citations) else None
        cid = _generate_citation_id(existing_ids)
        existing_ids.add(cid)

        # Build ChatSourceChunk
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

        # Build metadata line
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

    # Resolve images
    image_refs, image_parts = await resolve_images_from_chunks(
        chunks, db, workspace_id, existing_ids
    )

    return RetrievalResult(
        context=context,
        sources=sources,
        image_refs=image_refs,
        image_parts=image_parts,
    )


async def resolve_images_from_chunks(
    chunks: list,
    db: AsyncSession,
    workspace_id: int,
    existing_ids: set[str],
) -> tuple[list[ChatImageRef], list[dict]]:
    """Resolve images from chunk metadata."""
    # Collect image IDs from chunks
    seen_image_ids: set[str] = set()
    chunk_image_ids: list[str] = []
    for c in chunks:
        for iid in getattr(c, "image_refs", []) or []:
            if iid and iid not in seen_image_ids:
                seen_image_ids.add(iid)
                chunk_image_ids.append(iid)

    # Lookup by image_id
    resolved_images: list[DocumentImage] = []
    if chunk_image_ids:
        img_result = await db.execute(
            select(DocumentImage).where(DocumentImage.image_id.in_(chunk_image_ids))
        )
        resolved_images = list(img_result.scalars().all())

    # Fallback: page-based lookup
    if not resolved_images:
        source_pages = {
            (getattr(c, "document_id", 0), getattr(c, "page_no", 0))
            for c in chunks if getattr(c, "page_no", 0) > 0
        }
        if source_pages:
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
            # Deduplicate
            seen = set()
            deduped = []
            for img in resolved_images:
                if img.image_id not in seen:
                    seen.add(img.image_id)
                    deduped.append(img)
            resolved_images = deduped

    # Build response
    chat_image_refs: list[ChatImageRef] = []
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

        # Read image for vision
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

    return chat_image_refs, image_parts
