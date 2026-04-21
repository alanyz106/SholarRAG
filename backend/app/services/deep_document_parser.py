"""
Deep Document Parser
====================

Pluggable document parsing with interface:
  - DoclingDocumentParser: PDF/DOCX/PPTX/HTML via Docling (default)
  - Legacy handling for TXT/MD

Usage:
    parser = get_document_parser(workspace_id=1)
    result = parser.parse('/path/to/file.pdf', document_id=1, original_filename='report.pdf')
"""
from __future__ import annotations

import os  # MUST be imported before any huggingface related imports

# =====================================================
# CRITICAL: Set HuggingFace offline mode BEFORE any imports
# This must be at module level to prevent huggingface_hub
# from trying to download models when HybridChunker is imported
# =====================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_CACHE"] = "D:/huggingface_data/hub"

import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.services.models.parsed_document import (
    ExtractedImage,
    ExtractedTable,
    EnrichedChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

# File extensions handled by Docling vs legacy
_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html"}
_LEGACY_EXTENSIONS = {".txt", ".md"}


class DocumentParserBase(ABC):
    """
    Abstract base class for document parsers.

    Subclasses must implement:
    - parse(file_path, document_id, original_filename) -> ParsedDocument
    """

    @abstractmethod
    def parse(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Parse a document and return structured result."""
        ...

    @staticmethod
    @abstractmethod
    def is_docling_supported(file_path: str | Path) -> bool:
        """Check if the file format is supported by Docling (PDF/DOCX/PPTX/HTML)."""
        ...


class DoclingDocumentParser(DocumentParserBase):
    """
    Parses documents using Docling for rich structural extraction.

    - Converts PDF/DOCX/PPTX/HTML via Docling DocumentConverter
    - Chunks using HybridChunker (semantic + structural)
    - Extracts images and optionally captions them via Gemini Vision
    - Falls back to legacy text extraction for TXT/MD
    """

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        self.workspace_id = workspace_id
        self.output_dir = output_dir or (
            settings.BASE_DIR / "data" / "docling" / f"kb_{workspace_id}"
        )
        self._converter = None

    def _get_converter(self):
        """Lazy-init Docling DocumentConverter with image extraction."""
        if self._converter is not None:
            return self._converter

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.accelerator_options import AcceleratorOptions

        pipeline_options = PdfPipelineOptions()
        # Performance optimization: Use CPU with more threads for better throughput on weak GPUs
        pipeline_options.accelerator_options = AcceleratorOptions(
            device=settings.NEXUSRAG_DOCLING_DEVICE,
            num_threads=settings.NEXUSRAG_DOCLING_NUM_THREADS,
        )
        pipeline_options.generate_picture_images = settings.NEXUSRAG_ENABLE_IMAGE_EXTRACTION
        pipeline_options.images_scale = settings.NEXUSRAG_DOCLING_IMAGES_SCALE
        pipeline_options.do_formula_enrichment = settings.NEXUSRAG_ENABLE_FORMULA_ENRICHMENT

        self._converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        return self._converter

    @staticmethod
    def is_docling_supported(file_path: str | Path) -> bool:
        """Check if the file format is supported by Docling."""
        return Path(file_path).suffix.lower() in _DOCLING_EXTENSIONS

    def parse(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """
        Parse a document and return structured result.

        Args:
            file_path: Path to the document file
            document_id: Database document ID
            original_filename: Original filename for citations

        Returns:
            ParsedDocument with markdown, chunks, and images
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        start_time = time.time()

        if suffix in _DOCLING_EXTENSIONS:
            result = self._parse_with_docling(path, document_id, original_filename)
        elif suffix in _LEGACY_EXTENSIONS:
            result = self._parse_legacy(path, document_id, original_filename)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {_DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"Parsed document {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables"
        )
        return result

    def _parse_with_docling(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Parse with Docling for rich structural extraction."""
        converter = self._get_converter()

        # Convert document
        logger.info(f"Docling converting: {file_path}")
        conv_result = converter.convert(str(file_path))
        doc = conv_result.document

        # Extract images and build URL mapping for markdown references
        images, pic_url_list = self._extract_images_with_urls(doc, document_id)

        # Extract tables
        tables = self._extract_tables(doc, document_id)
        if settings.NEXUSRAG_ENABLE_TABLE_CAPTIONING and tables:
            self._caption_tables(tables)

        # Export to markdown (with page break markers if supported)
        markdown = self._export_markdown(doc)

        # Fix formula alignment issues (Docling outputs & without aligned environment)
        markdown = self._fix_formula_alignment(markdown)

        # Post-process: replace image placeholders with real markdown images
        markdown = self._inject_image_references(markdown, pic_url_list)

        # Post-process: inject table captions into markdown
        markdown = self._inject_table_captions(markdown, tables)

        # Get page count
        page_count = 0
        if hasattr(doc, "pages") and doc.pages:
            page_count = len(doc.pages)

        # Chunk with HybridChunker — pass images + tables for enrichment
        chunks = self._chunk_document(doc, document_id, original_filename, images, tables)

        # Count tables
        tables_count = len(tables)

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=markdown,
            page_count=page_count,
            chunks=chunks,
            images=images,
            tables=tables,
            tables_count=tables_count,
        )

    def _chunk_document(
        self,
        doc,
        document_id: int,
        original_filename: str,
        images: list[ExtractedImage] | None = None,
        tables: list[ExtractedTable] | None = None,
    ) -> list[EnrichedChunk]:
        """Chunk document using Docling's HybridChunker.

        When *images* / *tables* are provided, each chunk is enriched with
        references to images/tables that appear on the same page.  The
        descriptions are appended to the chunk text so they become part of
        the embedding, making image/table content semantically searchable.
        """
        from docling_core.transforms.chunker import HybridChunker
        
        chunker = HybridChunker(
            max_tokens=settings.NEXUSRAG_CHUNK_MAX_TOKENS,
            merge_peers=True,
        )

        # Build page→images lookup for O(1) matching
        page_images: dict[int, list[ExtractedImage]] = {}
        if images:
            for img in images:
                page_images.setdefault(img.page_no, []).append(img)

        # Build page→tables lookup for O(1) matching
        page_tables: dict[int, list[ExtractedTable]] = {}
        if tables:
            for tbl in tables:
                page_tables.setdefault(tbl.page_no, []).append(tbl)

        chunks = []
        # Track which images/tables have already been assigned to a chunk
        # on the same page so we avoid duplicating descriptions across many
        # chunks.  Each image/table is assigned to the FIRST chunk on its page.
        assigned_images: set[str] = set()
        assigned_tables: set[str] = set()

        for i, chunk in enumerate(chunker.chunk(doc)):
            # Extract page number from chunk metadata
            page_no = 0
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "page"):
                    page_no = chunk.meta.page or 0
                elif hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if hasattr(item, "prov") and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, "page_no"):
                                    page_no = prov.page_no or 0
                                    break
                            if page_no > 0:
                                break

            # Extract heading path from chunk metadata
            heading_path = []
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    heading_path = list(chunk.meta.headings)

            # Detect content types
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            has_table = False
            has_code = False
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        label = getattr(item, "label", "") or ""
                        if "table" in label.lower():
                            has_table = True
                        if "code" in label.lower():
                            has_code = True

            contextualized = ""
            if heading_path:
                contextualized = " > ".join(heading_path) + ": " + chunk_text[:100]

            # ── Image-aware enrichment ──────────────────────────────
            chunk_image_refs: list[str] = []  # image_ids
            if page_no > 0 and page_no in page_images:
                for img in page_images[page_no]:
                    if img.image_id not in assigned_images:
                        chunk_image_refs.append(img.image_id)
                        assigned_images.add(img.image_id)

            # Append image descriptions to chunk text so they become
            # part of the embedding vector (semantically searchable).
            enriched_text = chunk_text
            if chunk_image_refs and images:
                img_by_id = {im.image_id: im for im in images}
                desc_parts = []
                for img_id in chunk_image_refs:
                    img = img_by_id.get(img_id)
                    if img and img.caption:
                        desc_parts.append(
                            f"[Image on page {img.page_no}]: {img.caption}"
                        )
                if desc_parts:
                    enriched_text = (
                        chunk_text + "\n\n" + "\n".join(desc_parts)
                    )

            # ── Table-aware enrichment ────────────────────────────────
            chunk_table_refs: list[str] = []
            if page_no > 0 and page_no in page_tables:
                for tbl in page_tables[page_no]:
                    if tbl.table_id not in assigned_tables:
                        chunk_table_refs.append(tbl.table_id)
                        assigned_tables.add(tbl.table_id)

            if chunk_table_refs and tables:
                tbl_by_id = {t.table_id: t for t in tables}
                tbl_parts = []
                for tbl_id in chunk_table_refs:
                    tbl = tbl_by_id.get(tbl_id)
                    if tbl and tbl.caption:
                        tbl_parts.append(
                            f"[Table on page {tbl.page_no} ({tbl.num_rows}x{tbl.num_cols})]: {tbl.caption}"
                        )
                if tbl_parts:
                    enriched_text = enriched_text + "\n\n" + "\n".join(tbl_parts)

            chunks.append(EnrichedChunk(
                content=enriched_text,
                chunk_index=i,
                source_file=original_filename,
                document_id=document_id,
                page_no=page_no,
                heading_path=heading_path,
                image_refs=chunk_image_refs,
                table_refs=chunk_table_refs,
                has_table=has_table,
                has_code=has_code,
                contextualized=contextualized,
            ))

        if images:
            assigned_count = len(assigned_images)
            logger.info(
                f"Image-aware chunking: {assigned_count}/{len(images)} images "
                f"assigned to {len(chunks)} chunks"
            )

        if tables:
            assigned_tbl_count = len(assigned_tables)
            logger.info(
                f"Table-aware chunking: {assigned_tbl_count}/{len(tables)} tables "
                f"assigned to {len(chunks)} chunks"
            )

        return chunks

    def _export_markdown(self, doc) -> str:
        """Export document to markdown with page break markers if supported."""
        try:
            return doc.export_to_markdown(
                page_break_placeholder="\n\n---\n\n",
            )
        except TypeError:
            # Fallback for Docling versions without page_break_placeholder
            return doc.export_to_markdown()

    def _fix_formula_alignment(self, markdown: str) -> str:
        """Fix Docling formula alignment: wrap &-aligned formulas in aligned environment.

        Docling sometimes outputs multi-line formulas with & for alignment:
            $$ E = & \sum ... \\ p = & \text{...} $$

        KaTeX requires these to be in an aligned environment:
            $$\begin{aligned} E &= \sum ... \\ p &= \text{...} \end{aligned}$$

        Also handles HTML entity encoding (e.g., &amp; → &).

        Args:
            markdown: Raw markdown from Docling

        Returns:
            Fixed markdown with aligned environments
        """
        import re
        import html

        def wrap_formula(match: re.Match) -> str:
            formula = match.group(1).strip()
            # Unescape HTML entities first (e.g., &amp; → &)
            formula = html.unescape(formula)

            # Only fix if formula contains & alignment AND line breaks
            if '&' in formula and ('\\\\' in formula):
                # Avoid double-wrapping
                if not (formula.startswith('\\begin') and formula.endswith('\\end')):
                    return f'$$\n\\begin{{aligned}}\n{formula}\n\\end{{aligned}}\n$$'
            return f'$${formula}$$'

        # Match all $$...$$ blocks (DOTALL for multiline)
        return re.sub(r'\$\$(.*?)\$\$', wrap_formula, markdown, flags=re.DOTALL)

    def _extract_images_with_urls(
        self,
        doc,
        document_id: int,
    ) -> tuple[list[ExtractedImage], list[tuple[str, str]]]:
        """
        Extract images and build URL mapping for markdown placeholders.

        Returns:
            (images, pic_url_list) where pic_url_list has one (caption, url)
            tuple per doc.pictures element, in order.
        """
        if not settings.NEXUSRAG_ENABLE_IMAGE_EXTRACTION:
            return [], []

        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images: list[ExtractedImage] = []
        pic_to_image_idx: list[int] = []  # maps picture index → images list index
        picture_count = 0

        if not hasattr(doc, "pictures") or not doc.pictures:
            return [], []

        for pic in doc.pictures:
            if picture_count >= settings.NEXUSRAG_MAX_IMAGES_PER_DOC:
                pic_to_image_idx.append(-1)
                continue

            image_id = str(uuid.uuid4())

            # Get page number
            page_no = 0
            if hasattr(pic, "prov") and pic.prov:
                for prov in pic.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            try:
                image_path = images_dir / f"{image_id}.png"

                if hasattr(pic, "image") and pic.image:
                    pil_image = pic.image.pil_image
                    if pil_image:
                        pil_image.save(str(image_path), format="PNG")
                        width, height = pil_image.size
                    else:
                        pic_to_image_idx.append(-1)
                        continue
                else:
                    pic_to_image_idx.append(-1)
                    continue

                # Get caption
                caption = ""
                if hasattr(pic, "caption_text"):
                    caption = pic.caption_text(doc) if callable(pic.caption_text) else str(pic.caption_text or "")
                elif hasattr(pic, "text"):
                    caption = str(pic.text or "")

                images.append(ExtractedImage(
                    image_id=image_id,
                    document_id=document_id,
                    page_no=page_no,
                    file_path=str(image_path),
                    caption=caption,
                    width=width,
                    height=height,
                ))
                pic_to_image_idx.append(len(images) - 1)
                picture_count += 1

            except Exception as e:
                logger.warning(f"Failed to extract image from document {document_id}: {e}")
                pic_to_image_idx.append(-1)
                continue

        logger.info(f"Extracted {len(images)} images from document {document_id}")

        # Caption images with Gemini Vision (updates img.caption in-place)
        if settings.NEXUSRAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)

        # Build pic_url_list AFTER captioning so captions are up-to-date
        pic_url_list: list[tuple[str, str]] = []
        for idx in pic_to_image_idx:
            if idx >= 0:
                img = images[idx]
                url = f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                pic_url_list.append((img.caption, url))
            else:
                pic_url_list.append(("", ""))

        return images, pic_url_list

    def _inject_image_references(
        self, markdown: str, pic_url_list: list[tuple[str, str]]
    ) -> str:
        """Replace <!-- image --> placeholders with ![caption](url) markdown."""
        placeholder_count = len(re.findall(r"<!--\s*image\s*-->", markdown))

        if not pic_url_list:
            if placeholder_count > 0:
                logger.warning(
                    f"Markdown has {placeholder_count} image placeholders but "
                    f"pic_url_list is empty — images will NOT be injected"
                )
            return markdown

        logger.info(
            f"Injecting {len(pic_url_list)} image URLs into "
            f"{placeholder_count} placeholders"
        )

        injected = 0
        pic_iter = iter(pic_url_list)

        def replacer(match):
            nonlocal injected
            try:
                caption, url = next(pic_iter)
                if url:
                    safe_caption = caption.replace("[", "").replace("]", "")
                    # Markdown ![alt](url) must be single-line — collapse newlines
                    safe_caption = " ".join(safe_caption.split())
                    injected += 1
                    return f"\n![{safe_caption}]({url})\n"
                return ""
            except StopIteration:
                return ""

        result = re.sub(r'<!--\s*image\s*-->', replacer, markdown)
        logger.info(f"Injected {injected}/{placeholder_count} image references")
        return result

    # ------------------------------------------------------------------
    # Table extraction & captioning
    # ------------------------------------------------------------------

    def _extract_tables(self, doc, document_id: int) -> list[ExtractedTable]:
        """Extract tables from Docling document."""
        if not hasattr(doc, "tables") or not doc.tables:
            return []

        tables: list[ExtractedTable] = []
        for table in doc.tables:
            table_id = str(uuid.uuid4())

            # Get page number
            page_no = 0
            if hasattr(table, "prov") and table.prov:
                for prov in table.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            # Export to markdown
            try:
                content_md = table.export_to_markdown(doc)
            except Exception:
                content_md = ""

            if not content_md.strip():
                continue

            # Get dimensions
            num_rows = 0
            num_cols = 0
            if hasattr(table, "data") and table.data:
                num_rows = getattr(table.data, "num_rows", 0) or 0
                num_cols = getattr(table.data, "num_cols", 0) or 0

            tables.append(ExtractedTable(
                table_id=table_id,
                document_id=document_id,
                page_no=page_no,
                content_markdown=content_md,
                num_rows=num_rows,
                num_cols=num_cols,
            ))

        logger.info(f"Extracted {len(tables)} tables from document {document_id}")
        return tables

    _TABLE_CAPTION_PROMPT = (
        "你是文档分析助手。给定一个 markdown 表格，"
        "撰写简洁描述，涵盖：\n"
        "- 表格的目的/主题\n"
        "- 关键列名称及其含义\n"
        "- 显著的数值、趋势或异常值\n\n"
        "规则：\n"
        "- 写2-4句话，最多500个字符。\n"
        "- 实事求是 —— 仅描述表格中的内容。\n"
        "- 使用与表格内容相同的语言。"
        "如果表格是越南语，用越南语写。"
        "如果是英语，用英语写。\n\n"
        "表格：\n"
    )

    def _caption_tables(self, tables: list[ExtractedTable]) -> None:
        """Caption tables using LLM (text-only, no vision needed)."""
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMMessage

        provider = get_llm_provider()

        for tbl in tables:
            if tbl.caption:
                continue
            try:
                table_md = tbl.content_markdown
                # Truncate large tables
                if len(table_md) > settings.NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS:
                    table_md = table_md[:settings.NEXUSRAG_MAX_TABLE_MARKDOWN_CHARS] + "\n... (truncated)"

                message = LLMMessage(
                    role="user",
                    content=self._TABLE_CAPTION_PROMPT + table_md,
                    images=[],
                )
                result = provider.complete([message])
                if result:
                    tbl.caption = " ".join(result.strip().split())[:500]
            except Exception as e:
                logger.debug(f"Failed to caption table {tbl.table_id}: {e}")

    @staticmethod
    def _inject_table_captions(
        markdown: str, tables: list[ExtractedTable]
    ) -> str:
        """Inject table captions as blockquotes after matching table blocks in markdown."""
        if not tables:
            return markdown

        # Only process tables that have captions
        captioned = [t for t in tables if t.caption]
        if not captioned:
            return markdown

        lines = markdown.split("\n")
        result_lines: list[str] = []
        matched_count = 0

        # Build a lookup: first data row content → table caption
        # (skip the header row and separator row, use the first data row)
        table_lookup: dict[str, ExtractedTable] = {}
        for tbl in captioned:
            tbl_lines = tbl.content_markdown.strip().split("\n")
            # Find first data row (skip header + separator)
            for tl in tbl_lines:
                tl_stripped = tl.strip()
                if tl_stripped.startswith("|") and "---" not in tl_stripped:
                    # Use cell content as key (strip pipes and whitespace)
                    cells = [c.strip() for c in tl_stripped.split("|") if c.strip()]
                    if cells:
                        key = "|".join(cells[:3])  # Use first 3 cells as key
                        table_lookup[key] = tbl
                        break

        i = 0
        while i < len(lines):
            line = lines[i]
            result_lines.append(line)

            # Detect start of a table block (line starts with |)
            if line.strip().startswith("|"):
                # Collect all consecutive table lines
                table_block_start = i
                while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                    i += 1
                    result_lines.append(lines[i])

                # Try to match this table block to a captioned table
                block_lines = lines[table_block_start:i + 1]
                for bl in block_lines:
                    bl_stripped = bl.strip()
                    if bl_stripped.startswith("|") and "---" not in bl_stripped:
                        cells = [c.strip() for c in bl_stripped.split("|") if c.strip()]
                        if cells:
                            key = "|".join(cells[:3])
                            if key in table_lookup:
                                tbl = table_lookup.pop(key)
                                result_lines.append(f"\n> **Table:** {tbl.caption}")
                                matched_count += 1
                                break

            i += 1

        logger.info(
            f"Injected {matched_count}/{len(captioned)} table captions into markdown"
        )
        return "\n".join(result_lines)

    def _caption_images(self, images: list[ExtractedImage]) -> None:
        """Caption images using the configured LLM provider (sync, best-effort).

        Generates detailed descriptions so that image content is
        semantically searchable when embedded alongside text chunks.
        """
        from app.services.llm import get_llm_provider
        from app.services.llm.types import LLMImagePart, LLMMessage

        provider = get_llm_provider()
        if not provider.supports_vision():
            logger.warning("LLM provider does not support vision — skipping image captioning")
            return

        _CAPTION_PROMPT = (
            "仅描述你在本图中能直接看到的内容。"
            "不要推断、假设或添加任何不可见的信息。\n\n"
            "包括：\n"
            "- 视觉类型（图表、表格、示意图、照片、截图等）\n"
            "- 图像中可见的所有具体数字、百分比和标签\n"
            "- 轴标签、图例文本和类别名称（按原样显示）\n"
            "- 视觉上明显的趋势或比较\n\n"
            "规则：\n"
            "- 写2-4句简洁的话，最多400个字符。\n"
            "- 不要以'This image shows'或'Here is'开头。\n"
            "- 不要添加超出可见范围的任何数据、上下文或解释。\n"
            "- 如果图像中的文本不清楚，请说明。"
        )

        for img in images:
            if img.caption:  # already has caption from document
                continue
            try:
                image_path = Path(img.file_path)
                if not image_path.exists():
                    continue

                with open(image_path, "rb") as f:
                    image_bytes = f.read()

                message = LLMMessage(
                    role="user",
                    content=_CAPTION_PROMPT,
                    images=[LLMImagePart(data=image_bytes, mime_type=img.mime_type)],
                )
                result = provider.complete([message])
                if result:
                    # Collapse to single line — prevents breaking ![alt](url) markdown
                    img.caption = " ".join(result.strip().split())[:500]

            except Exception as e:
                logger.debug(f"Failed to caption image {img.image_id}: {e}")

    def _parse_legacy(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Fallback: parse TXT/MD with legacy loader + RecursiveCharacterTextSplitter."""
        from app.services.document_loader import load_document
        from app.services.chunker import DocumentChunker

        loaded = load_document(str(file_path))
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        text_chunks = chunker.split_text(
            text=loaded.content,
            source=original_filename,
            extra_metadata={"document_id": document_id, "file_type": loaded.file_type},
        )

        # Wrap legacy chunks as EnrichedChunks
        chunks = [
            EnrichedChunk(
                content=tc.content,
                chunk_index=tc.chunk_index,
                source_file=original_filename,
                document_id=document_id,
                page_no=0,
            )
            for tc in text_chunks
        ]

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=loaded.content,
            page_count=loaded.page_count,
            chunks=chunks,
            images=[],
            tables_count=0,
        )


def get_document_parser(workspace_id: int) -> DocumentParserBase:
    """Factory: create a DocumentParser based on DOCUMENT_PARSER_PROVIDER config.

    Default: DoclingDocumentParser (supports PDF, DOCX, PPTX, HTML).
    Fallback: LegacyDocumentParser for TXT/MD (auto-detected by file extension).
    """
    provider = getattr(settings, "DOCUMENT_PARSER_PROVIDER", "docling").lower()

    if provider == "docling":
        return DoclingDocumentParser(workspace_id=workspace_id)

    raise ValueError(
        f"DOCUMENT_PARSER_PROVIDER '{provider}' is not supported. "
        f"Available: docling"
    )


# Backward-compatible alias (deprecated, use get_document_parser)
DeepDocumentParser = DoclingDocumentParser
