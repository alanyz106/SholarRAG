#!/usr/bin/env python
"""
Debug script: Inspect raw markdown from Docling to diagnose formula issues.
Usage: python debug_doc_formulas.py <document_id>
"""
import sys
import asyncio
from pathlib import Path

# Add project root to path (backend's parent)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.database import async_session_maker
from app.models.document import Document
from sqlalchemy import select

async def inspect_document(document_id: int):
    """Fetch and print raw markdown for analysis."""
    async with async_session_maker() as db:
        result = await db.execute(
            select(Document).where(Document.id == document_id)
        )
        doc = result.scalar_one_or_none()

        if not doc:
            print(f"Document {document_id} not found")
            return

        print("=" * 80)
        print(f"Document: {doc.original_filename}")
        print(f"Status: {doc.status}")
        print(f"Parser: {doc.parser_version}")
        print(f"Page count: {doc.page_count}")
        print("=" * 80)
        print()

        if not doc.markdown_content:
            print("No markdown content available")
            return

        md = doc.markdown_content

        # Extract formula blocks
        import re

        # Find all blocks that might contain formulas
        patterns = [
            (r'\$\$.*?\$\$', '$$ ... $$'),
            (r'\$.*?\$', '$ ... $'),
            (r'\\\[.*?\\\]', '\\[ ... \\]'),
            (r'\\\(.*?\\\)', '\\( ... \\)'),
        ]

        print("SEARCHING FOR FORMULA PATTERNS:")
        print("-" * 80)
        for pattern, label in patterns:
            matches = re.findall(pattern, md, re.DOTALL)
            if matches:
                print(f"\n[{label}] Found {len(matches)} instance(s):")
                for i, m in enumerate(matches[:5], 1):  # Show first 5
                    preview = m.replace('\n', ' ')[:100]
                    print(f"  {i}. {preview}...")
                if len(matches) > 5:
                    print(f"  ... and {len(matches)-5} more")

        # Print a sample of the markdown (first 2000 chars)
        print("\n" + "=" * 80)
        print("FIRST 2000 CHARACTERS OF MARKDOWN:")
        print("=" * 80)
        print(md[:2000])
        print()

        # Count total length
        print(f"Total markdown length: {len(md)} characters")
        print()

        # Check for common issues
        issues = []
        if '\\' in md and ('$' in md or '\\[' in md):
            # Check for unescaped backslashes
            pass
        if '\t' in md:
            issues.append("Contains tab characters (may affect rendering)")
        if '\r' in md:
            issues.append("Contains CR line endings")

        if issues:
            print("POTENTIAL ISSUES:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("No obvious encoding issues detected")

async def list_documents():
    """List all documents for reference."""
    async with async_session_maker() as db:
        from sqlalchemy import select as sql_select
        result = await db.execute(sql_select(Document).order_by(Document.id.desc()).limit(10))
        docs = result.scalars().all()
        print("Recent documents (showing last 10):")
        print("-" * 80)
        for doc in docs:
            print(f"ID: {doc.id:4d} | WS: {doc.workspace_id} | Status: {doc.status.value:10s} | {doc.original_filename}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_doc_formulas.py <document_id>")
        print("\nNo document_id provided. Listing recent documents...\n")
        asyncio.run(list_documents())
        sys.exit(1)

    doc_id = int(sys.argv[1])
    asyncio.run(inspect_document(doc_id))
