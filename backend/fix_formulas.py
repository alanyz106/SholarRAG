#!/usr/bin/env python
"""
Fix formula alignment in already-processed documents.
Applies the _fix_formula_alignment logic to existing markdown_content.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from sqlalchemy import select
from app.core.database import async_session_maker
from app.models.document import Document
from app.services.deep_document_parser import DeepDocumentParser

async def fix_document_formulas(document_id: int, dry_run: bool = False):
    """Apply formula alignment fix to a processed document."""
    async with async_session_maker() as db:
        # Fetch document
        result = await db.execute(select(Document).where(Document.id == document_id))
        doc = result.scalar_one_or_none()

        if not doc:
            print(f"Document {document_id} not found")
            return

        if doc.status.value != "indexed":
            print(f"Document {document_id} is not indexed (status: {doc.status.value})")
            return

        print(f"Fixing document {document_id}: {doc.original_filename}")
        print(f"Parser: {doc.parser_version}, Pages: {doc.page_count}")
        print(f"Markdown length before: {len(doc.markdown_content or '')} chars")

        if not doc.markdown_content:
            print("No markdown content to fix")
            return

        # Use the same fix method from DeepDocumentParser
        parser = DeepDocumentParser(workspace_id=doc.workspace_id)
        fixed_markdown = parser._fix_formula_alignment(doc.markdown_content)

        if fixed_markdown == doc.markdown_content:
            print("No alignment formulas found (no changes)")
        else:
            print(f"Markdown length after: {len(fixed_markdown)} chars")
            if dry_run:
                print("[DRY RUN] Would update document with fixed markdown")
                # Show formulas specifically
                import re
                orig_formulas = re.findall(r'\$\$(.*?)\$\$', doc.markdown_content, re.DOTALL)
                fixed_formulas = re.findall(r'\$\$(.*?)\$\$', fixed_markdown, re.DOTALL)

                print(f"\nFound {len(orig_formulas)} formula blocks")

                # Show formulas that changed
                changes = 0
                for i, (orig, fixed) in enumerate(zip(orig_formulas, fixed_formulas), 1):
                    if orig != fixed:
                        changes += 1
                        print(f"\n--- Formula {i} (CHANGED) ---")
                        print("Original:")
                        print(f"  {orig[:200]}{'...' if len(orig)>200 else ''}")
                        print("\nFixed:")
                        print(f"  {fixed[:200]}{'...' if len(fixed)>200 else ''}")
                        if i >= 3:  # Only show first 3 changed formulas
                            print(f"\n... and {len(orig_formulas) - i} more formulas")
                            break

                if changes == 0:
                    print("No formula changes detected!")
                    print("\nFirst 500 chars of fixed markdown:")
                    print(fixed_markdown[:500])
            else:
                # Update database
                doc.markdown_content = fixed_markdown
                await db.commit()
                print("[OK] Document updated successfully")

async def list_documents():
    """List indexed documents."""
    async with async_session_maker() as db:
        result = await db.execute(
            select(Document.id, Document.workspace_id, Document.original_filename, Document.status)
            .where(Document.status == "indexed")
            .order_by(Document.id.desc())
            .limit(10)
        )
        docs = result.all()
        print("Indexed documents (last 10):")
        print("-" * 80)
        for doc_id, ws_id, filename, status in docs:
            print(f"ID: {doc_id:4d} | WS: {ws_id} | {filename}")
        print()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fix formula alignment in processed documents")
    parser.add_argument("document_id", type=int, nargs="?", help="Document ID to fix")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without updating")
    parser.add_argument("--list", action="store_true", help="List indexed documents")
    args = parser.parse_args()

    if args.list:
        asyncio.run(list_documents())
    elif args.document_id:
        asyncio.run(fix_document_formulas(args.document_id, dry_run=args.dry_run))
    else:
        parser.print_help()
