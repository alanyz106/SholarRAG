"""
Knowledge Graph Service
========================

Per-workspace Knowledge Graph using LightRAG with configurable LLM + embeddings.
File-based storage (NetworkX graph + NanoVectorDB) — no extra Docker services.

Usage:
    kg = KnowledgeGraphService(workspace_id=1)
    await kg.ingest("markdown text from document...")
    result = await kg.query("What are the key themes?", mode="hybrid")
    await kg.cleanup()
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, Set

import numpy as np

from app.core.config import settings
from app.services.llm import get_embedding_provider, get_llm_provider
from app.services.llm.types import LLMMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider-based adapters for LightRAG
# ---------------------------------------------------------------------------

async def _kg_llm_complete(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list] = None,
    keyword_extraction: bool = False,
    **kwargs,
) -> str:
    """LightRAG-compatible LLM function using the configured provider.

    Includes retry logic with exponential backoff for transient errors
    (e.g., 529 overloaded) to improve KG extraction reliability.
    """
    import asyncio

    provider = get_llm_provider()

    messages: list[LLMMessage] = []

    if system_prompt:
        messages.append(LLMMessage(role="system", content=system_prompt))

    if history_messages:
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append(LLMMessage(role=role, content=content))

    messages.append(LLMMessage(role="user", content=prompt))

    # Retry with exponential backoff for transient errors
    max_retries = 3
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return await provider.acomplete(
                messages, temperature=0.0, max_tokens=4096,
            )
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            # Retry on overloaded errors (529) or common transient issues
            if "529" in error_str or "overloaded" in error_str or "timeout" in error_str or "connection" in error_str:
                wait_time = (attempt + 1) * 2  # 2, 4, 6 seconds
                logger.warning(
                    f"KG LLM call failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                await asyncio.sleep(wait_time)
            else:
                # Non-transient error, don't retry
                raise

    # All retries exhausted
    raise last_error from None


async def _kg_embed(texts: list[str]) -> np.ndarray:
    """LightRAG-compatible embedding function using the configured provider."""
    provider = get_embedding_provider()
    return await provider.embed(texts)


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class KnowledgeGraphService:
    """
    Per-workspace Knowledge Graph service backed by LightRAG.

    Storage: file-based (NetworkX for graph, NanoVectorDB for vectors).
    Each knowledge base gets its own working directory.
    """

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.working_dir = str(
            settings.BASE_DIR / "data" / "lightrag" / f"kb_{workspace_id}"
        )
        self._rag = None
        self._initialized = False

    async def _get_rag(self):
        """Lazy-initialize LightRAG instance."""
        if self._rag is not None and self._initialized:
            return self._rag

        from lightrag import LightRAG
        from lightrag.utils import wrap_embedding_func_with_attrs
        from lightrag.kg.shared_storage import initialize_pipeline_status

        os.makedirs(self.working_dir, exist_ok=True)

        # Dynamic embedding dimension from the configured provider
        emb_provider = get_embedding_provider()
        embedding_dim = emb_provider.get_dimension()

        # Detect dimension mismatch when switching providers
        dim_marker = Path(self.working_dir) / ".embedding_dim"
        if dim_marker.exists():
            prev_dim = int(dim_marker.read_text().strip())
            if prev_dim != embedding_dim:
                logger.warning(
                    f"Embedding dimension changed ({prev_dim} → {embedding_dim}) "
                    f"for workspace {self.workspace_id}. Clearing KG data for rebuild."
                )
                shutil.rmtree(self.working_dir)
                os.makedirs(self.working_dir, exist_ok=True)
        dim_marker.write_text(str(embedding_dim))

        @wrap_embedding_func_with_attrs(embedding_dim=embedding_dim, max_token_size=8192)
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await _kg_embed(texts)

        self._rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=_kg_llm_complete,
            embedding_func=embedding_func,
            chunk_token_size=settings.NEXUSRAG_KG_CHUNK_TOKEN_SIZE,
            enable_llm_cache=True,
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage",
            doc_status_storage="JsonDocStatusStorage",
            addon_params={
                "language": settings.NEXUSRAG_KG_LANGUAGE,
                "entity_types": settings.NEXUSRAG_KG_ENTITY_TYPES,
            },
        )

        await self._rag.initialize_storages()
        await initialize_pipeline_status()
        self._initialized = True

        logger.info(
            f"LightRAG initialized for workspace {self.workspace_id} "
            f"(embedding_dim={embedding_dim})"
        )
        return self._rag

    async def ingest(self, markdown_content: str) -> None:
        """
        Ingest markdown content into the knowledge graph.
        LightRAG extracts entities and relationships automatically.
        """
        rag = await self._get_rag()

        if not markdown_content.strip():
            logger.warning(f"Empty content for workspace {self.workspace_id}, skipping KG ingest")
            return

        try:
            await rag.ainsert(markdown_content)
            logger.info(
                f"KG ingested {len(markdown_content)} chars for workspace {self.workspace_id}"
            )

            # Check if entities were actually extracted
            try:
                all_nodes = await rag.chunk_entity_relation_graph.get_all_nodes()
                if not all_nodes:
                    from app.core.config import settings
                    model = (
                        settings.OLLAMA_MODEL
                        if settings.LLM_PROVIDER.lower() == "ollama"
                        else settings.LLM_MODEL_FAST
                    )
                    logger.warning(
                        f"KG extraction produced 0 entities for workspace {self.workspace_id}. "
                        f"Model '{model}' may not support LightRAG's entity extraction format. "
                        f"Consider using a larger model (e.g. qwen3:14b, gemma3:12b) for KG."
                    )
            except Exception:
                pass

        except Exception as e:
            logger.error(f"KG ingest failed for workspace {self.workspace_id}: {e}")
            raise

    async def query(
        self,
        question: str,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> str:
        """
        Query the knowledge graph.

        Args:
            question: Natural language question
            mode: Query mode — "naive", "local", "global", "hybrid"
            top_k: Number of results

        Returns:
            LightRAG response text with KG-augmented answer
        """
        from lightrag import QueryParam

        rag = await self._get_rag()

        try:
            result = await asyncio.wait_for(
                rag.aquery(
                    question,
                    param=QueryParam(mode=mode, top_k=top_k),
                ),
                timeout=settings.NEXUSRAG_KG_QUERY_TIMEOUT,
            )
            return result or ""
        except asyncio.TimeoutError:
            logger.warning(
                f"KG query timed out after {settings.NEXUSRAG_KG_QUERY_TIMEOUT}s "
                f"for workspace {self.workspace_id}"
            )
            return ""
        except Exception as e:
            logger.error(f"KG query failed for workspace {self.workspace_id}: {e}")
            return ""

    async def cleanup(self) -> None:
        """Finalize storages on shutdown."""
        if self._rag:
            try:
                await self._rag.finalize_storages()
                logger.info(f"KG storages finalized for workspace {self.workspace_id}")
            except Exception as e:
                logger.warning(f"KG cleanup failed for workspace {self.workspace_id}: {e}")
            self._rag = None
            self._initialized = False

    def delete_project_data(self) -> None:
        """Delete all KG data for this knowledge base."""
        path = Path(self.working_dir)
        if path.exists():
            shutil.rmtree(path)
            logger.info(f"Deleted KG data for workspace {self.workspace_id}")
        self._rag = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Knowledge Graph exploration (Phase 9)
    # ------------------------------------------------------------------

    async def get_entities(
        self,
        search: str | None = None,
        entity_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        """
        List all entities in the knowledge graph.

        Returns list of dicts with: name, entity_type, description, degree.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
        except Exception as e:
            logger.error(f"Failed to get KG nodes for workspace {self.workspace_id}: {e}")
            return []

        entities = []
        for node in all_nodes:
            node_id = node.get("id", "")
            etype = node.get("entity_type", "Unknown")
            desc = node.get("description", "")

            # Filters
            if entity_type and etype.lower() != entity_type.lower():
                continue
            if search and search.lower() not in node_id.lower():
                continue

            # Get degree (number of relationships)
            try:
                degree = await storage.node_degree(node_id)
            except Exception:
                degree = 0

            entities.append({
                "name": node_id,
                "entity_type": etype,
                "description": desc,
                "degree": degree,
            })

        # Sort by degree descending
        entities.sort(key=lambda e: e["degree"], reverse=True)

        return entities[offset:offset + limit]

    async def get_relationships(
        self,
        entity_name: str | None = None,
        limit: int = 500,
    ) -> list[dict]:
        """
        List relationships in the knowledge graph.

        If entity_name is provided, returns only relationships involving that entity.
        Returns list of dicts with: source, target, description, keywords, weight.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get KG edges for workspace {self.workspace_id}: {e}")
            return []

        relationships = []
        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")

            if entity_name:
                if entity_name.lower() not in (src.lower(), tgt.lower()):
                    continue

            relationships.append({
                "source": src,
                "target": tgt,
                "description": edge.get("description", ""),
                "keywords": edge.get("keywords", ""),
                "weight": float(edge.get("weight", 1.0)),
            })

        return relationships[:limit]

    async def get_graph_data(
        self,
        center_entity: str | None = None,
        max_depth: int = 3,
        max_nodes: int = 150,
    ) -> dict:
        """
        Export graph data for frontend visualization.

        Returns {nodes: [...], edges: [...], is_truncated: bool}.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            label = center_entity if center_entity else "*"
            kg = await storage.get_knowledge_graph(
                node_label=label,
                max_depth=max_depth,
                max_nodes=max_nodes,
            )
        except Exception as e:
            logger.error(f"Failed to get KG graph for workspace {self.workspace_id}: {e}")
            return {"nodes": [], "edges": [], "is_truncated": False}

        nodes = []
        for n in kg.nodes:
            props = n.properties if hasattr(n, "properties") else {}
            try:
                degree = await storage.node_degree(n.id)
            except Exception:
                degree = 0
            nodes.append({
                "id": n.id,
                "label": n.id,
                "entity_type": props.get("entity_type", "Unknown"),
                "degree": degree,
            })

        edges = []
        for e in kg.edges:
            props = e.properties if hasattr(e, "properties") else {}
            edges.append({
                "source": e.source,
                "target": e.target,
                "label": props.get("description", "")[:80],
                "weight": float(props.get("weight", 1.0)),
            })

        return {
            "nodes": nodes,
            "edges": edges,
            "is_truncated": kg.is_truncated if hasattr(kg, "is_truncated") else False,
        }

    def _simple_tokenize(self, question: str) -> Set[str]:
        """
        Fallback tokenization: simple split + lowercase + filter.

        Used when entity extraction fails or is disabled.

        Returns:
            Set of keyword strings (lowercased, punctuation stripped)
        """
        raw_tokens = question.lower().split()
        keywords = set()
        for token in raw_tokens:
            cleaned = token.strip(".,?!:;\"'()[]{}").lower()
            if len(cleaned) >= 2:
                keywords.add(cleaned)
        return keywords

    async def get_relevant_context(
        self,
        question: str,
        max_entities: int = 20,
        max_relationships: int = 30,
    ) -> str:
        """
        Build RAG context from raw KG data using embedding-based entity search.

        Scheme B: Instead of keyword matching, directly search entities_vdb
        with query embedding to find semantically relevant entities.

        Steps:
          1. Compute query embedding using configured embedding provider
          2. Search entities_vdb for top-k most similar entities
          3. Get relationships connecting those entities
          4. Format everything as structured factual text

        Returns:
            Structured string of entities + relationships, or "" if nothing found.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get raw KG data for workspace {self.workspace_id}: {e}")
            return ""

        if not all_nodes:
            return ""

        # -- 1. Compute query embedding --
        try:
            emb_provider = get_embedding_provider()
            query_embedding = await emb_provider.embed([question])
            query_embedding = query_embedding[0]  # Extract first (and only) vector
        except Exception as e:
            logger.error(f"Failed to compute query embedding for workspace {self.workspace_id}: {e}")
            # Fallback to simple tokenization if embedding fails
            return await self._get_relevant_context_by_keywords(question, max_entities, max_relationships, all_nodes, all_edges)

        # -- 2. Search entities_vdb for similar entities --
        entities_vdb = rag.entities_vdb
        try:
            search_results = await entities_vdb.query(
                query=question,
                top_k=max_entities,
                query_embedding=query_embedding
            )
        except Exception as e:
            logger.error(f"Failed to query entities_vdb for workspace {self.workspace_id}: {e}")
            return await self._get_relevant_context_by_keywords(question, max_entities, max_relationships, all_nodes, all_edges)

        if not search_results:
            logger.info(f"No entities found via embedding search for workspace {self.workspace_id}")
            return ""

        # -- 3. Build matched entity info from search results --
        matched_entity_names: list[str] = []
        entity_info: dict[str, dict] = {}

        # LightRAG entities_vdb stores content as "entity_name\ndescription"
        for result in search_results:
            content = result.get("content", "")
            entity_name = result.get("entity_name", "")

            # Parse content to get name and description
            if content and "\n" in content:
                parts = content.split("\n", 1)
                name = parts[0]
                description = parts[1] if len(parts) > 1 else ""
            elif entity_name:
                name = entity_name
                description = result.get("description", "")
            else:
                continue

            matched_entity_names.append(name)
            entity_info[name] = {
                "entity_type": result.get("entity_type", "Unknown"),
                "description": description,
            }

        if not matched_entity_names:
            return ""

        # -- 4. Find relationships involving matched entities --
        relevant_rels: list[dict] = []
        matched_lower = {n.lower() for n in matched_entity_names}

        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src.lower() in matched_lower or tgt.lower() in matched_lower:
                relevant_rels.append({
                    "source": src,
                    "target": tgt,
                    "description": edge.get("description", ""),
                    "keywords": edge.get("keywords", ""),
                })
                # Also add connected entities we might have missed
                if src not in entity_info:
                    for n in all_nodes:
                        if n.get("id", "") == src:
                            entity_info[src] = {
                                "entity_type": n.get("entity_type", "Unknown"),
                                "description": n.get("description", ""),
                            }
                            break
                if tgt not in entity_info:
                    for n in all_nodes:
                        if n.get("id", "") == tgt:
                            entity_info[tgt] = {
                                "entity_type": n.get("entity_type", "Unknown"),
                                "description": n.get("description", ""),
                            }
                            break

            if len(relevant_rels) >= max_relationships:
                break

        # -- 5. Format as structured text --
        parts: list[str] = []

        # Entities section
        if matched_entity_names:
            parts.append("Entities found in documents:")
            for name in matched_entity_names[:max_entities]:
                info = entity_info.get(name, {})
                etype = info.get("entity_type", "")
                desc = info.get("description", "")
                # Truncate long descriptions
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                type_str = f" [{etype}]" if etype and etype != "Unknown" else ""
                if desc:
                    parts.append(f"- {name}{type_str}: {desc}")
                else:
                    parts.append(f"- {name}{type_str}")

        # Relationships section
        if relevant_rels:
            parts.append("")
            parts.append("Relationships:")
            for rel in relevant_rels:
                desc = rel["description"]
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                if desc:
                    parts.append(f"- {rel['source']} → {rel['target']}: {desc}")
                else:
                    parts.append(f"- {rel['source']} → {rel['target']}")

        result = "\n".join(parts)
        logger.info(
            f"KG raw context (embedding search): {len(matched_entity_names)} entities, "
            f"{len(relevant_rels)} relationships for workspace {self.workspace_id}"
        )
        return result

    async def _get_relevant_context_by_keywords(
        self,
        question: str,
        max_entities: int,
        max_relationships: int,
        all_nodes: list[dict],
        all_edges: list[dict],
    ) -> str:
        """
        Fallback: original keyword-based matching method.
        Used when embedding search fails or is not available.
        """
        keywords = self._simple_tokenize(question)
        if not keywords:
            return ""

        matched_entity_names: set[str] = set()
        entity_info: dict[str, dict] = {}

        for node in all_nodes:
            node_id = node.get("id", "")
            node_lower = node_id.lower()

            matched = False
            for kw in keywords:
                if kw in node_lower or node_lower in kw:
                    matched = True
                    break
                for part in node_lower.split("-"):
                    if kw in part or part in kw:
                        matched = True
                        break
                if matched:
                    break

            if matched:
                matched_entity_names.add(node_id)
                entity_info[node_id] = {
                    "entity_type": node.get("entity_type", "Unknown"),
                    "description": node.get("description", ""),
                }

        if not matched_entity_names and len(all_nodes) <= 50:
            # Small graph: include top entities by default
            for node in all_nodes[:10]:
                nid = node.get("id", "")
                matched_entity_names.add(nid)
                entity_info[nid] = {
                    "entity_type": node.get("entity_type", "Unknown"),
                    "description": node.get("description", ""),
                }

        if not matched_entity_names:
            return ""

        matched_list = list(matched_entity_names)[:max_entities]

        # Find relationships
        relevant_rels: list[dict] = []
        matched_lower = {n.lower() for n in matched_list}

        for edge in all_edges:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            if src.lower() in matched_lower or tgt.lower() in matched_lower:
                relevant_rels.append({
                    "source": src,
                    "target": tgt,
                    "description": edge.get("description", ""),
                    "keywords": edge.get("keywords", ""),
                })
                if src not in entity_info:
                    for n in all_nodes:
                        if n.get("id", "") == src:
                            entity_info[src] = {
                                "entity_type": n.get("entity_type", "Unknown"),
                                "description": n.get("description", ""),
                            }
                            break
                if tgt not in entity_info:
                    for n in all_nodes:
                        if n.get("id", "") == tgt:
                            entity_info[tgt] = {
                                "entity_type": n.get("entity_type", "Unknown"),
                                "description": n.get("description", ""),
                            }
                            break
            if len(relevant_rels) >= max_relationships:
                break

        # Format
        parts: list[str] = []
        if matched_list:
            parts.append("Entities found in documents:")
            for name in matched_list:
                info = entity_info.get(name, {})
                etype = info.get("entity_type", "")
                desc = info.get("description", "")
                if len(desc) > 200:
                    desc = desc[:200] + "..."
                type_str = f" [{etype}]" if etype and etype != "Unknown" else ""
                if desc:
                    parts.append(f"- {name}{type_str}: {desc}")
                else:
                    parts.append(f"- {name}{type_str}")

        if relevant_rels:
            parts.append("")
            parts.append("Relationships:")
            for rel in relevant_rels:
                desc = rel["description"]
                if len(desc) > 150:
                    desc = desc[:150] + "..."
                if desc:
                    parts.append(f"- {rel['source']} → {rel['target']}: {desc}")
                else:
                    parts.append(f"- {rel['source']} → {rel['target']}")

        result = "\n".join(parts)
        logger.info(
            f"KG raw context (keyword fallback): {len(matched_list)} entities, "
            f"{len(relevant_rels)} relationships for workspace {self.workspace_id}"
        )
        return result

    async def get_analytics(self) -> dict:
        """
        Compute KG analytics summary.

        Returns: entity_count, relationship_count, entity_types, top_entities, avg_degree.
        """
        rag = await self._get_rag()
        storage = rag.chunk_entity_relation_graph

        try:
            all_nodes = await storage.get_all_nodes()
            all_edges = await storage.get_all_edges()
        except Exception as e:
            logger.error(f"Failed to get KG analytics for workspace {self.workspace_id}: {e}")
            return {
                "entity_count": 0,
                "relationship_count": 0,
                "entity_types": {},
                "top_entities": [],
                "avg_degree": 0.0,
            }

        entity_count = len(all_nodes)
        relationship_count = len(all_edges)

        # Count entity types
        type_counts: dict[str, int] = {}
        entities_with_degree = []
        for node in all_nodes:
            etype = node.get("entity_type", "Unknown")
            type_counts[etype] = type_counts.get(etype, 0) + 1
            try:
                degree = await storage.node_degree(node.get("id", ""))
            except Exception:
                degree = 0
            entities_with_degree.append({
                "name": node.get("id", ""),
                "entity_type": etype,
                "description": node.get("description", ""),
                "degree": degree,
            })

        # Sort by degree for top entities
        entities_with_degree.sort(key=lambda e: e["degree"], reverse=True)
        top_entities = entities_with_degree[:10]

        avg_degree = (
            sum(e["degree"] for e in entities_with_degree) / entity_count
            if entity_count > 0
            else 0.0
        )

        return {
            "entity_count": entity_count,
            "relationship_count": relationship_count,
            "entity_types": type_counts,
            "top_entities": top_entities,
            "avg_degree": round(avg_degree, 2),
        }
