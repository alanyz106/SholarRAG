"""
Vector Store Service
====================
Unified interface for vector storage with pluggable backends.

Default backend: Qdrant Cloud (QdrantVectorStore)
Future backends: ChromaVectorStore, MilvusVectorStore, etc.
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Sequence, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """
    Abstract base class for vector store backends.
    All concrete implementations must provide these methods.
    """

    @abstractmethod
    def add_documents(
        self,
        ids: Sequence[str],
        embeddings: Sequence[list[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict] | None = None,
    ) -> None:
        """Add documents with embeddings to the store."""
        ...

    @abstractmethod
    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        """
        Query for similar documents.

        Returns dict with keys: ids, documents, metadatas, distances.
        ChromaDB-style interface (ids→list[str], documents→list[str],
        metadatas→list[dict], distances→list[float]).
        """
        ...

    @abstractmethod
    def delete_by_document_id(self, document_id: int) -> None:
        """Delete all chunks belonging to a specific document."""
        ...

    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return the number of documents in the collection."""
        ...

    @abstractmethod
    def get_by_ids(self, ids: Sequence[str]) -> dict:
        """Get documents by their IDs. Returns dict with documents and metadatas."""
        ...


# ---------------------------------------------------------------------------
# Qdrant Backend
# ---------------------------------------------------------------------------

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchAny

_qdrant_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """Get or create the Qdrant client singleton."""
    global _qdrant_client

    if _qdrant_client is None:
        if not settings.QDRANT_URL:
            raise ValueError(
                "QDRANT_URL is not set. "
                "Please configure QDRANT_URL in your .env file."
            )
        logger.info(f"Connecting to Qdrant Cloud: {settings.QDRANT_URL}")
        _qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY if settings.QDRANT_API_KEY else None,
        )
        logger.info("Connected to Qdrant Cloud successfully")

    return _qdrant_client


class QdrantVectorStore(VectorStoreBase):
    """
    Qdrant Cloud implementation of VectorStoreBase.
    Each knowledge base has its own collection for namespace isolation.
    """

    COLLECTION_PREFIX = "kb_"
    VECTOR_DIMENSION = 1024  # bge-m3-v2 dimension

    def __init__(self, workspace_id: int):
        self.workspace_id = workspace_id
        self.collection_name = f"{self.COLLECTION_PREFIX}{workspace_id}"
        self._collection_initialized = False

    def _ensure_collection(self) -> None:
        """Ensure the collection exists, create if not."""
        if self._collection_initialized:
            return

        client = get_qdrant_client()

        collections = client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.VECTOR_DIMENSION,
                    distance=Distance.DOT,
                ),
            )
            logger.info(f"Collection {self.collection_name} created successfully")

            try:
                client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="document_id",
                    field_schema="integer",
                )
                logger.info(f"Created payload index on document_id for {self.collection_name}")
            except Exception as e:
                logger.warning(f"Could not create payload index on document_id: {e}")

        self._collection_initialized = True

    def _recreate_collection(self) -> None:
        """Delete and recreate the collection (resets cached reference)."""
        client = get_qdrant_client()
        try:
            client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Deleted collection {self.collection_name} for dimension migration")
        except Exception:
            pass
        self._collection_initialized = False
        self._ensure_collection()

    def add_documents(
        self,
        ids: Sequence[str],
        embeddings: Sequence[list[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict] | None = None,
    ) -> None:
        if not ids:
            return

        self._ensure_collection()
        client = get_qdrant_client()

        points = []
        for i, original_id in enumerate(ids):
            payload = {
                "content": documents[i],
                "original_id": original_id,
            }
            if metadatas:
                payload.update(metadatas[i])

            points.append({
                "id": _str_to_uuid(original_id),
                "vector": embeddings[i],
                "payload": payload,
            })

        try:
            client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "dimension" in error_msg or "vector_size" in error_msg:
                logger.warning(
                    f"Dimension mismatch in {self.collection_name}: {e}. "
                    f"Recreating collection for new embedding model."
                )
                self._recreate_collection()
                client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
            else:
                raise

        logger.info(f"Added {len(ids)} documents to collection {self.collection_name}")

    def query(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict | None = None,
        include: list[str] | None = None,
    ) -> dict:
        if include is None:
            include = ["documents", "metadatas", "distances"]

        self._ensure_collection()
        client = get_qdrant_client()

        qdrant_filter = None
        if where:
            conditions = []
            for key, value in where.items():
                if isinstance(value, dict) and "$in" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value["$in"]),
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=[value]),
                        )
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        try:
            results = client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=n_results,
                query_filter=qdrant_filter,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "dimension" in error_msg or "vector_size" in error_msg:
                logger.warning(
                    f"Dimension mismatch on query in {self.collection_name}: {e}. "
                    f"Collection needs reindexing."
                )
                return {"ids": [], "documents": [], "metadatas": [], "distances": []}
            raise

        points = results.points if hasattr(results, "points") else results
        ids_out, documents_out, metadatas_out, distances_out = [], [], [], []

        for point in points:
            payload = point.payload or {}
            ids_out.append(payload.get("original_id", str(point.id)))
            documents_out.append(payload.get("content", ""))
            meta = {k: v for k, v in payload.items() if k not in ("content", "original_id")}
            metadatas_out.append(meta)
            score = point.score if hasattr(point, "score") else 0.0
            distances_out.append(score)

        return {
            "ids": ids_out,
            "documents": documents_out,
            "metadatas": metadatas_out,
            "distances": distances_out,
        }

    def delete_by_document_id(self, document_id: int) -> None:
        self._ensure_collection()
        client = get_qdrant_client()

        client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=[document_id]),
                    )
                ]
            ),
        )
        logger.info(f"Deleted chunks for document {document_id} from collection {self.collection_name}")

    def delete_collection(self) -> None:
        client = get_qdrant_client()
        try:
            client.delete_collection(collection_name=self.collection_name)
            self._collection_initialized = False
            logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection {self.collection_name}: {e}")

    def count(self) -> int:
        self._ensure_collection()
        client = get_qdrant_client()
        result = client.get_collection(collection_name=self.collection_name)
        return result.points_count

    def get_by_ids(self, ids: Sequence[str]) -> dict:
        self._ensure_collection()
        client = get_qdrant_client()

        qdrant_ids = [_str_to_uuid(id_) for id_ in ids]

        results = client.retrieve(
            collection_name=self.collection_name,
            ids=qdrant_ids,
            with_payload=True,
            with_vectors=False,
        )

        documents_out = []
        metadatas_out = []

        for point in results:
            payload = point.payload or {}
            documents_out.append(payload.get("content", ""))
            meta = {k: v for k, v in payload.items() if k not in ("content", "original_id")}
            metadatas_out.append(meta)

        return {
            "documents": documents_out,
            "metadatas": metadatas_out,
        }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store(workspace_id: int) -> VectorStoreBase:
    """Factory: create a VectorStore instance based on VECTOR_STORE_PROVIDER config.

    Currently only "qdrant" is supported.
    """
    provider = getattr(settings, "VECTOR_STORE_PROVIDER", "qdrant").lower()

    if provider == "qdrant":
        return QdrantVectorStore(workspace_id)

    raise ValueError(
        f"VECTOR_STORE_PROVIDER '{provider}' is not supported. "
        f"Available: qdrant"
    )


def _str_to_uuid(str_id: str) -> str:
    """Convert a string ID to a deterministic UUID for Qdrant."""
    namespace_uuid = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")
    return str(uuid.uuid5(namespace_uuid, str_id))


# ---------------------------------------------------------------------------
# Backward-compatible aliases (deprecated, use get_vector_store + VectorStoreBase)
# ---------------------------------------------------------------------------
VectorStore = QdrantVectorStore
