"""
Vector Store Service
Handles Qdrant Cloud operations for storing and retrieving document embeddings.
"""
from __future__ import annotations

import logging
from typing import Sequence, Optional, TYPE_CHECKING
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchAny

from app.core.config import settings

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Global Qdrant client
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


class VectorStore:
    """
    Vector store service for managing document embeddings in Qdrant.
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

        # Check if collection exists
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
        metadatas: Sequence[dict] | None = None
    ) -> None:
        """
        Add documents with their embeddings to the collection.
        Auto-handles dimension mismatch: if the collection was created with
        a different embedding dimension, it is deleted and recreated.
        """
        if not ids:
            return

        self._ensure_collection()
        client = get_qdrant_client()

        # Prepare points
        points = []
        for i, point_id in enumerate(ids):
            payload = {
                "content": documents[i],
            }
            if metadatas:
                payload.update(metadatas[i])

            points.append({
                "id": point_id,
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
        include: list[str] | None = None
    ) -> dict:
        """Query the collection for similar documents."""
        if include is None:
            include = ["documents", "metadatas", "distances"]

        self._ensure_collection()
        client = get_qdrant_client()

        # Build Qdrant filter from ChromaDB-style where clause
        qdrant_filter = None
        if where:
            conditions = []
            for key, value in where.items():
                if isinstance(value, dict) and "$in" in value:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value["$in"])
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=[value])
                        )
                    )
            if conditions:
                qdrant_filter = Filter(must=conditions)

        try:
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
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

        # Flatten results - Qdrant returns list of ScoredPoint
        ids = []
        documents = []
        metadatas = []
        distances = []

        for point in results:
            ids.append(str(point.id))
            payload = point.payload or {}

            # Content is stored in "content" field
            documents.append(payload.get("content", ""))

            # Metadata is everything except "content"
            meta = {k: v for k, v in payload.items() if k != "content"}
            metadatas.append(meta)

            # Qdrant returns score (Dot Product), we store as distance
            score = point.score if hasattr(point, 'score') else 0.0
            distances.append(score)

        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
        }

    def delete_by_document_id(self, document_id: int) -> None:
        """Delete all chunks belonging to a specific document."""
        self._ensure_collection()
        client = get_qdrant_client()

        client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchAny(any=[document_id])
                    )
                ]
            ),
        )
        logger.info(f"Deleted chunks for document {document_id} from collection {self.collection_name}")

    def delete_collection(self) -> None:
        """Delete the entire collection for this knowledge base."""
        client = get_qdrant_client()
        try:
            client.delete_collection(collection_name=self.collection_name)
            self._collection_initialized = False
            logger.info(f"Deleted collection {self.collection_name}")
        except Exception as e:
            logger.warning(f"Failed to delete collection {self.collection_name}: {e}")

    def count(self) -> int:
        """Return the number of documents in the collection."""
        self._ensure_collection()
        client = get_qdrant_client()
        result = client.get_collection(collection_name=self.collection_name)
        return result.points_count

    def get_by_ids(self, ids: Sequence[str]) -> dict:
        """Get documents by their IDs."""
        self._ensure_collection()
        client = get_qdrant_client()

        results = client.retrieve(
            collection_name=self.collection_name,
            ids=list(ids),
            with_payload=True,
            with_vectors=False,
        )

        documents = []
        metadatas = []

        for point in results:
            payload = point.payload or {}
            documents.append(payload.get("content", ""))
            meta = {k: v for k, v in payload.items() if k != "content"}
            metadatas.append(meta)

        return {
            "documents": documents,
            "metadatas": metadatas,
        }


def get_vector_store(workspace_id: int) -> VectorStore:
    """Factory function to create a VectorStore for a knowledge base."""
    return VectorStore(workspace_id)
