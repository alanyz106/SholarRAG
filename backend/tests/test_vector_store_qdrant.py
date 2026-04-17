"""
Tests for Qdrant VectorStore implementation.
"""
import pytest
from app.services.vector_store import VectorStore, get_qdrant_client
from app.core.config import settings


class TestQdrantConnection:
    """Test Qdrant client connectivity."""

    def test_qdrant_client_initializes(self):
        """Verify Qdrant client can be initialized."""
        if not settings.QDRANT_URL:
            pytest.skip("QDRANT_URL not configured")
        client = get_qdrant_client()
        assert client is not None
        collections = client.get_collections()
        assert collections is not None


class TestVectorStore:
    """Test VectorStore CRUD operations."""

    @pytest.fixture
    def vector_store(self):
        """Create a VectorStore for testing."""
        return VectorStore(workspace_id=9999)

    def test_collection_naming(self, vector_store):
        """Test collection name follows kb_{id} pattern."""
        assert vector_store.collection_name == "kb_9999"

    def test_add_and_query_documents(self, vector_store):
        """Test adding documents and querying them."""
        if not settings.QDRANT_URL:
            pytest.skip("QDRANT_URL not configured")

        # Setup
        test_ids = ["test_1", "test_2"]
        test_embeddings = [[0.1] * 1024, [0.2] * 1024]  # Fake 1024-dim embeddings
        test_documents = ["This is test document 1", "This is test document 2"]
        test_metadatas = [
            {"document_id": 1, "chunk_index": 0, "source": "test.pdf", "page_no": 1},
            {"document_id": 1, "chunk_index": 1, "source": "test.pdf", "page_no": 1},
        ]

        # Add documents
        vector_store.add_documents(
            ids=test_ids,
            embeddings=test_embeddings,
            documents=test_documents,
            metadatas=test_metadatas,
        )

        # Query (use same fake embedding since we're testing connectivity)
        results = vector_store.query(
            query_embedding=[0.1] * 1024,
            n_results=2,
        )

        # Verify structure
        assert "ids" in results
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results

        # Cleanup
        vector_store.delete_collection()

    def test_delete_by_document_id(self, vector_store):
        """Test deleting documents by document_id."""
        if not settings.QDRANT_URL:
            pytest.skip("QDRANT_URL not configured")

        # Add test document
        vector_store.add_documents(
            ids=["test_del_1", "test_del_2"],
            embeddings=[[0.1] * 1024, [0.2] * 1024],
            documents=["Doc content 1", "Doc content 2"],
            metadatas=[
                {"document_id": 100, "chunk_index": 0},
                {"document_id": 100, "chunk_index": 1},
            ],
        )

        # Delete by document_id
        vector_store.delete_by_document_id(document_id=100)

        # Verify count is 0
        assert vector_store.count() == 0

        # Cleanup
        vector_store.delete_collection()

    def test_count(self, vector_store):
        """Test document count."""
        if not settings.QDRANT_URL:
            pytest.skip("QDRANT_URL not configured")

        # Initially empty
        initial_count = vector_store.count()

        # Add documents
        vector_store.add_documents(
            ids=["count_1", "count_2", "count_3"],
            embeddings=[[0.1] * 1024, [0.2] * 1024, [0.3] * 1024],
            documents=["Doc 1", "Doc 2", "Doc 3"],
            metadatas=[
                {"document_id": 200, "chunk_index": 0},
                {"document_id": 200, "chunk_index": 1},
                {"document_id": 200, "chunk_index": 2},
            ],
        )

        assert vector_store.count() == initial_count + 3

        # Cleanup
        vector_store.delete_collection()
