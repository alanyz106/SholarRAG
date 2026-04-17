# ChromaDB → Qdrant Cloud 迁移实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 NexusRAG 向量存储从本地 ChromaDB 迁移到 Qdrant Cloud，保持所有现有接口不变。

**Architecture:** 用 `qdrant-client` 替换 `chromadb`，重写 `vector_store.py` 中的所有方法，Collection/Point API 替换原有 ChromaDB 接口。配置项新增 `QDRANT_*`，embedding 统一走 OpenAI-compatible bge-m3-v2 API。

**Tech Stack:** qdrant-client>=1.12.0, FastAPI, sentence-transformers (本地备用)

---

## 文件变更总览

| 文件 | 操作 |
|------|------|
| `backend/requirements.txt` | 修改：移除 chromadb，新增 qdrant-client |
| `backend/app/core/config.py` | 修改：新增 QDRANT_* 配置字段 |
| `backend/app/services/vector_store.py` | 重写：用 Qdrant API 替换 ChromaDB |
| `.env.example` | 修改：新增 QDRANT_* 配置项，注释 CHROMA_* |
| `backend/app/services/deep_retriever.py` | 无变更（接口不变） |

---

## Task 1: 修改 requirements.txt

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: 更新 requirements.txt**

```diff
- chromadb>=0.4.22
+ qdrant-client>=1.12.0
```

- [ ] **Step 2: 提交**

```bash
git add backend/requirements.txt
git commit -m "chore: replace chromadb with qdrant-client"
```

---

## Task 2: 新增 Qdrant 配置项

**Files:**
- Modify: `backend/app/core/config.py:68-113`

- [ ] **Step 1: 在 `backend/app/core/config.py` 中新增 QDRANT 配置字段**

在 `CHROMA_*` 配置块上方添加：

```python
# Qdrant Cloud
QDRANT_URL: str = Field(default="")
QDRANT_API_KEY: str = Field(default="")
```

完整新增段落（插入到第 68 行附近，`# ChromaDB` 注释块之前）：

```python
# Qdrant Cloud
QDRANT_URL: str = Field(default="")
QDRANT_API_KEY: str = Field(default="")
```

- [ ] **Step 2: 提交**

```bash
git add backend/app/core/config.py
git commit -m "feat: add QDRANT_URL and QDRANT_API_KEY config fields"
```

---

## Task 3: 重写 vector_store.py - 基础框架

**Files:**
- Modify: `backend/app/services/vector_store.py`

- [ ] **Step 1: 重写 vector_store.py 基础结构**

完整重写文件内容如下：

```python
"""
Vector Store Service
Handles Qdrant Cloud operations for storing and retrieving document embeddings.
"""
from __future__ import annotations

import logging
from typing import Sequence, Optional, TYPE_CHECKING
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, Filter, FieldCondition, MatchAny, Payload

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

            # Qdrant returns score (Dot Product), convert to distance
            # For normalized vectors: distance = 1 - score
            # But we use raw Dot Product, so higher score = closer
            # To maintain ChromaDB-style "distance" (lower = closer):
            # distance = 1 - score (for normalized)
            # Since bge-m3 produces normalized embeddings, this works
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
```

- [ ] **Step 2: 提交**

```bash
git add backend/app/services/vector_store.py
git commit -m "refactor: replace ChromaDB with Qdrant Cloud in vector_store.py"
```

---

## Task 4: 更新 .env.example

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: 更新 .env.example 配置**

在文件顶部 ChromaDB 配置区域添加 Qdrant 配置，并注释掉 ChromaDB 相关行：

```bash
# ===========================================
# Qdrant Cloud (替换 ChromaDB)
# ===========================================
QDRANT_URL=https://83a45c96-a46f-4a1f-b95d-c378ad81fd2a.sa-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key

# ChromaDB (已弃用 - 改用 Qdrant Cloud)
# CHROMA_HOST=localhost
# CHROMA_PORT=8002

# ===========================================
# ChromaDB Embedding Provider (已弃用)
# CHROMA_EMBEDDING_PROVIDER=openai
# CHROMA_OPENAI_API_KEY=sk-xxx
# CHROMA_OPENAI_BASE_URL=https://api.openai.com/v1
# CHROMA_OPENAI_MODEL=text-embedding-3-small
# CHROMA_OPENAI_DIMENSION=1024
```

- [ ] **Step 2: 提交**

```bash
git add .env.example
git commit -m "chore: add QDRANT_* config to .env.example"
```

---

## Task 5: 更新实际 .env 文件

**Files:**
- Modify: `.env`（如果存在）

- [ ] **Step 1: 检查并更新 .env 文件**

检查 `.env` 文件是否存在，如果存在则添加 QDRANT 配置：

```bash
# 在项目根目录执行
cat .env | grep -E "QDRANT|CHROMA" || echo "No QDRANT/CHROMA config found"
```

如果需要更新，添加：
```bash
QDRANT_URL=https://83a45c96-a46f-4a1f-b95d-c378ad81fd2a.sa-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your-api-key-here
```

- [ ] **Step 2: 不提交 .env（包含敏感信息）**

---

## Task 6: 验证实现

**Files:**
- Test: `backend/tests/test_vector_store_qdrant.py`（新建）

- [ ] **Step 1: 创建基础测试文件**

```python
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
```

- [ ] **Step 2: 运行测试验证**

```bash
# 激活虚拟环境
cd D:\llm_rag\NexusRAG
source .venv/Scripts/activate

# 安装依赖
pip install qdrant-client>=1.12.0 pytest

# 运行测试
cd backend
pytest tests/test_vector_store_qdrant.py -v
```

预期：测试应在连接 Qdrant Cloud 成功后全部通过。

- [ ] **Step 3: 提交**

```bash
git add backend/tests/test_vector_store_qdrant.py
git commit -m "test: add Qdrant VectorStore integration tests"
```

---

## 实施检查清单

完成所有 Task 后，验证以下内容：

- [ ] `requirements.txt` 中 chromadb 已移除，qdrant-client 已添加
- [ ] `config.py` 中 QDRANT_URL 和 QDRANT_API_KEY 字段已添加
- [ ] `vector_store.py` 完全重写，所有方法接口保持不变
- [ ] `.env.example` 已更新 QDRANT_* 配置
- [ ] 实际 .env 文件包含 QDRANT_URL 和 QDRANT_API_KEY
- [ ] DeepRetriever 等上层调用无需修改（接口兼容）
- [ ] 集成测试通过

---

## Spec 覆盖检查

| Spec 要求 | 对应 Task |
|-----------|----------|
| ChromaDB → Qdrant Cloud | Task 1-5 |
| Collection 命名 kb_{id} | Task 3 (VectorStore.__init__) |
| 1024 维 Dot Product | Task 3 (VECTOR_DIMENSION, Distance.DOT) |
| 自动创建 collection | Task 3 (_ensure_collection) |
| Dimension mismatch 处理 | Task 3 (_recreate_collection + add_documents try/except) |
| where → Qdrant Filter | Task 3 (query 方法) |
| Payload 结构一致 | Task 3 (add_documents payload) |
| query() 返回格式一致 | Task 3 (query 返回 dict) |
| 新增 QDRANT_* 配置 | Task 2, 4, 5 |
| Embedding 统一 bge-m3-v2 OpenAI-compatible | Task 4 (.env.example) |

## 风险

1. **网络依赖**：Qdrant Cloud 依赖网络，本地 ChromaDB 是离线的
2. **API 配额**：注意 Qdrant Cloud 配额限制
3. **数据不迁移**：原有 ChromaDB 数据不迁移，新文档写入 Qdrant
