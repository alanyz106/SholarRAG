# ChromaDB → Qdrant Cloud 重构设计文档

## 1. 背景与目标

将 NexusRAG 项目中的向量数据库从本地 ChromaDB 迁移到 Qdrant Cloud，适配原有的主要功能。选用 bge-m3-v2 作为 embedding 模型（1024维，OpenAI-compatible API）。

## 2. 架构概览

```
现有架构：
  ChromaDB (本地 PersistentClient)
  ↓
  VectorStore 类（封装 ChromaDB API）
  ↓
  DeepRetriever / RAG Service → API层

迁移后架构：
  Qdrant Cloud (托管服务)
  ↓
  VectorStore 类（封装 Qdrant Client API）
  ↓
  DeepRetriever / RAG Service → API层
```

## 3. 核心变更

| 文件 | 变更内容 |
|------|----------|
| `requirements.txt` | 移除 `chromadb`，新增 `qdrant-client>=1.12.0` |
| `config.py` | 新增 `QDRANT_*` 配置组，注释掉 `CHROMA_*` 相关配置 |
| `vector_store.py` | 完全重写：用 `qdrant_client` 替换 `chromadb`，Collection/Point API 替换 |
| `.env.example` | 新增 `QDRANT_*` 配置项 |
| `deep_retriever.py` | 仅修改 import，逻辑不变（接口保持一致） |

## 4. Qdrant Collection 设计

### Collection 命名
- 每个 workspace 一个 collection，命名格式：`kb_{workspace_id}`
- 与原有 ChromaDB 命名一致，改动最小

### 向量配置
- **维度**：1024（bge-m3-v2）
- **距离算法**：Dot Product（Qdrant 推荐，比 Cosine 在许多场景下更快更准）

### Payload 结构
与 ChromaDB metadata 结构一致：

```json
{
  "document_id": 123,
  "chunk_index": 0,
  "source": "filename.pdf",
  "page_no": 1,
  "heading_path": "Chapter 1 > Section A",
  "image_ids": "img_1|img_2",
  "table_ids": "tbl_1",
  "has_table": false,
  "has_code": false
}
```

## 5. 接口保持不变

```python
class VectorStore:
    COLLECTION_PREFIX = "kb_"

    def __init__(self, workspace_id: int): ...
    def add_documents(ids, embeddings, documents, metadatas) -> None
    def query(query_embedding, n_results, where, include) -> dict
    def delete_by_document_id(document_id) -> None
    def delete_collection() -> None
    def count() -> int
    def get_by_ids(ids) -> dict
```

### query() 返回格式保持一致

```python
{
    "ids": [...],
    "documents": [...],
    "metadatas": [...],
    "distances": [...]
}
```

注意：Qdrant 返回的是 "scores"（Dot Product 相似度），ChromaDB 返回的是 "distances"（Cosine 距离）。需要将 scores 转换为 distances 以保持接口兼容：
- `distance = 1 - score`（适用于归一化向量的 Dot Product）

## 6. 配置项设计

### .env 新增配置

```bash
# Qdrant Cloud
QDRANT_URL=https://83a45c96-a46f-4a1f-b95d-c378ad81fd2a.sa-east-1-0.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...

# Embedding（统一走 OpenAI-compatible bge-m3-v2）
CHROMA_EMBEDDING_PROVIDER=openai
CHROMA_OPENAI_API_KEY=your-api-key
CHROMA_OPENAI_BASE_URL=https://your-bge-m3-v2-endpoint/v1
CHROMA_OPENAI_MODEL=bge-m3-v2
CHROMA_OPENAI_DIMENSION=1024
```

### config.py 新增字段

```python
# Qdrant Cloud
QDRANT_URL: str = Field(default="")
QDRANT_API_KEY: str = Field(default="")
```

## 7. 实现要点

### 7.1 Qdrant Client 初始化
```python
from qdrant_client import QdrantClient

_qdrant_client: Optional[QdrantClient] = None

def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
    return _qdrant_client
```

### 7.2 Collection 自动创建
首次添加文档时检查 collection 是否存在，不存在则自动创建：
```python
def _ensure_collection(self):
    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    if self.collection_name not in collections:
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.DOT),
        )
```

### 7.3 Dimension Mismatch 处理
添加文档时若遇到 dimension 不匹配，删除并重建 collection：
```python
except Exception as e:
    if "dimension" in str(e).lower():
        self._recreate_collection()
        # Retry...
```

### 7.4 删除 Collection
使用 `delete_collection` API。

### 7.5 查询条件转换
ChromaDB 的 `where={"document_id": {"$in": [1,2]}}` 转换为 Qdrant filter：
```python
from qdrant_client.models import Filter, FieldCondition, MatchAny

if where:
    conditions = []
    for key, value in where.items():
        if isinstance(value, dict) and "$in" in value:
            conditions.append(
                FieldCondition(key=key, match=MatchAny(any=value["$in"]))
            )
    qdrant_filter = Filter(must=conditions) if conditions else None
```

## 8. 依赖变更

```diff
# requirements.txt
- chromadb>=0.4.22
+ qdrant-client>=1.12.0
```

## 9. 测试计划

1. **单元测试**：VectorStore 类的 CRUD 操作
2. **集成测试**：实际连接 Qdrant Cloud 进行查询
3. **回归测试**：确保现有 RAG 流程（上传文档 → 检索 → 生成回答）正常工作

## 10. 风险与注意事项

1. **网络依赖**：Qdrant Cloud 依赖网络连接，本地 ChromaDB 是本地的
2. **API 配额**：注意 Qdrant Cloud 的 API 配额限制
3. **数据迁移**：不保留原有 ChromaDB 数据，新文档将写入 Qdrant
