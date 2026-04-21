# NexusRAG 重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重构项目配置和代码，删除废弃的 ChromaDB/Gemini/Ollama 代码，统一变量命名，按功能分组 .env 配置

**Architecture:**
1. 配置层：重写 `config.py`，清理废弃变量，按功能分组 `.env.example`
2. Provider 层：删除 Ollama/Gemini/Sentence-Transformers 相关代码，只保留 OpenAI-compatible
3. 服务层：拆分 `rag.py` 中的 chat 端点为独立服务
4. 清理层：删除废弃文件，更新 import

**Tech Stack:** Python 3.11, FastAPI, Pydantic Settings, Qdrant, OpenAI-compatible API

---

## 文件变更总览

| 操作 | 文件 |
|------|------|
| 删除 | `backend/app/services/llm/gemini.py` |
| 删除 | `backend/app/services/llm/ollama.py` |
| 删除 | `backend/app/services/llm/sentence_transformer.py` |
| 删除 | `backend/app/services/llm/gitee_ai.py` |
| 修改 | `backend/app/core/config.py` |
| 修改 | `backend/app/services/llm/__init__.py` |
| 修改 | `backend/app/services/embedder.py` |
| 修改 | `backend/app/services/reranker.py` |
| 修改 | `backend/app/api/rag.py` |
| 修改 | `backend/app/api/chat_agent.py` |
| 修改 | `backend/.env.example` |
| 创建 | `backend/app/services/chat_service.py` |
| 创建 | `backend/app/services/retrieval_service.py` |

---

## Task 1: 重写 config.py 配置类

**Files:**
- Modify: `backend/app/core/config.py`

- [ ] **Step 1: 备份现有 config.py**

```python
# 备份原有配置类结构，标记需要删除的字段
```

- [ ] **Step 2: 重写 Settings 类，删除废弃变量**

新的配置分组结构：

```python
# === 数据库 ===
DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@localhost:5433/nexusrag")

# === 向量存储 (Qdrant) ===
VECTOR_STORE_PROVIDER: str = "qdrant"  # 固定为 qdrant，删除 chroma 配置
QDRANT_URL: str = Field(default="")
QDRANT_API_KEY: str = Field(default="")

# === 嵌入服务 (向量检索) ===
EMBEDDING_PROVIDER: str = "openai"  # 只支持 openai
EMBEDDING_MODEL: str = "text-embedding-3-small"
EMBEDDING_DIMENSION: int = 1536
EMBEDDING_OPENAI_API_KEY: str = Field(default="")
EMBEDDING_OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")
EMBEDDING_OPENAI_ORGANIZATION: Optional[str] = Field(default=None)

# === LLM (对话) ===
LLM_PROVIDER: str = "openai"  # 只支持 openai
LLM_MODEL: str = "gpt-4o-mini"
LLM_OPENAI_API_KEY: str = Field(default="")
LLM_OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")
LLM_OPENAI_ORGANIZATION: Optional[str] = Field(default=None)
LLM_MAX_TOKENS: int = 8192

# === 重排序服务 ===
RERANKER_PROVIDER: str = "gitee_ai"  # gitee_ai | siliconflow
GITEE_AI_API_TOKEN: str = Field(default="")
GITEE_AI_RERANK_MODEL: str = "bge-reranker-v2-m3"
GITEE_AI_RERANK_TOP_N: int = 10
SILICONFLOW_API_KEY: str = Field(default="")
SILICONFLOW_RERANK_MODEL: str = "BAAI/bge-reranker-v2-m3"
SILICONFLOW_RERANK_TOP_N: int = 10

# === 知识图谱 ===
KG_ENABLED: bool = True
KG_LANGUAGE: str = "Chinese"
KG_ENTITY_TYPES: list[str] = [...]
KG_OPENAI_API_KEY: str = Field(default="")  # 独立 key

# === NexusRAG Pipeline ===
NEXUSRAG_DOCLING_DEVICE: str = "auto"
NEXUSRAG_CHUNK_MAX_TOKENS: int = 512
NEXUSRAG_VECTOR_PREFETCH: int = 20
NEXUSRAG_RERANKER_TOP_K: int = 8
NEXUSRAG_MIN_RELEVANCE_SCORE: float = 0.15

# === 评估 (Judge) ===
JUDGE_PROVIDER: str = "openai"
JUDGE_OPENAI_API_KEY: str = Field(default="")
JUDGE_OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1")
JUDGE_OPENAI_MODEL: str = "gpt-4o"

# === CORS ===
CORS_ORIGINS: list[str] = ["http://localhost:5174"]
```

**删除的字段（不要在代码中引用）：**
- `LLM_THINKING_LEVEL`, `LLM_MODEL_FAST` (Gemini)
- `GOOGLE_AI_API_KEY` (Gemini)
- `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_ENABLE_THINKING` (Ollama)
- `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_EMBEDDING_PROVIDER`
- `CHROMA_OPENAI_*`, `CHROMA_GEMINI_*`, `CHROMA_OLLAMA_*`
- `KG_EMBEDDING_PROVIDER`, `KG_EMBEDDING_MODEL`, `KG_EMBEDDING_DIMENSION`
- `KG_OPENAI_API_KEY`, `KG_OPENAI_BASE_URL`, `KG_OPENAI_ORGANIZATION`
- `COHERE_*`, `JINA_*`, `MODELSCOPE_*`
- `NEXUSRAG_EMBEDDING_MODEL` (合并到 EMBEDDING_MODEL)
- `NEXUSRAG_RERANKER_MODEL` (合并到 RERANKER_* 配置)
- `RERANKER_PROVIDER` 保留，但精简选项
- `NEXUSRAG_KG_LANGUAGE`, `NEXUSRAG_KG_ENTITY_TYPES` → 移到 `KG_*`

- [ ] **Step 3: 运行测试验证配置加载**

```bash
cd D:/llm_rag/NexusRAG
source .venv/Scripts/activate
cd backend
python -c "from app.core.config import settings; print(settings.LLM_PROVIDER, settings.EMBEDDING_PROVIDER)"
```

预期输出: `openai openai`

- [ ] **Step 4: 提交**

```bash
git add backend/app/core/config.py
git commit -m "refactor(config): rewrite settings, remove deprecated ChromaDB/Gemini/Ollama vars"
```

---

## Task 2: 删除废弃的 LLM Provider 文件

**Files:**
- Delete: `backend/app/services/llm/gemini.py`
- Delete: `backend/app/services/llm/ollama.py`
- Delete: `backend/app/services/llm/sentence_transformer.py`
- Delete: `backend/app/services/llm/gitee_ai.py`

- [ ] **Step 1: 确认无其他文件引用这些模块**

```bash
cd D:/llm_rag/NexusRAG
grep -r "from app.services.llm.gemini\|from app.services.llm.ollama\|from app.services.llm.sentence_transformer\|from app.services.llm.gitee_ai" backend/ --include="*.py"
```

预期: 无输出

- [ ] **Step 2: 删除文件**

```bash
rm backend/app/services/llm/gemini.py
rm backend/app/services/llm/ollama.py
rm backend/app/services/llm/sentence_transformer.py
rm backend/app/services/llm/gitee_ai.py
```

- [ ] **Step 3: 验证删除成功**

```bash
ls backend/app/services/llm/
```

预期输出: `__init__.py base.py openai.py types.py`

- [ ] **Step 4: 提交**

```bash
git add backend/app/services/llm/
git commit -m "refactor(llm): remove deprecated providers (gemini, ollama, sentence_transformer, gitee_ai)"
```

---

## Task 3: 重写 llm/__init__.py

**Files:**
- Modify: `backend/app/services/llm/__init__.py`

- [ ] **Step 1: 重写 get_llm_provider，只支持 openai**

```python
@lru_cache
def get_llm_provider() -> LLMProvider:
    """Create (and cache) the LLM provider. Only OpenAI-compatible is supported."""
    from app.core.config import settings

    if settings.LLM_PROVIDER != "openai":
        raise ValueError(f"LLM_PROVIDER must be 'openai', got '{settings.LLM_PROVIDER}'")

    from app.services.llm.openai import OpenAILLMProvider

    if not settings.LLM_OPENAI_API_KEY:
        raise ValueError("LLM_OPENAI_API_KEY is required when LLM_PROVIDER=openai")

    return OpenAILLMProvider(
        api_key=settings.LLM_OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        base_url=settings.LLM_OPENAI_BASE_URL,
        organization=settings.LLM_OPENAI_ORGANIZATION,
    )
```

- [ ] **Step 2: 重写 get_embedding_provider，只支持 openai**

```python
@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    """Create (and cache) the embedding provider. Only OpenAI-compatible is supported."""
    from app.core.config import settings

    if settings.EMBEDDING_PROVIDER != "openai":
        raise ValueError(f"EMBEDDING_PROVIDER must be 'openai', got '{settings.EMBEDDING_PROVIDER}'")

    from app.services.llm.openai import OpenAIEmbeddingProvider

    api_key = settings.EMBEDDING_OPENAI_API_KEY or settings.LLM_OPENAI_API_KEY
    base_url = settings.EMBEDDING_OPENAI_BASE_URL or settings.LLM_OPENAI_BASE_URL
    organization = settings.EMBEDDING_OPENAI_ORGANIZATION or settings.LLM_OPENAI_ORGANIZATION

    if not api_key:
        raise ValueError("EMBEDDING_OPENAI_API_KEY or LLM_OPENAI_API_KEY is required")

    return OpenAIEmbeddingProvider(
        api_key=api_key,
        model=settings.EMBEDDING_MODEL,
        base_url=base_url,
        organization=organization,
    )
```

- [ ] **Step 3: 清理 __all__ 列表**

```python
__all__ = [
    "get_llm_provider",
    "get_embedding_provider",
    "LLMProvider",
    "EmbeddingProvider",
]
```

- [ ] **Step 4: 验证导入**

```bash
cd backend
python -c "from app.services.llm import get_llm_provider, get_embedding_provider; print('OK')"
```

- [ ] **Step 5: 提交**

```bash
git add backend/app/services/llm/__init__.py
git commit -m "refactor(llm): simplify to OpenAI-only provider, remove branching logic"
```

---

## Task 4: 重写 embedder.py

**Files:**
- Modify: `backend/app/services/embedder.py`

- [ ] **Step 1: 重写 EmbeddingService，只保留 openai 分支**

删除 `sentence_transformers`、`gemini`、`ollama` 分支。新的 EmbeddingService 直接使用 `get_embedding_provider()`。

```python
class EmbeddingService:
    """Service for generating text embeddings. Only OpenAI-compatible provider."""

    def __init__(self, model_name: Optional[str] = None):
        from app.services.llm import get_embedding_provider
        self._provider = get_embedding_provider()
        self.model_name = model_name or settings.EMBEDDING_MODEL

    @property
    def dimension(self) -> int:
        return self._provider.get_dimension()

    def embed_text(self, text: str) -> list[float]:
        embeddings = self._provider.embed_sync([text])
        return embeddings[0].tolist()

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        embeddings = self._provider.embed_sync(list(texts))
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)
```

- [ ] **Step 2: 移除 settings 中的 CHROMA_EMBEDDING_* 引用**

将 `embedder.py` 中所有 `settings.CHROMA_EMBEDDING_*` 替换为新命名的 `settings.EMBEDDING_*`

- [ ] **Step 3: 验证导入**

```bash
cd backend
python -c "from app.services.embedder import get_embedding_service; print('OK')"
```

- [ ] **Step 4: 提交**

```bash
git add backend/app/services/embedder.py
git commit -m "refactor(embedder): simplify to OpenAI-only, use unified EMBEDDING_* config"
```

---

## Task 5: 重写 reranker.py

**Files:**
- Modify: `backend/app/services/reranker.py`

- [ ] **Step 1: 分析现有 reranker.py 结构**

```bash
head -100 backend/app/services/reranker.py
```

- [ ] **Step 2: 清理 reranker.py，只保留 gitee_ai 和 siliconflow**

删除 `sentence_transformers`、`cohere`、`jina`、`modelscope` 相关代码

- [ ] **Step 3: 更新配置引用**

将 `settings.GITEE_AI_*` 保留，更新 `settings.SILICONFLOW_*` 命名一致性

- [ ] **Step 4: 验证**

```bash
cd backend
python -c "from app.services.reranker import get_reranker_service; print('OK')"
```

- [ ] **Step 5: 提交**

```bash
git add backend/app/services/reranker.py
git commit -m "refactor(reranker): remove deprecated providers, keep only gitee_ai and siliconflow"
```

---

## Task 6: 创建 retrieval_service.py

**Files:**
- Create: `backend/app/services/retrieval_service.py`

- [ ] **Step 1: 创建 retrieval_service.py**

从 `rag.py` 的 chat 端点提取检索逻辑，从 `chat_agent.py` 提取 `_execute_search_documents` 逻辑。

```python
"""
Retrieval Service
================
Unified retrieval logic for NexusRAG: vector search, KG search, image resolution.
"""
from dataclasses import dataclass
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_

from app.models.document import DocumentImage
from app.services.llm.types import LLMImagePart

@dataclass
class RetrievalResult:
    """Result from retrieval service."""
    context: str
    chunks: list
    citations: list
    image_refs: list
    image_parts: list[dict]  # For vision models

async def retrieve(
    workspace_id: int,
    question: str,
    top_k: int,
    document_ids: Optional[list[int]],
    db: AsyncSession,
    existing_ids: set[str],
) -> RetrievalResult:
    """Execute hybrid retrieval for a question."""
    # 实现检索逻辑
    pass

def resolve_images_from_chunks(
    chunks: list,
    db: AsyncSession,
    workspace_id: int,
    existing_ids: set[str],
) -> tuple[list, list, list[dict]]:
    """Resolve images from chunk metadata."""
    pass
```

- [ ] **Step 2: 验证文件创建**

```bash
ls backend/app/services/retrieval_service.py
```

- [ ] **Step 3: 提交**

```bash
git add backend/app/services/retrieval_service.py
git commit -m "feat(retrieval): extract retrieval logic into standalone service"
```

---

## Task 7: 创建 chat_service.py

**Files:**
- Create: `backend/app/services/chat_service.py`

- [ ] **Step 1: 创建 chat_service.py**

从 `rag.py` 的 `/chat/{workspace_id}` 端点提取 chat 逻辑：
- Prompt 构建（CONTEXT → RULES → QUESTION 结构）
- 来源格式化（`Source [xxxx]` 格式）
- LLM 调用和响应处理
- 消息持久化

```python
"""
Chat Service
============
Unified chat logic: prompt building, LLM calls, response processing.
"""
from dataclasses import dataclass
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

@dataclass
class ChatResult:
    answer: str
    sources: list
    related_entities: list
    kg_summary: Optional[str]
    image_refs: list
    thinking: Optional[str]

async def chat(
    workspace_id: int,
    question: str,
    chunks: list,
    citations: list,
    images: list,
    db: AsyncSession,
) -> ChatResult:
    """Execute chat with documents as context."""
    pass

def build_user_message(
    question: str,
    context: str,
    image_context: list[str],
) -> str:
    """Build user message: CONTEXT → RULES → QUESTION structure."""
    pass
```

- [ ] **Step 2: 验证文件创建**

- [ ] **Step 3: 提交**

```bash
git add backend/app/services/chat_service.py
git commit -m "feat(chat): extract chat logic into standalone service"
```

---

## Task 8: 简化 rag.py chat 端点

**Files:**
- Modify: `backend/app/api/rag.py`

- [ ] **Step 1: 分析 chat 端点中需要保留的逻辑**

chat 端点 (~300行) 应该简化为：
1. 验证 workspace
2. 调用 retrieval_service 获取检索结果
3. 调用 chat_service 获取 chat 结果
4. 持久化消息

- [ ] **Step 2: 重写 /chat/{workspace_id} 端点**

使用新创建的 services，简化到 ~50 行

- [ ] **Step 3: 清理废弃的 import**

删除 `OLLAMA_ENABLE_THINKING` 等引用

- [ ] **Step 4: 验证**

```bash
cd backend
python -c "from app.api.rag import router; print('OK')"
```

- [ ] **Step 5: 提交**

```bash
git add backend/app/api/rag.py
git commit -m "refactor(rag): use chat_service and retrieval_service, remove 250+ lines"
```

---

## Task 9: 简化 chat_agent.py

**Files:**
- Modify: `backend/app/api/chat_agent.py`

- [ ] **Step 1: 分析 chat_agent.py 中的问题**

- 删除 `is_gemini` 判断分支
- 删除 `OLLAMA_TOOL_SYSTEM` 提示词
- 删除 `GEMINI_TOOL_SYSTEM` 提示词
- 只保留 OpenAI-compatible 的工具调用逻辑

- [ ] **Step 2: 简化 _get_openai_tool() 和工具调用循环**

- [ ] **Step 3: 删除 Gemini 特殊处理**

删除 `is_gemini` 判断，只保留 `is_openai`

- [ ] **Step 4: 验证**

```bash
cd backend
python -c "from app.api.chat_agent import chat_stream_endpoint; print('OK')"
```

- [ ] **Step 5: 提交**

```bash
git add backend/app/api/chat_agent.py
git commit -m "refactor(chat_agent): remove Gemini/Ollama handling, keep OpenAI-only"
```

---

## Task 10: 重写 .env.example

**Files:**
- Modify: `D:/llm_rag/NexusRAG/.env.example`

- [ ] **Step 1: 按分组重写 .env.example**

```bash
# ===========================================
# 数据库
# ===========================================
DATABASE_URL=postgresql+asyncpg://postgres:123456@localhost:5432/nexusrag

# ===========================================
# 向量存储 (Qdrant)
# ===========================================
QDRANT_URL=https://your-qdrant-cloud.com
QDRANT_API_KEY=your-qdrant-api-key

# ===========================================
# 嵌入服务 (向量检索用)
# ===========================================
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_OPENAI_API_KEY=sk-xxx
EMBEDDING_OPENAI_BASE_URL=https://api.chatanywhere.tech/v1

# ===========================================
# LLM (对话用)
# ===========================================
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
LLM_OPENAI_API_KEY=sk-xxx
LLM_OPENAI_BASE_URL=https://api.chatanywhere.tech/v1
LLM_MAX_TOKENS=8192

# ===========================================
# 重排序服务
# ===========================================
# Gitee AI
RERANKER_PROVIDER=gitee_ai
GITEE_AI_API_TOKEN=xxx
GITEE_AI_RERANK_MODEL=bge-reranker-v2-m3
GITEE_AI_RERANK_TOP_N=10

# SiliconFlow (备选)
# RERANKER_PROVIDER=siliconflow
# SILICONFLOW_API_KEY=sk-xxx
# SILICONFLOW_RERANK_MODEL=BAAI/bge-reranker-v2-m3
# SILICONFLOW_RERANK_TOP_N=10

# ===========================================
# 知识图谱
# ===========================================
KG_ENABLED=true
KG_LANGUAGE=Chinese
KG_ENTITY_TYPES=["Model","Dataset","Method","Task","Metric","Framework","Architecture","Layer","Optimizer","Loss","Organization","Person","Publication","Year","Hardware","Hyperparameter","Code","Technology"]
KG_OPENAI_API_KEY=sk-xxx

# ===========================================
# NexusRAG Pipeline
# ===========================================
NEXUSRAG_DOCLING_DEVICE=auto
NEXUSRAG_DOCLING_NUM_THREADS=4
NEXUSRAG_CHUNK_MAX_TOKENS=512
NEXUSRAG_VECTOR_PREFETCH=20
NEXUSRAG_RERANKER_TOP_K=8
NEXUSRAG_MIN_RELEVANCE_SCORE=0.15
NEXUSRAG_ENABLE_IMAGE_EXTRACTION=true
NEXUSRAG_ENABLE_TABLE_CAPTIONING=false
NEXUSRAG_ENABLE_FORMULA_ENRICHMENT=true

# ===========================================
# 评估 (Judge)
# ===========================================
JUDGE_PROVIDER=openai
JUDGE_OPENAI_API_KEY=sk-xxx
JUDGE_OPENAI_BASE_URL=https://api.openai.com/v1
JUDGE_OPENAI_MODEL=gpt-4o

# ===========================================
# CORS
# ===========================================
CORS_ORIGINS=["http://localhost:5174","http://localhost:3000"]
```

- [ ] **Step 2: 验证语法**

- [ ] **Step 3: 提交**

```bash
git add .env.example
git commit -m "refactor(env): regroup variables by function, remove deprecated vars"
```

---

## Task 11: 最终验证

**Files:**
- All modified files

- [ ] **Step 1: 验证后端可以启动**

```bash
cd D:/llm_rag/NexusRAG
source .venv/Scripts/activate
cd backend
python -c "
from app.main import app
from app.core.config import settings
from app.services.llm import get_llm_provider
from app.services.embedder import get_embedding_service
from app.services.reranker import get_reranker_service
print('All imports OK')
print(f'LLM_PROVIDER={settings.LLM_PROVIDER}')
print(f'EMBEDDING_PROVIDER={settings.EMBEDDING_PROVIDER}')
print(f'RERANKER_PROVIDER={settings.RERANKER_PROVIDER}')
"
```

- [ ] **Step 2: 运行测试（如果存在）**

```bash
cd backend
pytest tests/ -v --tb=short 2>&1 | head -50
```

- [ ] **Step 3: 检查无废弃变量引用**

```bash
grep -r "CHROMA_\|OLLAMA_\|GEMINI_\|GOOGLE_AI\|LLM_MODEL_FAST\|LLM_THINKING_LEVEL" backend/app/ --include="*.py" | grep -v "__pycache__"
```

预期: 无输出

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "refactor: complete cleanup - remove deprecated providers, unify config"
```

---

## 实施检查清单

完成所有 task 后，确认以下内容：

- [ ] `.env.example` 按功能分组
- [ ] `config.py` 无废弃变量
- [ ] `gemini.py`, `ollama.py`, `sentence_transformer.py`, `gitee_ai.py` 已删除
- [ ] `llm/__init__.py` 只支持 openai
- [ ] `embedder.py` 只使用 openai
- [ ] `rag.py` chat 端点已拆分
- [ ] `chat_agent.py` 无 Gemini/Ollama 判断
- [ ] `reranker.py` 只支持 gitee_ai 和 siliconflow
- [ ] 所有 import 测试通过
- [ ] 后端可以正常启动
