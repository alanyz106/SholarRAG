# ScholarRAG

A knowledge base Q&A system powered by RAG (Retrieval-Augmented Generation), supporting semantic search, knowledge graph, and multi-format document processing.

## Features

- **Multi-format Document Support** — PDF, DOCX, PPTX, Excel, and more via Docling parser
- **Hybrid Retrieval** — Combines vector search (ChromaDB) with knowledge graph (LightRAG) for accurate answers
- **Knowledge Graph** — Entity extraction and relationship mapping for deeper context understanding
- **LLM Chat** — Supports Gemini API and local Ollama models
- **Reranking** — Cross-encoder reranking for improved relevance ranking
- **Workspace Management** — Multi-knowledge-base isolation with independent chat histories

## Tech Stack

| Component | Technology |
|-----------|------------|
| Backend | FastAPI + Python 3.11+ |
| Frontend | React 19 + TypeScript + Vite |
| Database | PostgreSQL |
| Vector DB | ChromaDB |
| KG Engine | LightRAG |
| Embedding | sentence-transformers / OpenAI |
| LLM | Gemini / Ollama |

## Quick Start

### Prerequisites

- Python 3.11+, Node.js 22 LTS
- PostgreSQL 15 (port 5433)
- ChromaDB server running (port 8002)

### Backend

```bash
# Activate virtual environment
source .venv/Scripts/activate

# Install dependencies
pip install -r backend/requirements.txt

# Download embedding models (~2.5GB)
python backend/scripts/download_models.py

# Start backend
cd backend
uvicorn app.main:app --reload --port 8080
```

### Frontend

```bash
pnpm install
pnpm dev
```

Visit `http://localhost:5174` to start using ScholarRAG.

## Configuration

Copy `.env.example` to `.env` and configure:

```bash
# LLM Provider (gemini or ollama)
LLM_PROVIDER=gemini
GOOGLE_AI_API_KEY=your_api_key

# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/nexusrag
```

Full configuration options are documented in `.env.example`.

## Project Structure

```
ScholarRAG/
├── backend/
│   ├── app/
│   │   ├── api/          # REST API endpoints
│   │   ├── core/          # Config, database, dependencies
│   │   ├── models/       # SQLAlchemy models
│   │   ├── schemas/      # Pydantic schemas
│   │   └── services/     # Business logic (RAG, embedding, KG)
│   └── scripts/           # Utility scripts
├── frontend/              # React + TypeScript frontend
└── tutorial/             # Learning materials
```

## License

MIT
