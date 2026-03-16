# Value Stream RAG

**Enterprise RAG System for Value Stream Recommendation from Idea-Card PPTs**

> Given an uploaded PowerPoint idea-card, this system recommends the most relevant Value Streams using Azure AI Search, historical idea cards, and a LangGraph orchestration pipeline.

---

## Quick Start

```bash
# 1. Clone & install
git clone <repo-url> && cd value-stream-rag
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your Azure credentials

# 3. Ingest historical PPTs
python scripts/ingest_historical.py --dir data/historical_ppts --manifest data/manifest.json

# 4. Start API
uvicorn src.api.main:app --reload --port 8000

# 5. Upload an idea card
curl -X POST http://localhost:8000/api/v1/recommendations/upload \
  -F "file=@idea_card.pptx"
```

---

## Documentation

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the complete design documentation including:

- Executive Summary
- Problem Framing
- Architecture Overview (with Mermaid diagrams)
- Data Model Design
- Chunking Strategy
- LangGraph Workflow Design
- API Design
- Evaluation Framework
- Production Considerations

---

## Project Structure

```
value-stream-rag/
├── src/
│   ├── api/              # FastAPI application & routers
│   ├── config/           # Settings & logging
│   ├── models/           # Domain models & ORM
│   ├── ingestion/        # PPT parsing (MarkItDown) & pipeline
│   ├── chunking/         # Chunking strategies
│   ├── embeddings/       # Embedding service
│   ├── search/           # Azure AI Search & ChromaDB
│   ├── ranking/          # Scoring & ranking
│   ├── graph/            # LangGraph nodes & workflows
│   ├── services/         # Service layer & DI container
│   └── evaluation/       # Offline evaluation framework
├── tests/
│   ├── unit/             # Unit tests (no external deps)
│   ├── integration/      # Integration tests (local services)
│   └── evaluation/       # Offline eval scripts
├── data/
│   ├── historical_ppts/  # Historical idea card PPTs
│   ├── value_streams/    # VS reference data
│   ├── uploads/          # Runtime PPT uploads
│   └── indexes/          # ChromaDB persistence
├── docs/                 # Architecture docs & diagrams
└── scripts/              # CLI scripts
```

---

## Running Tests

```bash
# Unit tests only (no external services)
pytest tests/unit/ -v

# Integration tests (requires python-pptx, local ChromaDB)
pytest tests/integration/ -v -m integration

# All tests with coverage
pytest --cov=src --cov-report=html
```
