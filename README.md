# HybridRAG — Intelligent Document Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-1.2-1C3C3C?style=flat&logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/ChromaDB-1.5-F97316?style=flat" />
  <img src="https://img.shields.io/badge/Groq-Llama_3.1-8B5CF6?style=flat" />
  <img src="https://img.shields.io/badge/License-MIT-22C55E?style=flat" />
  <img src="https://github.com/bhavyasik/RAG-project/actions/workflows/ci.yml/badge.svg" />
</p>

<p align="center">
  A production-ready Retrieval-Augmented Generation system with <strong>hybrid BM25 + vector search</strong>,
  <strong>real-time streaming responses</strong>, <strong>multi-session memory</strong>, and a
  premium dark-mode Web UI — all served from a single FastAPI server.
</p>

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Hybrid Retrieval** | Dense vector search (ChromaDB) + sparse BM25 keyword search run in parallel |
| **Reciprocal Rank Fusion** | RRF algorithm intelligently merges both ranked lists into a single result |
| **Query Decomposition** | LLM automatically splits multi-entity questions into independent sub-queries |
| **Score-Drop Filtering** | Adaptive thresholding keeps only relevant context, reducing hallucinations |
| **Streaming Responses** | Answers stream token-by-token over Server-Sent Events (SSE) |
| **Session Memory** | Each chat session retains the last 6 Q&A turns for context-aware follow-ups |
| **Document Upload** | Upload PDF or TXT files via the UI — indexed instantly without a restart |
| **Web UI** | Premium dark-mode single-page app served directly at `http://localhost:8000` |
| **Docker Ready** | Multi-stage Dockerfile for lightweight, production-grade container images |

---

## 🚀 Quickstart (60 seconds)

```bash
# 1. Clone the repo
git clone https://github.com/bhavyasik/RAG-project.git
cd RAG-project

# 2. Set your Groq API key
cp .env.example .env
# Edit .env → add your GROQ_API_KEY (free at console.groq.com)

# 3a. Run with Docker (recommended)
docker-compose up

# 3b. Or run locally with uv
uv pip install -e .
python -m app.api
```

Open **http://localhost:8000** in your browser.  
Upload a document, start a chat, and ask questions.

---

## 🏗 Architecture

```text
User Query (via Web UI or API)
    │
    ▼
Query Decomposition ──── LLM decides if the question targets multiple
    │                    independent entities and splits accordingly
    ▼
Hybrid Retrieval ──┬──── Dense Vector Search  (ChromaDB + HuggingFace embeddings)
                   └──── Sparse BM25 Search   (Okapi BM25 on full corpus)
    │
    ▼
Reciprocal Rank Fusion  (w_vec=0.6 / (k+rank) + w_bm25=0.4 / (k+rank))
    │
    ▼
Score-Drop Filtering    (always keep K_MIN=3, drop at 50% of best score)
    │
    ▼
Context Builder         (≤6,000 chars, hard char budget)
    │
    ▼
LLM Answer Generation   (Groq API — Llama 3.1 8B Instant, streamed)
    │
    ▼
Session Memory Update   (last 6 Q&A pairs, injected into next prompt)
```

---

## 📡 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve the Web UI |
| `/health` | GET | Service health check |
| `/files` | GET | List indexed documents |
| `/upload` | POST | Upload a PDF or TXT file |
| `/ingest` | POST | Manually trigger re-indexing |
| `/sessions` | POST | Create a new chat session |
| `/sessions` | GET | List all active sessions |
| `/sessions/{id}` | DELETE | Clear a session's memory |
| `/query` | POST | Batch query (returns full answer) |
| `/query-stream` | POST | Streaming query via SSE |
| `/docs` | GET | Interactive OpenAPI documentation |

### Example — streaming query

```bash
# 1. Create a session
SESSION=$(curl -s -X POST http://localhost:8000/sessions | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# 2. Stream an answer
curl -N -X POST http://localhost:8000/query-stream \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"Who founded Google?\", \"session_id\": \"$SESSION\"}"
```

---

## ⚙️ Configuration

All settings are controlled via environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Get one free at [console.groq.com](https://console.groq.com) |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model for answer generation |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE` | `800` | Characters per document chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks |
| `K` | `20` | Candidates retrieved per search method |
| `BM25_WEIGHT` | `0.4` | BM25 weight in RRF fusion (vector = 1 - this) |
| `RRF_K` | `60` | RRF smoothing constant |
| `K_MIN` | `3` | Minimum chunks always returned |
| `MAX_CONTEXT_DOCS` | `10` | Maximum chunks passed to the LLM |
| `SCORE_DROP_THRESHOLD` | `0.5` | Relative score drop-off threshold |
| `MAX_CONTEXT_CHARS` | `6000` | Hard context character budget |
| `PORT` | `8000` | Server port |

---

## 📁 Project Structure

```
rag-project/
├── app/
│   ├── api.py            # FastAPI server — all endpoints
│   ├── config.py         # Configuration (env vars + defaults)
│   ├── ingestion.py      # Document loading & chunking
│   ├── vector_store.py   # ChromaDB operations
│   ├── retriever.py      # Hybrid BM25 + vector search + RRF fusion
│   ├── rag_service.py    # LLM interaction (batch + streaming)
│   ├── rag_app.py        # RAG orchestration pipeline
│   ├── memory.py         # In-memory session store
│   ├── main.py           # Interactive CLI entry point
│   └── static/
│       └── index.html    # Single-page Web UI
├── tests/
│   ├── test_retriever.py # Unit tests — RRF fusion math
│   ├── test_memory.py    # Unit tests — session memory
│   └── test_api.py       # Integration tests — all endpoints
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI
├── Dockerfile            # Multi-stage production build
├── docker-compose.yml    # Compose for API + ingestion profiles
├── pyproject.toml        # Project metadata & dependencies (uv)
└── .env.example          # Environment variable template
```

---

## 🐳 Docker Deployment

```bash
# Build and start the API server
docker-compose up -d rag-api

# Run document ingestion once
docker-compose --profile ingest up rag-ingest

# View logs
docker-compose logs -f rag-api

# Or build manually
docker build -t hybrid-rag .
docker run -d -p 8000:8000 \
  -e GROQ_API_KEY=your_key \
  -v rag-index:/app/index \
  hybrid-rag
```

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run test suite
pytest tests/ -v
```

Tests run without any API keys or a built vector index — everything external is mocked.

---

## 🔧 Technical Implementation Details

The hybrid retrieval pipeline works as follows:

1. **Query Decomposition** — A strict LLM prompt determines if the question requires parallel sub-queries (e.g. "Google and Microsoft revenue" → two queries). Uses `ast.literal_eval` for safe parsing.

2. **Dense Search (ChromaDB)** — HuggingFace `all-MiniLM-L6-v2` embeddings. Chroma distance is converted to similarity: `score = 1 / (1 + distance)`.

3. **Sparse Search (BM25)** — Okapi BM25 on the full corpus. Regex tokenizer (`[a-z0-9]+`) strips punctuation. Index is built once and cached in-memory.

4. **RRF Fusion** — `score = w_vec/(k+rank_vec) + w_bm25/(k+rank_bm25)`. Default: `w_vec=0.6`, `w_bm25=0.4`, `k=60`.

5. **Score-Drop Filtering** — Always keeps top `K_MIN=3`. Adds chunks while `score ≥ best×0.5`, up to `MAX_CONTEXT_DOCS=10`.

6. **Streaming** — LangChain's `llm.stream()` yields token chunks over SSE. Session memory (last 6 turns) is injected into the prompt before each query.

---

## 📄 License

MIT © [Bhavyadeep Singh](https://github.com/bhavyasik)
