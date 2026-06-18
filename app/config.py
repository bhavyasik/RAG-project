"""Configuration constants for the RAG pipeline.

All settings can be overridden via environment variables.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = os.getenv("DOCS_PATH", str(BASE_DIR / "docs"))
INDEX_PATH = os.getenv("INDEX_PATH", str(BASE_DIR / "index"))

# ── Ingestion ──────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Models ─────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")

# ── Retrieval (per search method) ─────────────────────────────
K = int(os.getenv("K", "20"))           # Candidates per search method
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))  # BM25 weight in RRF
RRF_K = int(os.getenv("RRF_K", "60"))   # RRF smoothing constant

# ── Filtering ─────────────────────────────────────────────────
K_MIN = int(os.getenv("K_MIN", "3"))          # Min chunks returned
MAX_PER_QUERY = int(os.getenv("MAX_PER_QUERY", "8"))  # Cap per sub-query
MAX_CONTEXT_DOCS = int(os.getenv("MAX_CONTEXT_DOCS", "10"))  # Max after filter
SCORE_DROP_THRESHOLD = float(
    os.getenv("SCORE_DROP_THRESHOLD", "0.5")
)  # Relative score threshold
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))  # Context limit

# ── Server ────────────────────────────────────────────────────
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")