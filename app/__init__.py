"""RAG pipeline application package.

Exposes the main classes and configuration so consumers can do::

    from app import RAGApp, Ingestion
"""

__version__ = "0.1.0"

from .config import (
    BASE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    K,
    K_MIN,
    MAX_PER_QUERY,
    MAX_CONTEXT_DOCS,
    SCORE_DROP_THRESHOLD,
    MAX_CONTEXT_CHARS,
    BM25_WEIGHT,
    RRF_K,
    LLM_MODEL,
    EMBEDDING_MODEL,
    DOCS_PATH,
    INDEX_PATH,
    HOST,
    PORT,
    LOG_LEVEL,
)
from .vector_store import VectorStore
from .retriever import Retriever
from .ingestion import Ingestion, run_ingestion
from .rag_service import RAGService
from .rag_app import RAGApp
from .api import app, run_server

__all__ = [
    # Version
    "__version__",
    # Configuration
    "BASE_DIR",
    "CHUNK_SIZE",
    "CHUNK_OVERLAP",
    "K",
    "K_MIN",
    "MAX_PER_QUERY",
    "MAX_CONTEXT_DOCS",
    "SCORE_DROP_THRESHOLD",
    "MAX_CONTEXT_CHARS",
    "BM25_WEIGHT",
    "RRF_K",
    "LLM_MODEL",
    "EMBEDDING_MODEL",
    "DOCS_PATH",
    "INDEX_PATH",
    "HOST",
    "PORT",
    "LOG_LEVEL",
    # Classes
    "VectorStore",
    "Retriever",
    "Ingestion",
    "RAGService",
    "RAGApp",
    # Functions
    "run_ingestion",
    # API
    "app",
    "run_server",
]
