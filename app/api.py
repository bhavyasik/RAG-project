"""FastAPI server for RAG application.

New in this version
-------------------
- Serves the Web UI (app/static/index.html) at the root ``/``.
- ``POST /upload``        — upload a PDF or TXT file, triggers re-ingestion.
- ``GET  /files``         — list files in the knowledge base docs directory.
- ``GET  /sessions``      — list active in-memory chat sessions.
- ``DELETE /sessions/{id}`` — clear a session's history.
- ``POST /query-stream``  — streaming SSE endpoint with memory.
"""

import logging
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import HOST, PORT, LOG_LEVEL, INDEX_PATH, DOCS_PATH
from .ingestion import Ingestion
from .memory import (
    new_session,
    get_history,
    get_history_string,
    append_turn,
    list_sessions,
    clear_session,
    session_exists,
)
from .rag_app import RAGApp
from .vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Allowed upload extensions
ALLOWED_EXTENSIONS = {".pdf", ".txt"}

# Path to the static UI directory
STATIC_DIR = Path(__file__).resolve().parent / "static"


# ── Global app state ───────────────────────────────────────────────────────
class AppState:
    """Holds global state for the FastAPI app."""

    rag_app: Optional[RAGApp] = None
    vector_store: Optional[VectorStore] = None


app_state = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise and cleanup application state."""
    logger.info("Starting RAG API server...")

    # Ensure docs directory exists
    Path(DOCS_PATH).mkdir(parents=True, exist_ok=True)

    vector_store = VectorStore()
    health = vector_store.health_check()
    logger.info("Vector store health: %s", health.get("status", "unknown"))

    if health.get("status") == "healthy":
        app_state.rag_app = RAGApp()
        logger.info("RAG application initialised")
    else:
        logger.warning(
            "Vector store not ready (%s). Use /upload or /ingest first.",
            health.get("error", "unknown"),
        )

    app_state.vector_store = vector_store
    yield

    logger.info("Shutting down RAG API server...")


# ── FastAPI application ────────────────────────────────────────────────────
app = FastAPI(
    title="HybridRAG API",
    description=(
        "Retrieval-Augmented Generation with hybrid BM25 + vector search, "
        "streaming responses, and multi-session memory."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# Mount static files (JS, CSS assets if any)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request / Response models ──────────────────────────────────────────────
class QueryRequest(BaseModel):
    """Request model for the batch query endpoint."""

    question: str = Field(..., min_length=1, description="The question to ask")


class QueryResponse(BaseModel):
    """Response model for the batch query endpoint."""

    answer: str = Field(..., description="The generated answer")
    sources: Optional[List[str]] = Field(
        default=None, description="Source documents used"
    )


class StreamQueryRequest(BaseModel):
    """Request model for the streaming query endpoint."""

    question: str = Field(..., min_length=1, description="The question to ask")
    session_id: str = Field(..., description="Session ID for conversation memory")


class IngestRequest(BaseModel):
    """Request model for the ingest endpoint."""

    force: bool = Field(
        default=False, description="Force re-ingestion even if index exists"
    )


class IngestResponse(BaseModel):
    """Response model for the ingest endpoint."""

    status: str
    documents_loaded: int
    chunks_created: int
    message: str


class HealthResponse(BaseModel):
    """Response model for the health endpoint."""

    status: str
    vector_store: dict
    rag_app_ready: bool


class FileInfo(BaseModel):
    """Metadata for a single file in the knowledge base."""

    name: str
    size_bytes: int
    extension: str


class SessionInfo(BaseModel):
    """Info about an active chat session."""

    session_id: str
    turn_count: int


# ── UI Endpoint ────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_ui():
    """Serve the single-page Web UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Web UI not found. Ensure app/static/index.html exists.",
        )
    return FileResponse(str(index_file))


# ── Health ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check the health status of the RAG application."""
    vs_health = (
        app_state.vector_store.health_check()
        if app_state.vector_store
        else {"status": "uninitialized", "error": "Vector store not initialized"}
    )
    return HealthResponse(
        status=(
            "healthy"
            if vs_health.get("status") == "healthy" and app_state.rag_app
            else "unhealthy"
        ),
        vector_store=vs_health,
        rag_app_ready=app_state.rag_app is not None,
    )


# ── Knowledge base file listing ────────────────────────────────────────────
@app.get("/files", response_model=List[FileInfo])
async def list_files():
    """List all files currently in the knowledge base docs directory."""
    docs_path = Path(DOCS_PATH)
    if not docs_path.exists():
        return []
    files = []
    for f in sorted(docs_path.iterdir()):
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            files.append(
                FileInfo(
                    name=f.name,
                    size_bytes=f.stat().st_size,
                    extension=f.suffix.lower(),
                )
            )
    return files


# ── Document upload ────────────────────────────────────────────────────────
@app.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or TXT file and re-index the knowledge base."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Only PDF and TXT are allowed.",
        )

    docs_path = Path(DOCS_PATH)
    docs_path.mkdir(parents=True, exist_ok=True)
    dest = docs_path / file.filename

    # Save uploaded file
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        logger.info("Uploaded file saved: %s", dest)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {e}",
        )

    # Re-run full ingestion to rebuild both Chroma + BM25 indices
    try:
        ingestion = Ingestion()
        documents = ingestion.load_documents()
        chunks = ingestion.create_chunks(documents)
        ingestion.data_ingest(chunks)

        # Invalidate BM25 cache and reinitialise RAG app
        app_state.rag_app = RAGApp()

        logger.info(
            "Re-ingestion complete — %d docs / %d chunks", len(documents), len(chunks)
        )
        return IngestResponse(
            status="success",
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            message=f"'{file.filename}' uploaded and indexed successfully.",
        )
    except Exception as e:
        logger.exception("Re-ingestion after upload failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Indexing failed: {e}",
        )


# ── Ingest (manual trigger) ────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents from the docs directory into the vector store."""
    try:
        vs_health = app_state.vector_store.health_check()
        if vs_health.get("status") == "healthy" and not request.force:
            return IngestResponse(
                status="skipped",
                documents_loaded=0,
                chunks_created=0,
                message="Index already exists. Use force=true to re-ingest.",
            )

        ingestion = Ingestion()
        documents = ingestion.load_documents()
        chunks = ingestion.create_chunks(documents)
        ingestion.data_ingest(chunks)
        app_state.rag_app = RAGApp()

        return IngestResponse(
            status="success",
            documents_loaded=len(documents),
            chunks_created=len(chunks),
            message="Documents ingested successfully.",
        )
    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {e}",
        )


# ── Session management ─────────────────────────────────────────────────────
@app.post("/sessions", response_model=dict)
async def create_session():
    """Create a new chat session and return its ID."""
    sid = new_session()
    return {"session_id": sid}


@app.get("/sessions", response_model=List[SessionInfo])
async def get_sessions():
    """List all active chat sessions."""
    return [
        SessionInfo(session_id=sid, turn_count=len(get_history(sid)) // 2)
        for sid in list_sessions()
    ]


@app.delete("/sessions/{session_id}", response_model=dict)
async def delete_session(session_id: str):
    """Clear all history for the given session."""
    if not session_exists(session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


# ── Streaming query (Web UI) ───────────────────────────────────────────────
@app.post("/query-stream")
async def query_stream(request: StreamQueryRequest):
    """Stream an answer using Server-Sent Events with conversation memory."""
    if not app_state.rag_app:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG application not initialised. Upload documents and run /ingest first.",
        )

    if not session_exists(request.session_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{request.session_id}' not found. Create one via POST /sessions.",
        )

    history = get_history_string(request.session_id)

    # Collect the full answer in a mutable container so the generator closure
    # can write to it and we can persist it to memory after streaming.
    accumulated: list[str] = []

    def event_stream():
        for line in app_state.rag_app.query_stream(request.question, history):
            # Collect token content for memory persistence
            try:
                import json as _json
                parsed = _json.loads(line.removeprefix("data: ").strip())
                if parsed.get("event") == "token":
                    accumulated.append(parsed["data"])
                elif parsed.get("event") == "done":
                    # Persist to memory once streaming is complete
                    full_answer = "".join(accumulated)
                    if full_answer:
                        append_turn(
                            request.session_id, request.question, full_answer
                        )
            except Exception:
                pass  # Parsing errors don't affect the stream
            yield line

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── Batch query (CLI / Swagger) ────────────────────────────────────────────
@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Ask a question and get a full answer (non-streaming)."""
    if not app_state.rag_app:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG application not initialised. Run /ingest first.",
        )
    try:
        answer = app_state.rag_app.query(request.question)
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {e}",
        )


# ── Server entry point ─────────────────────────────────────────────────────
def run_server():
    """Run the FastAPI server using uvicorn."""
    import uvicorn

    logger.info("Starting server on %s:%s", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
