"""FastAPI integration tests using TestClient.

These tests mock all external services (Chroma, Groq) so they run entirely
without credentials or a built index — safe for GitHub Actions CI.
"""

from __future__ import annotations
import io
import pytest
from unittest.mock import MagicMock, patch


# ── Fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture()
def client():
    """Return a TestClient with all external deps mocked."""
    # Patch heavy external imports before importing api
    mock_vs = MagicMock()
    mock_vs.health_check.return_value = {"status": "healthy", "document_count": 10}
    mock_vs.get.return_value = {"documents": [], "metadatas": []}

    mock_rag = MagicMock()
    mock_rag.query.return_value = "Mocked answer"

    with patch("app.vector_store.Chroma"), \
         patch("app.vector_store.HuggingFaceEmbeddings"), \
         patch("app.api.RAGApp", return_value=mock_rag), \
         patch("app.api.VectorStore", return_value=mock_vs):

        from fastapi.testclient import TestClient
        from app.api import app
        yield TestClient(app)


# ── /health ────────────────────────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "vector_store" in data
        assert "rag_app_ready" in data


# ── /files ─────────────────────────────────────────────────────────────────
class TestFiles:
    def test_files_returns_list(self, client):
        r = client.get("/files")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ── /sessions ──────────────────────────────────────────────────────────────
class TestSessions:
    def test_create_session(self, client):
        r = client.post("/sessions")
        assert r.status_code == 200
        data = r.json()
        assert "session_id" in data
        assert isinstance(data["session_id"], str)

    def test_list_sessions(self, client):
        client.post("/sessions")
        r = client.get("/sessions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_delete_session(self, client):
        sid = client.post("/sessions").json()["session_id"]
        r = client.delete(f"/sessions/{sid}")
        assert r.status_code == 200
        assert "cleared" in r.json()["message"]

    def test_delete_nonexistent_session(self, client):
        r = client.delete("/sessions/does-not-exist")
        assert r.status_code == 404


# ── /upload ────────────────────────────────────────────────────────────────
class TestUpload:
    def test_upload_invalid_extension(self, client):
        fake = io.BytesIO(b"hello")
        r = client.post("/upload", files={"file": ("notes.docx", fake, "application/octet-stream")})
        assert r.status_code == 400
        assert "Unsupported file type" in r.json()["detail"]

    def test_upload_valid_txt(self, client, tmp_path, monkeypatch):
        # Patch DOCS_PATH to a temp dir so we don't write to real docs/
        monkeypatch.setattr("app.api.DOCS_PATH", str(tmp_path))
        monkeypatch.setattr("app.api.Ingestion", MagicMock(return_value=MagicMock(
            load_documents=MagicMock(return_value=[MagicMock()]),
            create_chunks=MagicMock(return_value=[MagicMock(), MagicMock()]),
            data_ingest=MagicMock(),
        )))
        fake = io.BytesIO(b"Sample document content for testing.")
        r = client.post("/upload", files={"file": ("sample.txt", fake, "text/plain")})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert data["chunks_created"] == 2


# ── /query ─────────────────────────────────────────────────────────────────
class TestQuery:
    def test_query_returns_answer(self, client):
        r = client.post("/query", json={"question": "What is Google?"})
        assert r.status_code == 200
        assert "answer" in r.json()
        assert r.json()["answer"] == "Mocked answer"

    def test_query_empty_question(self, client):
        r = client.post("/query", json={"question": ""})
        assert r.status_code == 422  # Pydantic validation error
