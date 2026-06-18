"""Vector store operations using Chroma and HuggingFace embeddings."""

import logging
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from .config import INDEX_PATH, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages Chroma vector store creation, saving, and loading."""

    def __init__(
        self,
        index_path: str = INDEX_PATH,
        embedding_model: str = EMBEDDING_MODEL,
    ) -> None:
        self.index_path = index_path
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model
        )

    def create_vector_store(self, chunks) -> Chroma:
        """Build a Chroma index from document chunks."""
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.index_path,
        )
        logger.info("Chroma index created with %d chunks.", len(chunks))
        return vector_store

    def load(self) -> Chroma:
        """Load the persisted Chroma index.

        Raises
        ------
        FileNotFoundError
            If the index directory does not exist.
        """
        if not Path(self.index_path).exists():
            raise FileNotFoundError(
                f"Vector store index not found at {self.index_path}. "
                "Run ingestion first to create the index."
            )
        return Chroma(
            persist_directory=self.index_path,
            embedding_function=self.embedding_model,
        )

    def health_check(self) -> dict:
        """Check if the vector store is healthy and has documents.

        Returns
        -------
        dict
            Health status with document count and index path.
        """
        try:
            vs = self.load()
            collection = vs.get()
            doc_count = len(collection.get("documents", []))
            return {
                "status": "healthy" if doc_count > 0 else "empty",
                "document_count": doc_count,
                "index_path": self.index_path,
            }
        except FileNotFoundError as e:
            return {"status": "unhealthy", "error": str(e)}
        except Exception as e:
            logger.error("Vector store health check failed: %s", e)
            return {"status": "unhealthy", "error": str(e)}