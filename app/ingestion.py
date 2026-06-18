"""Document ingestion pipeline — load, chunk, and index documents."""

import logging

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DOCS_PATH, EMBEDDING_MODEL, CHUNK_OVERLAP, CHUNK_SIZE, INDEX_PATH
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


def run_ingestion() -> None:
    """Convenience function to run full ingestion pipeline."""
    logger.info("Starting document ingestion...")
    pipeline = Ingestion()
    documents = pipeline.load_documents()
    logger.info("Loaded %d document pages.", len(documents))
    chunks = pipeline.create_chunks(documents)
    logger.info("Created %d chunks.", len(chunks))
    pipeline.data_ingest(chunks)
    logger.info("Vector store saved.")


class Ingestion:
    """Load documents from disk, split into chunks, and index them."""

    def __init__(
        self,
        docs_path: str = DOCS_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        index_path: str = INDEX_PATH,
    ) -> None:
        self.docs_path = docs_path
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.index_path = index_path

    def load_documents(self):
        """Load all PDF and TXT files from the docs directory."""
        pdf_loader = DirectoryLoader(
            self.docs_path,
            glob="*.pdf",
            loader_cls=PyPDFLoader,
        )
        txt_loader = DirectoryLoader(
            self.docs_path,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"autodetect_encoding": True},
        )
        documents = pdf_loader.load() + txt_loader.load()
        logger.info(
            "Loaded %d document pages from %s.", len(documents), self.docs_path
        )
        return documents

    def create_chunks(self, documents):
        """Split documents into smaller chunks for embedding."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(
            "Created %d chunks (size=%d, overlap=%d).",
            len(chunks),
            self.chunk_size,
            self.chunk_overlap,
        )
        return chunks

    def data_ingest(self, chunks):
        """Create a vector store from chunks and persist it to disk."""
        vs = VectorStore(
            index_path=self.index_path,
            embedding_model=self.embedding_model,
        )
        vs.create_vector_store(chunks)
        logger.info("Vector store persisted to %s.", self.index_path)