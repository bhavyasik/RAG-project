"""Retrieval module — hybrid BM25 + vector search with RRF fusion."""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .config import K, INDEX_PATH, EMBEDDING_MODEL, BM25_WEIGHT, RRF_K
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── Tokenizer ──────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase regex tokenizer — strips punctuation, splits on word
    boundaries.  Much better than naive str.split() for BM25."""
    return _TOKEN_RE.findall(text.lower())


# ── Retriever ──────────────────────────────────────────────────
class Retriever:
    """Hybrid retriever: dense vector search + sparse BM25, fused via
    Reciprocal Rank Fusion (RRF).

    **Score convention: higher = more relevant** throughout.
    """

    def __init__(
        self,
        k: int = K,
        index_path: str = INDEX_PATH,
        embedding_model: str = EMBEDDING_MODEL,
        bm25_weight: float = BM25_WEIGHT,
        rrf_k: int = RRF_K,
    ) -> None:
        self.k = k
        self.bm25_weight = bm25_weight
        self.rrf_k = rrf_k
        self.vector_store = VectorStore(
            index_path=index_path,
            embedding_model=embedding_model,
        )
        # Lazily built BM25 corpus — cached across queries
        self._bm25_docs: list[Document] | None = None
        self._bm25_index: BM25Okapi | None = None

    # ── Public API ─────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Tuple[Document, float]]:
        """Run hybrid retrieval and return fused results.

        Returns
        -------
        list[tuple[Document, float]]
            ``(document, rrf_score)`` sorted **descending** (higher = better).
        """
        vector_results = self._vector_search(query)
        bm25_results = self._bm25_search(query)
        fused = self._rrf_fuse(vector_results, bm25_results)

        logger.info(
            "Hybrid — vector: %d | BM25: %d | fused: %d",
            len(vector_results),
            len(bm25_results),
            len(fused),
        )
        return fused

    # ── Dense retrieval ────────────────────────────────────────

    def _vector_search(self, query: str) -> List[Tuple[Document, float]]:
        """Chroma similarity search.

        Chroma returns (doc, distance) where lower distance = better.
        Convert to similarity score: similarity = 1 / (1 + distance)
        so that higher = better (consistent with BM25).
        """
        vs = self.vector_store.load()
        results = vs.similarity_search_with_score(query, k=self.k)
        # Convert distance to similarity score (higher = better)
        return [(doc, 1.0 / (1.0 + dist)) for doc, dist in results]

    # ── Sparse retrieval ───────────────────────────────────────

    def _ensure_bm25_index(self) -> None:
        """Build the BM25 index once and cache it for the session."""
        if self._bm25_index is not None:
            return

        vs = self.vector_store.load()
        collection = vs.get()

        if not collection["documents"]:
            self._bm25_docs = []
            return

        self._bm25_docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(
                collection["documents"], collection["metadatas"]
            )
        ]

        tokenized = [_tokenize(d.page_content) for d in self._bm25_docs]
        self._bm25_index = BM25Okapi(tokenized)
        logger.info("BM25 index built — %d documents.", len(self._bm25_docs))

    def _bm25_search(self, query: str) -> List[Tuple[Document, float]]:
        """BM25 keyword search.  Returns ``(doc, bm25_score)`` sorted
        descending (higher = more relevant).  Zero-score docs are dropped."""
        self._ensure_bm25_index()

        if not self._bm25_docs or self._bm25_index is None:
            return []

        scores = self._bm25_index.get_scores(_tokenize(query))

        scored = [
            (doc, float(s))
            for doc, s in zip(self._bm25_docs, scores)
            if s > 0
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[: self.k]

    # ── Fusion ─────────────────────────────────────────────────

    def _rrf_fuse(
        self,
        vector_results: List[Tuple[Document, float]],
        bm25_results: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion of two ranked lists.

        RRF score = w_vec / (k + rank_vec)  +  w_bm25 / (k + rank_bm25)

        Both input lists are assumed pre-sorted (rank 1 = best).
        Output is sorted descending by fused score (**higher = better**).
        """
        w_vec = 1.0 - self.bm25_weight
        w_bm25 = self.bm25_weight
        k = self.rrf_k

        fused: dict[str, tuple[Document, float]] = {}

        # Vector contribution (rank 1 = lowest Chroma distance = best)
        for rank, (doc, _) in enumerate(vector_results, start=1):
            key = doc.page_content
            rrf = w_vec / (k + rank)
            prev = fused.get(key)
            fused[key] = (doc, (prev[1] if prev else 0.0) + rrf)

        # BM25 contribution (already sorted descending)
        for rank, (doc, _) in enumerate(bm25_results, start=1):
            key = doc.page_content
            rrf = w_bm25 / (k + rank)
            prev = fused.get(key)
            fused[key] = (doc, (prev[1] if prev else 0.0) + rrf)

        ranked = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        return ranked[: self.k]