"""Unit tests for RRF fusion math in app.retriever.

These tests are self-contained — no Chroma index, no embeddings, no API keys.
We extract and test the _rrf_fuse logic by instantiating it with mocked deps.
"""

from __future__ import annotations
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


# ── Helpers ────────────────────────────────────────────────────────────────
def make_doc(content: str, source: str = "test.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


def make_retriever():
    """Return a Retriever with all external deps mocked out."""
    with patch("app.retriever.VectorStore"):
        from app.retriever import Retriever
        r = Retriever.__new__(Retriever)
        r.k = 20
        r.bm25_weight = 0.4
        r.rrf_k = 60
        r.vector_store = MagicMock()
        r._bm25_docs = None
        r._bm25_index = None
        return r


# ── Tests ──────────────────────────────────────────────────────────────────
class TestRRFFusion:
    """Tests for Retriever._rrf_fuse()"""

    def test_only_vector_results(self):
        r = make_retriever()
        docs = [make_doc(f"chunk {i}") for i in range(3)]
        vec = [(d, 1.0) for d in docs]
        fused = r._rrf_fuse(vec, [])
        # All scores should be positive and sum to > 0
        assert len(fused) == 3
        for _, score in fused:
            assert score > 0

    def test_only_bm25_results(self):
        r = make_retriever()
        docs = [make_doc(f"chunk {i}") for i in range(3)]
        bm25 = [(d, 2.0) for d in docs]
        fused = r._rrf_fuse([], bm25)
        assert len(fused) == 3

    def test_merged_scores_higher_than_single_source(self):
        """A doc in both lists should outscore one only in a single list."""
        r = make_retriever()
        shared_doc = make_doc("shared content")
        only_vec   = make_doc("only in vector")

        vec  = [(shared_doc, 1.0), (only_vec, 0.9)]
        bm25 = [(shared_doc, 5.0)]

        fused = r._rrf_fuse(vec, bm25)
        scores = {doc.page_content: score for doc, score in fused}

        assert scores["shared content"] > scores["only in vector"]

    def test_deduplication(self):
        """The same document content should appear only once in output."""
        r = make_retriever()
        doc = make_doc("duplicate content")
        vec  = [(doc, 1.0)]
        bm25 = [(doc, 3.0)]
        fused = r._rrf_fuse(vec, bm25)
        contents = [d.page_content for d, _ in fused]
        assert contents.count("duplicate content") == 1

    def test_output_sorted_descending(self):
        r = make_retriever()
        docs = [make_doc(f"doc {i}") for i in range(5)]
        vec  = [(d, 1.0) for d in docs]
        bm25 = [(d, 1.0) for d in reversed(docs)]
        fused = r._rrf_fuse(vec, bm25)
        scores = [s for _, s in fused]
        assert scores == sorted(scores, reverse=True)

    def test_weights_applied(self):
        """BM25 weight=0.4 → vector contribution (0.6) should dominate."""
        r = make_retriever()
        vec_only  = make_doc("vector only")
        bm25_only = make_doc("bm25 only")

        # vec_only is rank-1 in vector; bm25_only is rank-1 in BM25
        vec  = [(vec_only,  1.0)]
        bm25 = [(bm25_only, 5.0)]

        fused = r._rrf_fuse(vec, bm25)
        scores = {d.page_content: s for d, s in fused}

        # vector weight (0.6) > bm25 weight (0.4) at same rank
        assert scores["vector only"] > scores["bm25 only"]

    def test_empty_inputs(self):
        r = make_retriever()
        assert r._rrf_fuse([], []) == []

    def test_rrf_k_affects_scores(self):
        """A lower rrf_k means rank difference matters more."""
        from app.retriever import Retriever
        r_low  = make_retriever(); r_low.rrf_k  = 1
        r_high = make_retriever(); r_high.rrf_k = 1000

        doc1 = make_doc("rank1"); doc2 = make_doc("rank2")
        vec = [(doc1, 1.0), (doc2, 0.5)]

        fused_low  = {d.page_content: s for d, s in r_low._rrf_fuse(vec, [])}
        fused_high = {d.page_content: s for d, s in r_high._rrf_fuse(vec, [])}

        # Gap should be larger with low k
        gap_low  = fused_low["rank1"]  - fused_low["rank2"]
        gap_high = fused_high["rank1"] - fused_high["rank2"]
        assert gap_low > gap_high
