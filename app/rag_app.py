"""RAG orchestration — query decomposition, retrieval, filtering, generation."""

import logging
import json
import time
from typing import Iterator, List, Tuple

from langchain_core.documents import Document
from .rag_service import RAGService
from .retriever import Retriever
from .config import (
    K_MIN,
    MAX_PER_QUERY,
    MAX_CONTEXT_DOCS,
    SCORE_DROP_THRESHOLD,
    MAX_CONTEXT_CHARS,
)

logger = logging.getLogger(__name__)


class RAGApp:
    """Coordinates the full RAG pipeline:
    input validation → decomposition → hybrid retrieval →
    score-drop filtering → context building → answer generation.
    """

    def __init__(self) -> None:
        logger.info("Initializing RAG Application...")
        self.retriever = Retriever()
        self.rag_service = RAGService()
        self.k_min = K_MIN
        self.max_per_query = MAX_PER_QUERY
        self.max_context_docs = MAX_CONTEXT_DOCS
        self.score_threshold = SCORE_DROP_THRESHOLD
        self.max_context_chars = MAX_CONTEXT_CHARS
        logger.info("RAG Application ready.")

    # ══════════════════════════════════════════════════════════
    # Public API
    # ══════════════════════════════════════════════════════════

    def query(self, question: str) -> str:
        """Full RAG pipeline — returns the generated answer string."""

        # ── 1. Validate ────────────────────────────────────────
        question = question.strip()
        if not question:
            logger.warning("Empty query received — skipping pipeline.")
            return "⚠️  Please enter a valid question."

        total_start = time.time()

        try:
            # ── 2. Decompose ───────────────────────────────────────
            sub_queries = self.rag_service.decompose_query(question)
            is_multi = len(sub_queries) > 1

            if is_multi:
                logger.info(
                    "Multi-entity → %d sub-queries: %s",
                    len(sub_queries),
                    sub_queries,
                )
            else:
                logger.info("Single-entity query.")

            # ── 3. Retrieve ────────────────────────────────────────
            t0 = time.time()
            if is_multi:
                results = self._retrieve_multi(sub_queries)
            else:
                results = self.retriever.retrieve(question)
            retrieval_time = time.time() - t0

            if not results:
                return "No relevant context found in the knowledge base."

            # ── 4. Filter ──────────────────────────────────────────
            filtered = self._filter_by_score(results)
            logger.info(
                "Filtering — before: %d → after: %d",
                len(results),
                len(filtered),
            )

            # ── 5. Build context (respecting char budget) ──────────
            final_docs = [doc for doc, _ in filtered]
            context = self._build_context(final_docs)
            logger.info(
                "Context — chunks: %d | chars: %d",
                len(final_docs),
                len(context),
            )

            # ── 6. Generate answer ─────────────────────────────────
            t0 = time.time()
            answer = self.rag_service.generate_answer(context, question)
            llm_time = time.time() - t0

            total_time = time.time() - total_start
            logger.info(
                "Latency — retrieval: %.2fs | LLM: %.2fs | total: %.2fs",
                retrieval_time,
                llm_time,
                total_time,
            )

            return answer
        except RuntimeError as e:
            logger.error("RAG pipeline error: %s", e)
            return f"Error processing query: {e}"
        except Exception as e:
            logger.exception("Unexpected error in RAG pipeline: %s", e)
            return "An unexpected error occurred. Please try again."

    def query_stream(
        self, question: str, history: str = ""
    ) -> Iterator[str]:
        """Streaming RAG pipeline for the Web UI.

        Yields a series of Server-Sent Event lines:

        1. A JSON metadata line with retrieved source filenames::

               data: {"event": "sources", "data": ["Google.txt", ...]}

        2. One or more token chunks (plain text)::

               data: {"event": "token", "data": "..."}

        3. A done signal::

               data: {"event": "done", "data": ""}
        """
        question = question.strip()
        if not question:
            yield 'data: {"event": "error", "data": "Empty question"}\n\n'
            return

        try:
            # Decompose
            sub_queries = self.rag_service.decompose_query(question)
            is_multi = len(sub_queries) > 1

            # Retrieve
            if is_multi:
                results = self._retrieve_multi(sub_queries)
            else:
                results = self.retriever.retrieve(question)

            if not results:
                yield (
                    'data: {"event": "token", "data": '
                    '"No relevant context found in the knowledge base."}\n\n'
                )
                yield 'data: {"event": "done", "data": ""}\n\n'
                return

            # Filter + build context
            filtered = self._filter_by_score(results)
            final_docs = [doc for doc, _ in filtered]
            context = self._build_context(final_docs)

            # Emit sources metadata first
            sources = list({
                doc.metadata.get("source", "unknown").split("/")[-1].split("\\")[-1]
                for doc in final_docs
            })
            yield f"data: {json.dumps({'event': 'sources', 'data': sources})}\n\n"

            # Stream answer tokens
            for chunk in self.rag_service.generate_answer_stream(
                context, question, history
            ):
                payload = json.dumps({"event": "token", "data": chunk})
                yield f"data: {payload}\n\n"

            yield 'data: {"event": "done", "data": ""}\n\n'

        except RuntimeError as e:
            logger.error("Streaming RAG pipeline error: %s", e)
            err = json.dumps({"event": "error", "data": str(e)})
            yield f"data: {err}\n\n"
        except Exception as e:
            logger.exception("Unexpected streaming error: %s", e)
            yield 'data: {"event": "error", "data": "An unexpected error occurred."}\n\n'

    # ══════════════════════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════════════════════

    def _retrieve_multi(
        self, sub_queries: List[str]
    ) -> List[Tuple[Document, float]]:
        """Retrieve per sub-query, deduplicate, keep highest score."""
        all_results: List[Tuple[Document, float]] = []

        for sub in sub_queries:
            all_results.extend(self.retriever.retrieve(sub))

        if not all_results:
            return []

        # Deduplicate by content — keep highest RRF score
        unique: dict[str, Tuple[Document, float]] = {}
        for doc, score in all_results:
            key = doc.page_content
            if key not in unique or score > unique[key][1]:
                unique[key] = (doc, score)

        return sorted(unique.values(), key=lambda x: x[1], reverse=True)

    def _filter_by_score(
        self, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Score-drop filtering (higher = better).

        1. Always return at least ``K_MIN`` chunks.
        2. Expand while ``score >= SCORE_DROP_THRESHOLD × best_score``.
        3. Never exceed ``MAX_CONTEXT_DOCS``.
        """
        if not results:
            return []

        # Ensure descending order (highest score first)
        results = sorted(results, key=lambda x: x[1], reverse=True)

        best_score = results[0][1]
        threshold = best_score * self.score_threshold

        # Always keep K_MIN
        selected = list(results[: self.k_min])

        for doc, score in results[self.k_min :]:
            if score >= threshold and len(selected) < self.max_context_docs:
                selected.append((doc, score))
            else:
                break

        logger.info(
            "Score filter — best: %.6f | threshold: %.6f | kept: %d",
            best_score,
            threshold,
            len(selected),
        )
        for doc, score in selected:
            src = doc.metadata.get("source", "unknown")
            logger.debug("  └─ %.6f | %s", score, src)

        return selected

    def _build_context(self, docs: List[Document]) -> str:
        """Concatenate chunk contents, hard-capped at MAX_CONTEXT_CHARS."""
        parts: list[str] = []
        total = 0

        for doc in docs:
            text = doc.page_content
            if total + len(text) > self.max_context_chars:
                remaining = self.max_context_chars - total
                if remaining > 100:
                    parts.append(text[:remaining])
                break
            parts.append(text)
            total += len(text)

        return "\n\n".join(parts)