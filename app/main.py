"""Main entry point — run the full RAG pipeline from the command line."""

import logging
import sys

from .ingestion import run_ingestion
from .rag_app import RAGApp
from .api import run_server
from .config import LOG_LEVEL

# ── Logging setup ──────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    """Interactive CLI: choose to ingest, query, or run server."""
    print("=" * 50)
    print("  RAG Pipeline — Interactive Mode")
    print("=" * 50)
    print("\nChoose an option:")
    print("  1. Ingest  — index documents from docs/")
    print("  2. Query   — ask a question")
    print("  3. Server  — start FastAPI server")
    print()

    choice = input("Enter your choice (1, 2, or 3): ").strip()

    if choice == "1":
        run_ingestion()

    elif choice == "2":
        app = RAGApp()
        while True:
            question = input("\nEnter your query (or 'exit'): ").strip()
            if question.lower() == "exit":
                break
            if not question:
                print("⚠️  Empty query — please type a question.")
                continue

            answer = app.query(question)
            print("\n" + "─" * 50)
            print("Answer:")
            print(answer)
            print("─" * 50)

    elif choice == "3":
        print("\nStarting FastAPI server...")
        print("Press Ctrl+C to stop")
        run_server()

    else:
        print(f"Invalid choice: '{choice}'. Please enter 1, 2, or 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()
