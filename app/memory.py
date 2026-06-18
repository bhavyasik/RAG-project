"""In-memory session store for multi-turn conversation history.

Each session holds a list of ``{"role": "user"|"assistant", "content": str}``
dicts.  The store is intentionally kept as a plain module-level singleton so
it is straightforward to swap for a Redis or DB-backed implementation later.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import DefaultDict

# Keep at most this many Q&A *pairs* per session (older turns are pruned).
MAX_HISTORY_TURNS: int = 6

# session_id → ordered list of message dicts
_store: DefaultDict[str, list[dict]] = defaultdict(list)


# ── Public API ─────────────────────────────────────────────────────────────


def new_session() -> str:
    """Generate and register a fresh session ID."""
    sid = str(uuid.uuid4())
    _store[sid]  # initialise empty list via defaultdict
    return sid


def get_history(session_id: str) -> list[dict]:
    """Return the raw message list for *session_id* (may be empty)."""
    return list(_store[session_id])


def get_history_string(session_id: str) -> str:
    """Return conversation history as a plain text string for LLM injection.

    Format::

        User: <question>
        Assistant: <answer>
        User: <question>
        ...
    """
    messages = _store.get(session_id, [])
    if not messages:
        return ""
    lines = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def append_turn(session_id: str, question: str, answer: str) -> None:
    """Append a completed Q&A turn and prune old turns beyond MAX_HISTORY_TURNS."""
    _store[session_id].append({"role": "user", "content": question})
    _store[session_id].append({"role": "assistant", "content": answer})

    # Prune: keep only the last MAX_HISTORY_TURNS pairs (× 2 messages each)
    max_messages = MAX_HISTORY_TURNS * 2
    if len(_store[session_id]) > max_messages:
        _store[session_id] = _store[session_id][-max_messages:]


def list_sessions() -> list[str]:
    """Return all session IDs that currently exist in the store."""
    return list(_store.keys())


def clear_session(session_id: str) -> None:
    """Wipe all history for the given session."""
    _store[session_id] = []


def session_exists(session_id: str) -> bool:
    """Return True if the session ID has been registered."""
    return session_id in _store
