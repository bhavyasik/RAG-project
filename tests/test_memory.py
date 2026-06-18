"""Unit tests for app.memory — in-memory session store.

No external services required.
"""

from __future__ import annotations
import pytest


# Re-import to get a clean module state per test via monkeypatching
def fresh_memory():
    """Return module with a clean internal store."""
    import importlib
    import app.memory as mem
    # Reset the internal store directly
    mem._store.clear()
    return mem


class TestNewSession:
    def test_returns_string(self):
        mem = fresh_memory()
        sid = mem.new_session()
        assert isinstance(sid, str) and len(sid) > 0

    def test_unique_ids(self):
        mem = fresh_memory()
        ids = {mem.new_session() for _ in range(10)}
        assert len(ids) == 10

    def test_session_appears_in_list(self):
        mem = fresh_memory()
        sid = mem.new_session()
        assert sid in mem.list_sessions()


class TestAppendAndHistory:
    def test_empty_history_string(self):
        mem = fresh_memory()
        sid = mem.new_session()
        assert mem.get_history_string(sid) == ""

    def test_single_turn(self):
        mem = fresh_memory()
        sid = mem.new_session()
        mem.append_turn(sid, "Hello?", "Hi there!")
        h = mem.get_history_string(sid)
        assert "User: Hello?" in h
        assert "Assistant: Hi there!" in h

    def test_multiple_turns_ordered(self):
        mem = fresh_memory()
        sid = mem.new_session()
        mem.append_turn(sid, "Q1", "A1")
        mem.append_turn(sid, "Q2", "A2")
        h = mem.get_history_string(sid)
        assert h.index("Q1") < h.index("Q2")

    def test_raw_history_messages(self):
        mem = fresh_memory()
        sid = mem.new_session()
        mem.append_turn(sid, "question", "answer")
        msgs = mem.get_history(sid)
        assert msgs[0] == {"role": "user",      "content": "question"}
        assert msgs[1] == {"role": "assistant",  "content": "answer"}


class TestPruning:
    def test_pruning_respects_max_turns(self):
        mem = fresh_memory()
        sid = mem.new_session()
        # Fill beyond limit
        for i in range(mem.MAX_HISTORY_TURNS + 5):
            mem.append_turn(sid, f"Q{i}", f"A{i}")
        msgs = mem.get_history(sid)
        assert len(msgs) == mem.MAX_HISTORY_TURNS * 2

    def test_pruning_keeps_latest(self):
        mem = fresh_memory()
        sid = mem.new_session()
        for i in range(mem.MAX_HISTORY_TURNS + 3):
            mem.append_turn(sid, f"Q{i}", f"A{i}")
        h = mem.get_history_string(sid)
        # Oldest messages should be gone
        assert "Q0" not in h
        # Latest should be present
        last = mem.MAX_HISTORY_TURNS + 2
        assert f"Q{last}" in h


class TestClearAndExists:
    def test_clear_removes_messages(self):
        mem = fresh_memory()
        sid = mem.new_session()
        mem.append_turn(sid, "q", "a")
        mem.clear_session(sid)
        assert mem.get_history(sid) == []

    def test_session_exists(self):
        mem = fresh_memory()
        assert not mem.session_exists("nonexistent")
        sid = mem.new_session()
        assert mem.session_exists(sid)

    def test_clear_does_not_delete_session(self):
        """Clearing keeps the session ID alive."""
        mem = fresh_memory()
        sid = mem.new_session()
        mem.clear_session(sid)
        assert mem.session_exists(sid)
