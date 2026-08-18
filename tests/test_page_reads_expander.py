"""Phase 7: the "reads this page made" expander is off unless explicitly
turned on, and never draws anything when a page made no reads."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app  # noqa: E402


def test_reads_debug_is_off_by_default(monkeypatch):
    monkeypatch.delenv("SHOW_PAGE_READS", raising=False)
    assert app._reads_debug_enabled() is False


def test_reads_debug_recognises_common_truthy_spellings(monkeypatch):
    for value in ("1", "true", "True", "yes", "YES"):
        monkeypatch.setenv("SHOW_PAGE_READS", value)
        assert app._reads_debug_enabled() is True, value


def test_reads_debug_stays_off_for_anything_else(monkeypatch):
    for value in ("0", "false", "no", ""):
        monkeypatch.setenv("SHOW_PAGE_READS", value)
        assert app._reads_debug_enabled() is False, value


def test_render_page_reads_draws_nothing_for_a_page_with_no_reads():
    # No reads means an early return before any st.* call - the empty-render
    # case never has to know it isn't running inside a script context.
    assert app._render_page_reads([]) is None
