"""The HTML component layer's safety and arithmetic.

These components emit markup built from Jira and GitHub data. The two ways that
goes wrong are injection (a ticket summary containing markup executes in the
CEO's browser) and bars that lie (widths not proportional to values). Both are
pinned here.
"""

from __future__ import annotations

import pandas as pd

import theme_html


def test_a_hostile_ticket_title_is_text_not_markup(monkeypatch):
    """A summary like <img onerror=...> must render as text.

    The dashboard is read by the person who decides renewals; its inputs are
    written by the people being measured. That asymmetry is why escaping is a
    test and not a hope.
    """
    captured = {}
    monkeypatch.setattr(
        theme_html.st, "markdown", lambda body, **k: captured.setdefault("html", body)
    )
    frame = pd.DataFrame(
        {
            "url": ["https://github.com/x/y/pull/7"],
            "title": ['<img src=x onerror="alert(1)">'],
        }
    )
    theme_html.table(frame, [("url", "PR", "link"), ("title", "Title", "text")], title="t")
    assert "<img src=x" not in captured["html"]
    assert "&lt;img" in captured["html"]


def test_a_non_http_link_renders_as_none_not_as_a_link(monkeypatch):
    """javascript: URLs from a poisoned field must not become clickable."""
    captured = {}
    monkeypatch.setattr(
        theme_html.st, "markdown", lambda body, **k: captured.setdefault("html", body)
    )
    frame = pd.DataFrame({"url": ["javascript:alert(1)"], "title": ["x"]})
    theme_html.table(frame, [("url", "PR", "link"), ("title", "T", "text")], title="t")
    assert "javascript:" not in captured["html"]
    assert "none" in captured["html"]


def test_a_scheme_that_merely_starts_with_http_is_not_a_link(monkeypatch):
    """``httpfoo:`` passes a prefix check and must still not be emitted."""
    captured = {}
    monkeypatch.setattr(
        theme_html.st, "markdown", lambda body, **k: captured.setdefault("html", body)
    )
    frame = pd.DataFrame({"url": ["httpevil:alert(1)"], "title": ["x"]})
    theme_html.table(frame, [("url", "PR", "link"), ("title", "T", "text")], title="t")
    assert "httpevil" not in captured["html"]
    assert "none" in captured["html"]


def test_bar_widths_are_proportional_to_values(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        theme_html.st, "markdown", lambda body, **k: captured.setdefault("html", body)
    )
    theme_html.hbars(
        [("a", 100.0, "100"), ("b", 50.0, "50")], title="t", severity=False
    )
    assert 'width:100.0%' in captured["html"]
    assert 'width:50.0%' in captured["html"]


def test_severity_coloring_marks_the_worst_repo_red(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        theme_html.st, "markdown", lambda body, **k: captured.setdefault("html", body)
    )
    theme_html.hbars(
        [("bad", 100.0, "100%"), ("fine", 10.0, "10%")], title="t", severity=True
    )
    html = captured["html"]
    assert "--s8" in html  # the 100% bar is critical red
    assert "--s3" in html  # the 10% bar is calm green


def test_tiles_escape_labels_and_values(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        theme_html.st, "markdown", lambda body, **k: captured.setdefault("html", body)
    )
    theme_html.tiles([("<b>lbl</b>", "<i>9</i>", "note", "neutral")])
    assert "<b>lbl</b>" not in captured["html"]
    assert "&lt;b&gt;lbl&lt;/b&gt;" in captured["html"]


def test_zero_rows_render_an_empty_table_not_an_error(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        theme_html.st, "markdown", lambda body, **k: captured.setdefault("html", body)
    )
    theme_html.table(pd.DataFrame(), [("a", "A", "text")], title="t")
    assert "<tbody></tbody>" in captured["html"]
