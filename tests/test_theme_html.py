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


# ---------------------------------------------------------------------------
# Task 1B - the full component kit. Everything below exercises the new,
# returning form of the contract; the tests above cover the legacy shapes
# still auto-detected for pages/code.py and pages/delivery.py.
# ---------------------------------------------------------------------------

import inspect

_CONTRACT = {
    "css": [],
    "render": ["fragments"],
    "page_header": ["brand", "freshness", "sources"],
    "section": ["title", "note"],
    "foot": ["text"],
    "hero": ["kicker", "tone", "big", "unit", "headline", "meter_pct", "sub", "link"],
    "decide_cards": ["cards"],
    "callout": ["tone", "title", "body"],
    "tiles": ["cards"],
    "spark": ["series", "hue", "fill", "w", "h"],
    "linechart": ["series", "x_labels", "hues", "w", "h", "aria"],
    "legend": ["keys"],
    "hbars": ["bars"],
    "table": ["columns", "rows"],
    "avatar": ["initials", "hue"],
    "rolechip": ["label"],
    "scorebar": ["pct", "tone", "value", "note"],
    "chip": ["label", "tone"],
    "scorecard": ["components", "overall", "measurable", "note"],
    "intro_band": ["text"],
    "evrow": ["html_fragment"],
    "innocent": ["text"],
    "maskbar": ["days", "scale"],
    "sprint_card": ["board", "accent", "name", "window", "rows"],
    "stub_cards": ["cards"],
}


def test_every_contracted_component_exists_with_its_named_parameters():
    """The frozen API, checked by name: every function Phase 3 calls exists,
    and its first N parameters are the ones the contract names (extra
    keyword-only params like ``write=``/``tab=``/``section=`` for the
    backward-compat dispatchers are allowed to follow)."""
    for name, params in _CONTRACT.items():
        assert hasattr(theme_html, name), f"missing {name}"
        func = getattr(theme_html, name)
        sig = inspect.signature(func)
        actual = list(sig.parameters)
        assert actual[: len(params)] == params, (name, actual)


def test_named_tuples_are_exported_and_buildable_positionally_and_by_keyword():
    theme_html.Tile("Open PRs", "79")
    theme_html.Tile(label="Open PRs", value="79")
    theme_html.Bar("repo", "12", 40.0)
    theme_html.Bar(name="repo", value="12", pct=40.0)
    theme_html.Column("Title")
    theme_html.Column(label="Title", kind="link")
    theme_html.Cell("x")
    theme_html.Cell(value="x", tone="warn")
    theme_html.DecideCard("Triage", "warn", "5", "stuck", "why")
    theme_html.Component("Delivery", 20, 84.0)


# ---------------------------------------------------------------------------
# spark() / linechart() - server-side SVG, no <script>, snapshot-tested.
# ---------------------------------------------------------------------------

_SPARK_SERIES = [2, 4, 3, 5, 3.5, 4, 6, 3, 5, 4, 4.5, 5]

_SPARK_EXPECTED = (
    '<svg class="spark" viewBox="0 0 96 30" width="96" height="30">'
    '<polyline points="3.00,27.00 11.18,15.00 19.36,21.00 27.55,9.00 35.73,18.00 '
    '43.91,15.00 52.09,3.00 60.27,21.00 68.45,9.00 76.64,15.00 84.82,12.00 93.00,9.00" '
    'fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round" '
    'stroke-linecap="round"/><circle cx="93.00" cy="9.00" r="3.5" fill="var(--s1)" '
    'stroke="#fff" stroke-width="2"/></svg>'
)


def test_spark_matches_its_exact_snapshot():
    assert theme_html.spark(_SPARK_SERIES, "s1") == _SPARK_EXPECTED


def test_spark_has_no_script_tag():
    out = theme_html.spark(_SPARK_SERIES, "s1", fill=True)
    assert "<script" not in out
    assert out.startswith("<svg")
    assert "<polygon" in out  # fill=True draws the area under the line


def test_spark_single_point_and_empty_series_do_not_crash():
    single = theme_html.spark([7.0], "s2")
    assert single.count("<circle") == 1
    assert "<polyline" in single  # a one-point polyline, drawn but invisible

    empty = theme_html.spark([], "s2")
    assert empty == '<svg class="spark" viewBox="0 0 96 30" width="96" height="30"></svg>'


_LINE_RESOLVED = [70, 88, 64, 92, 81, 75, 102, 84, 93, 78, 89, 96]
_LINE_MERGED = [98, 110, 92, 130, 104, 120, 141, 109, 133, 117, 124, 126]
_LINE_WEEKS = [
    "May 25", "Jun 1", "Jun 8", "Jun 15", "Jun 22", "Jun 29",
    "Jul 6", "Jul 13", "Jul 20", "Jul 27", "Aug 3", "Aug 10",
]


def test_linechart_matches_its_exact_snapshot():
    out = theme_html.linechart(
        {"resolved": _LINE_RESOLVED, "merged": _LINE_MERGED},
        _LINE_WEEKS,
        ["s1", "s2"],
        aria="Line chart of weekly tickets resolved and PRs merged",
    )
    assert out.startswith(
        '<svg viewBox="0 0 620 210" width="100%" height="210" role="img" '
        'aria-label="Line chart of weekly tickets resolved and PRs merged">'
    )
    assert "<script" not in out
    # Twelve points per series, two series, none dropped.
    assert out.count("<circle") == 24
    assert out.count("<polyline") == 2
    assert '<title>May 25: 70 resolved</title>' in out
    assert '<title>Aug 10: 126 merged</title>' in out
    # Gridlines run 0..200 in steps of 50, "nice"-rounded above the true max of 141.
    assert '<text x="32" y="18.0" text-anchor="end">200</text>' in out


def test_linechart_handles_a_single_series():
    out = theme_html.linechart({"a": [1, 2, 3]}, ["x", "y", "z"], ["s1"])
    assert out.count("<circle") == 3
    assert out.count("<polyline") == 1
    assert "<script" not in out


def test_linechart_handles_series_of_differing_length():
    out = theme_html.linechart(
        {"a": [1, 2, 3], "b": [5, 6]}, ["x", "y", "z"], ["s1", "s2"]
    )
    assert out.count("<circle") == 5  # 3 + 2, nothing crashed on the mismatch
    assert out.count("<polyline") == 2


def test_linechart_handles_an_all_zero_series():
    out = theme_html.linechart({"a": [0, 0, 0]}, ["x", "y", "z"], ["s1"])
    assert "<script" not in out
    assert out.count("<circle") == 3
    # A flat line at the bottom of the plot, not a crash from dividing by a
    # zero-width domain.
    assert 'points="40.00,182.00 320.00,182.00 600.00,182.00"' in out


def test_linechart_handles_an_empty_series_dict():
    out = theme_html.linechart({}, ["x", "y"], ["s1"])
    assert "<script" not in out
    assert "<circle" not in out
    assert "<polyline" not in out
    # Still a labelled frame: the axis and the x-tick text are drawn.
    assert '>x</text>' in out and '>y</text>' in out


def test_linechart_handles_an_empty_list_for_one_series():
    out = theme_html.linechart({"a": []}, ["x", "y"], ["s1"])
    assert "<script" not in out
    assert "<circle" not in out
    assert "<polyline" not in out


# ---------------------------------------------------------------------------
# table() - the new cell kinds and their escaping.
# ---------------------------------------------------------------------------

_HOSTILE = '<img src=x onerror=alert(1)>'


def test_every_cell_kind_except_html_escapes_hostile_input():
    columns = [
        theme_html.Column("Text", "text"),
        theme_html.Column("Num", "num"),
        theme_html.Column("Link", "link"),
        theme_html.Column("Strong", "strong-num"),
        theme_html.Column("Chip", "chip"),
        theme_html.Column("Avatar", "avatar"),
        theme_html.Column("Scorebar", "scorebar"),
    ]
    row = [
        theme_html.Cell(_HOSTILE),
        theme_html.Cell(_HOSTILE),
        theme_html.Cell(_HOSTILE),
        theme_html.Cell(_HOSTILE),
        theme_html.Cell(_HOSTILE, tone="warn"),
        theme_html.Cell(_HOSTILE, hue="s1"),
        theme_html.Cell(_HOSTILE, tone="good", pct=50, note=_HOSTILE),
    ]
    out = theme_html.table(columns, [row])
    assert _HOSTILE not in out
    assert out.count("&lt;img src=x onerror=alert(1)&gt;") == 7


def test_html_kind_passes_a_server_built_fragment_through_unescaped():
    columns = [theme_html.Column("Html", "html")]
    row = [theme_html.Cell("<b>safe server html</b>")]
    out = theme_html.table(columns, [row])
    assert "<b>safe server html</b>" in out


def test_table_new_form_returns_a_bare_table_not_a_card():
    out = theme_html.table([theme_html.Column("A", "text")], [[theme_html.Cell("x")]])
    assert out.startswith("<table")
    assert "chart-title" not in out
    assert '<div class="card">' not in out


# ---------------------------------------------------------------------------
# render() - the single .vv scope.
# ---------------------------------------------------------------------------


def test_render_wraps_every_fragment_in_one_vv_div(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        theme_html.st, "markdown", lambda body, **k: captured.setdefault("html", body)
    )
    theme_html.render("<p>a</p>", "<p>b</p>")
    assert captured["html"] == '<div class="vv"><p>a</p><p>b</p></div>'


# ---------------------------------------------------------------------------
# scorecard() - the insufficient-component invariant.
# ---------------------------------------------------------------------------


def test_insufficient_component_never_renders_as_a_number():
    components = [
        theme_html.Component("Delivery", 20, 84.0),
        # A caller could still hand this a numeric score by mistake - it
        # must never reach the page.
        theme_html.Component("Urgent response", 5, 999.0, "no High+ held", sufficient=False),
    ]
    out = theme_html.scorecard(components, overall="81", measurable="95", note="")
    assert "999" not in out
    assert "n/a" in out
    assert 'class="comp na"' in out


def test_sufficient_component_renders_its_score_as_a_number():
    components = [theme_html.Component("Delivery", 20, 84.0, "n=18 PRs")]
    out = theme_html.scorecard(components, overall="84", measurable="20", note="")
    assert ">84</div>" in out
    assert "n/a" not in out


# ---------------------------------------------------------------------------
# tabular-nums on every numeric column.
# ---------------------------------------------------------------------------


def test_numeric_table_columns_carry_tabular_nums():
    assert "td.num" in theme_html._CSS and "tabular-nums" in theme_html._CSS
    out = theme_html.table(
        [theme_html.Column("N", "num")], [[theme_html.Cell(3)]]
    )
    assert '<th class="num">N</th>' in out
    assert '<td class="num">3</td>' in out


# ---------------------------------------------------------------------------
# tiles()/hbars() - new returning form, and the report-recording hook that
# fixes Delivery/Code's absence from the printable report (see
# docs/assumptions/1B.md for the verdict on the original bug).
# ---------------------------------------------------------------------------


def test_tiles_new_form_returns_a_string_and_never_writes(monkeypatch):
    calls = []
    monkeypatch.setattr(theme_html.st, "markdown", lambda *a, **k: calls.append(a))
    out = theme_html.tiles([theme_html.Tile("Open PRs", "79", delta="+3", delta_dir="up", delta_good=False)])
    assert calls == []  # nothing written - the caller owns render()
    assert '<div class="kpis">' in out
    assert "▲" in out and "+3" in out
    assert 'class="delta crit-t"' in out  # up is bad here: colour follows delta_good, not the arrow


def test_hbars_new_form_returns_bare_rows_no_card():
    out = theme_html.hbars([theme_html.Bar("vinovoss-crm", "100%", 100.0, tone="s8")])
    assert out.startswith('<div class="hbars">')
    assert "chart-title" not in out
    assert "var(--s8)" in out


def test_hbars_dim_bar_omits_inline_background_so_the_css_class_wins():
    out = theme_html.hbars([theme_html.Bar("No team set", "15", 13.0, dim=True)])
    assert 'class="hbar dim"' in out
    assert "background" not in out.split("track")[1].split(">")[0]


def test_a_rendered_tile_set_appears_in_the_report_recorder(monkeypatch):
    """The fix: tiles() records into the same session-state report
    page_shared._tile()/_kpis() use, when a caller passes tab=/section=."""
    state: dict = {}
    monkeypatch.setattr(theme_html.st, "session_state", state)
    theme_html.tiles(
        [theme_html.Tile("Open PRs", "79", note="excl. drafts")],
        tab="Engineering",
        section="Code",
    )
    built = state["tab_reports"]["Engineering"]
    rendered = built.html()
    assert "Open PRs" in rendered
    assert "79" in rendered


def test_hbars_also_records_when_given_a_tab_and_section(monkeypatch):
    state: dict = {}
    monkeypatch.setattr(theme_html.st, "session_state", state)
    theme_html.hbars(
        [theme_html.Bar("vinovoss-crm", "100%", 100.0)],
        tab="Engineering",
        section="Code",
    )
    built = state["tab_reports"]["Engineering"]
    rendered = built.html()
    assert "vinovoss-crm" in rendered
    assert "100%" in rendered


def test_recording_is_a_no_op_without_tab_and_section():
    """A caller that never passes tab=/section= (any test, any snippet) must
    not need a live st.session_state at all - _record() returns before ever
    touching it."""
    out = theme_html.tiles([theme_html.Tile("x", "1")])
    assert '<div class="kpis">' in out
