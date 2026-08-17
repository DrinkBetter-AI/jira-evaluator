"""The navigation's invariants.

The dashboard used to be one page. It is six, and the ways that quietly breaks
are all structural: two pages sharing a url_path (Streamlit refuses the pair), no
default (nothing answers "/"), a page with no heading (the title falls back to a
generic one and the reader is told nothing), or a new page that nobody remembered
to make reachable.
"""

from __future__ import annotations

import app


def test_exactly_one_page_is_the_default():
    """Two defaults is ambiguous; none means "/" answers with nothing."""
    defaults = [spec for spec in app._page_specs() if spec.default]
    assert len(defaults) == 1
    assert defaults[0].title == app.TODAY_PAGE_TITLE


def test_the_default_page_owns_the_empty_path():
    """Streamlit forces the default's url_path to "" - declaring another is a lie.

    The /engineering 404 came from exactly this: a page declared at "engineering"
    while also being the default, so the path was ignored and the address people
    had bookmarked answered "Page not found".
    """
    default = next(spec for spec in app._page_specs() if spec.default)
    assert default.url_path == ""


def test_no_two_pages_share_a_path():
    """A page hash is derived from the path alone, so st.navigation rejects a repeat."""
    paths = [spec.url_path for spec in app._page_specs()]
    assert len(paths) == len(set(paths))


def test_engineering_keeps_its_address():
    """Links to /engineering are in Slack and in bookmarks. They must keep working."""
    paths = {spec.url_path for spec in app._page_specs()}
    assert "engineering" in paths


def test_every_page_has_a_heading_of_its_own():
    """A page with no heading renders "VinoVoss Dashboard" and says nothing.

    This is the assertion that fails when someone adds a page and forgets that a
    one-word navigation label is not a heading.
    """
    missing = [
        spec.title for spec in app._page_specs() if spec.title not in app.PAGE_HEADINGS
    ]
    assert missing == []


def test_every_page_renders_from_a_callable():
    for spec in app._page_specs():
        assert callable(spec.render), spec.title


def test_the_six_engineering_pages_are_all_registered():
    """Named explicitly: a page silently dropped from the list is the failure mode."""
    titles = [spec.title for spec in app._page_specs()]
    for expected in (
        app.TODAY_PAGE_TITLE,
        app.PEOPLE_PAGE_TITLE,
        app.DELIVERY_PAGE_TITLE,
        app.CODE_PAGE_TITLE,
        app.PLANNING_PAGE_TITLE,
        app.ENGINEERING_PAGE_TITLE,
    ):
        assert expected in titles


def test_business_is_absent_when_it_cannot_be_read(monkeypatch):
    """The shop's page should not appear when its credentials are not configured."""
    monkeypatch.setattr(app, "_business_readable", lambda: False)
    assert app.BUSINESS_PAGE_TITLE not in [s.title for s in app._page_specs()]


def test_business_appears_when_it_can_be_read(monkeypatch):
    monkeypatch.setattr(app, "_business_readable", lambda: True)
    specs = app._page_specs()
    assert specs[-1].title == app.BUSINESS_PAGE_TITLE
    assert specs[-1].url_path == "business"
    # Still unique once it joins.
    paths = [spec.url_path for spec in specs]
    assert len(paths) == len(set(paths))
