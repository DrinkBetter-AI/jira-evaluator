"""Run each Streamlit smoke test as its own case.

The smoke tests under ``apptests/`` are scripts: they build a stubbed dashboard,
render a page and assert on what it says, printing a line per scenario. They are
run here in a subprocess rather than imported, so one script's monkeypatched
dashboard cannot leak into the next, and so their output is attached to the
failure rather than lost.

They are slow (each renders the app several times), so mark them::

    python3 -m pytest tests -q -m "not slow"
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

APPTESTS = Path(__file__).resolve().parent / "apptests"

SCRIPTS = sorted(p.name for p in APPTESTS.glob("*_apptest.py"))


@pytest.mark.slow
@pytest.mark.parametrize("script", SCRIPTS)
def test_page_renders_what_its_figures_say(script: str) -> None:
    done = subprocess.run(
        [sys.executable, str(APPTESTS / script)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert done.returncode == 0, f"{script}\n{done.stdout[-4000:]}\n{done.stderr[-4000:]}"
