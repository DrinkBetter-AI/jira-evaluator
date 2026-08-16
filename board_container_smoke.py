"""Does a printed board come out whole inside the container it is deployed in?

Two things the deployment can lose silently: WeasyPrint's typesetting libraries,
which would leave every board offered as HTML, and a headless browser that
refuses to start as root, which would leave every chart on the board a labelled
gap. Both are caught here rather than in production, by printing one board with
a chart on it inside the image itself.
"""

import time

import pandas as pd
import plotly.express as px

import snapshot


def main() -> None:
    board = snapshot.Snapshot("engineering")
    board.observe("header", ("Engineering",), {})
    board.observe("metric", ("Open tickets", 13), {})
    frame = pd.DataFrame({"week": ["-2w", "-1w", "now"], "hours": [12, 4.6, 47]})
    board.observe("dataframe", (frame,), {"hide_index": True})
    board.observe("plotly_chart", (px.bar(frame, x="week", y="hours"),), {})
    board.observe("bar_chart", (frame.set_index("week"),), {})

    started = time.time()
    page = board.html()
    charts = page.count("data:image/png;base64,")
    printed = snapshot.to_pdf(page)
    print(f"charts drawn: {charts} of 2")
    print(f"pdf bytes: {0 if printed is None else len(printed)}")
    print(f"seconds: {time.time() - started:.1f}")
    if charts != 2 or not printed or not printed.startswith(b"%PDF-"):
        raise SystemExit("the container cannot print a whole board")
    print("the container prints a whole board: ok")


if __name__ == "__main__":
    main()
