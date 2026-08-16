FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# What printing the board needs and Python cannot bring with it: Pango and
# Cairo for WeasyPrint's typesetting, a font for it to typeset with, and a
# headless Chromium for Kaleido to draw the charts in. Without these the page
# still serves; only the PDF loses its charts, or itself.
RUN apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        fonts-dejavu-core \
        libcairo2 \
        libgdk-pixbuf-2.0-0 \
        libpango-1.0-0 \
        libpangoft2-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
ENV BROWSER_PATH=/usr/bin/chromium

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The theme lives here; without it the hosted app falls back to Streamlit's default.
COPY .streamlit ./.streamlit
COPY *.py ./

# Cloud Run injects PORT; Streamlit needs it on the command line.
ENV PORT=8080
EXPOSE 8080
CMD exec streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0
