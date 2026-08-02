FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The theme lives here; without it the hosted app falls back to Streamlit's default.
COPY .streamlit ./.streamlit
COPY *.py ./

# Cloud Run injects PORT; Streamlit needs it on the command line.
ENV PORT=8080
EXPOSE 8080
CMD exec streamlit run app.py --server.port "$PORT" --server.address 0.0.0.0
