# RoadWatch dashboard image. Data are NOT baked in: mount the built data/ and outputs/ directories read-only (see docs/final/integration_runbook.md).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# exact tested dependency versions first (better layer caching), then the package
COPY requirements-lock.txt pyproject.toml ./
RUN pip install -r requirements-lock.txt
COPY src ./src
COPY config ./config
COPY app.py ./
RUN pip install --no-deps -e .

# optional integration SDKs (only needed if you enable those features)
ARG WITH_INTEGRATIONS=0
RUN if [ "$WITH_INTEGRATIONS" = "1" ]; then pip install google-genai psycopg[binary]; fi

# unprivileged user; outputs/integrations is the only writable path (caches)
RUN useradd --create-home --uid 10001 roadwatch && mkdir -p /app/data /app/outputs/integrations && chown -R roadwatch /app/outputs/integrations
USER roadwatch

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).read()==b'ok' else 1)"

CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--browser.gatherUsageStats=false", "--server.enableCORS=true", "--server.enableXsrfProtection=true", "--client.toolbarMode=minimal"]
