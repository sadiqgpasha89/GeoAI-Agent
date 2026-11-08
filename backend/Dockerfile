# syntax=docker/dockerfile:1.7
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY backend/pyproject.toml ./pyproject.toml
COPY backend/src ./src
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --upgrade pip && /opt/venv/bin/pip install ".[postgres]"

FROM python:3.12-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 GEOAI_DATA_DIR=/data
RUN useradd --system --uid 10001 --home /app geoai && mkdir -p /data /app && chown geoai /data /app
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
USER geoai
# Deterministic synthetic sample scenes (no third-party imagery is bundled).
RUN geoai-generate-samples --out /data/samples
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
CMD ["uvicorn", "--factory", "geoai.api.app:app_factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
