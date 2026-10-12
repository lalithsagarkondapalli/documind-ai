# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --wheel-dir /wheels -r requirements.txt

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 WEB_CONCURRENCY=2
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /data && chown 10001:10001 /data
# Writable default so the container boots as a non-root user with zero config.
# Real deployments override this with PostgreSQL via DATABASE_URL (compose and Kubernetes do).
ENV DATABASE_URL=sqlite:////data/documind.db
WORKDIR /app
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt && rm -rf /wheels
COPY app ./app
USER 10001
EXPOSE 8000
# python (not curl) because the slim image has no curl
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${WEB_CONCURRENCY} --proxy-headers"]
