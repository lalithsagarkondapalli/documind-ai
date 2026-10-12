"""DocuMind AI API. Run with: uvicorn app.main:app"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

from app.agents.graph import Deps, build_graph
from app.api.deps import AppState
from app.api.v1 import auth, documents, query, reviews
from app.core.cache import QueryCache
from app.core.config import Settings, get_settings
from app.core.security import hash_password
from app.core.tracing import setup_tracing
from app.db.models import User, make_session_factory
from app.services.index_sync import IndexSyncer
from app.services.llm import build_llm
from app.services.retrieval import RetrievalEngine

log = logging.getLogger("documind")
DEFAULT_JWT_SECRET = "change-me-in-production"
MIN_JWT_SECRET_BYTES = 32  # RFC 7518 section 3.2 minimum for HS256

DEMO_USERS = [
    ("admin", "acme", "admin"),
    ("reviewer", "acme", "reviewer"),
    ("viewer", "acme", "viewer"),
    ("globex-admin", "globex", "admin"),
]


def _seed_demo_users(session_factory, password: str) -> None:
    """Idempotent and race-safe: several replicas may start at once and all try to seed."""
    for username, org, role in DEMO_USERS:
        with session_factory() as db:
            if db.scalar(select(User).where(User.username == username)):
                continue
            db.add(User(org_id=org, username=username, role=role, password_hash=hash_password(password)))
            try:
                db.commit()
            except IntegrityError:
                db.rollback()  # another replica inserted it first; that is fine


def build_state(settings: Settings) -> AppState:
    if settings.environment == "production":
        if settings.jwt_secret == DEFAULT_JWT_SECRET:
            raise RuntimeError("Refusing to start in production with the default JWT secret")
        if len(settings.jwt_secret.encode()) < MIN_JWT_SECRET_BYTES:
            raise RuntimeError(f"JWT_SECRET must be at least {MIN_JWT_SECRET_BYTES} bytes in production (HS256)")

    _, session_factory = make_session_factory(settings.database_url)
    if settings.environment != "production":
        _seed_demo_users(session_factory, settings.demo_password)

    dense = None
    if settings.vector_backend == "qdrant":
        from app.services.vectorstore import QdrantBackend

        dense = QdrantBackend(url=settings.qdrant_url)
    engine = RetrievalEngine([], dense_backend=dense)

    # Vectors already live in a shared Qdrant server (other replicas wrote them); embedded/in-memory
    # indexes are per-process, so they must be (re)built locally.
    syncer = IndexSyncer(
        session_factory, engine, settings.index_sync_interval_seconds, upsert_dense=not settings.qdrant_url
    )
    syncer.sync()  # rebuild the index from persisted chunks at startup

    llm = build_llm(settings.llm_provider, settings.small_model, settings.frontier_model)
    graph = build_graph(Deps(engine, llm, settings.review_confidence_threshold, settings.retrieval_top_k))
    cache = QueryCache(settings.redis_url, settings.cache_ttl_seconds)
    if settings.environment == "production" and cache.backend != "redis":
        log.warning(
            "No REDIS_URL set: each replica has its own cache, so cached answers can go stale "
            "across replicas after an upload or review. Set REDIS_URL for multi-replica deployments."
        )
    return AppState(settings, session_factory, engine, graph, cache, llm, syncer)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.documind = build_state(settings)
        log.info(
            "DocuMind started: env=%s vector=%s cache=%s llm=%s",
            settings.environment,
            settings.vector_backend,
            app.state.documind.cache.backend,
            settings.llm_provider,
        )
        yield

    app = FastAPI(
        title="DocuMind AI",
        version="1.0.0",
        lifespan=lifespan,
        description="Multi-agent RAG for regulatory and financial document review.",
    )
    setup_tracing(app, settings.otlp_endpoint)
    for r in (auth.router, documents.router, query.router, reviews.router):
        app.include_router(r)

    @app.get("/health", tags=["ops"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready():
        st: AppState = app.state.documind
        with st.session_factory() as db:
            db.execute(text("SELECT 1"))
        return {"status": "ready", "chunks_indexed": len(st.engine.chunks), "cache": st.cache.backend}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
