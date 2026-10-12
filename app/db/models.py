"""SQLAlchemy models. SQLite locally, PostgreSQL in compose/Kubernetes (DATABASE_URL)."""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine
from sqlalchemy.exc import IntegrityError, OperationalError, ProgrammingError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(String, index=True)
    username: Mapped[str] = mapped_column(String, unique=True)
    role: Mapped[str] = mapped_column(String)  # admin | reviewer | viewer
    password_hash: Mapped[str] = mapped_column(String)


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(String, index=True)
    filename: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, default="indexed")
    page_count: Mapped[int] = mapped_column(Integer, default=1)
    n_chunks: Mapped[int] = mapped_column(Integer, default=0)
    uploaded_by: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ChunkRow(Base):
    __tablename__ = "chunks"
    seq: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)  # replicas sync on this
    id: Mapped[str] = mapped_column(String, unique=True, index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    org_id: Mapped[str] = mapped_column(String, index=True)
    page_number: Mapped[int] = mapped_column(Integer)
    chunk_index: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)


class RiskFinding(Base):
    __tablename__ = "risk_findings"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(String, index=True)
    document_id: Mapped[str | None] = mapped_column(String, nullable=True)
    rule_id: Mapped[str] = mapped_column(String)
    rule_name: Mapped[str] = mapped_column(String)
    severity: Mapped[str] = mapped_column(String)
    rationale: Mapped[str] = mapped_column(Text)
    source_chunk_ids: Mapped[list] = mapped_column(JSON)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String, default="pending_review", index=True)
    reviewed_by: Mapped[str | None] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    """Append-only: the application never updates or deletes rows in this table."""

    __tablename__ = "audit_log"
    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    org_id: Mapped[str] = mapped_column(String)
    actor_type: Mapped[str] = mapped_column(String)  # agent | human | system
    actor_id: Mapped[str] = mapped_column(String)
    action: Mapped[str] = mapped_column(String)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    __table_args__ = (Index("idx_audit_org_created", "org_id", "created_at"),)


def make_session_factory(database_url: str):
    kwargs = {"connect_args": {"check_same_thread": False}} if database_url.startswith("sqlite") else {}
    if database_url in ("sqlite://", "sqlite:///:memory:"):
        from sqlalchemy.pool import StaticPool

        kwargs["poolclass"] = StaticPool
    engine = create_engine(database_url, pool_pre_ping=True, **kwargs)
    # Replicas starting together can race on first-ever CREATE TABLE; retry rather than crash-loop.
    for attempt in range(5):
        try:
            Base.metadata.create_all(engine)
            break
        except (OperationalError, ProgrammingError, IntegrityError):
            if attempt == 4:
                raise
            time.sleep(1 + attempt)
    return engine, sessionmaker(bind=engine, expire_on_commit=False)
