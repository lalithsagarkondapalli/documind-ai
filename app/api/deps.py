from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session, sessionmaker


@dataclass
class AppState:
    settings: Any
    session_factory: sessionmaker
    engine: Any  # RetrievalEngine
    graph: Any  # compiled LangGraph
    cache: Any  # QueryCache
    llm: Any
    syncer: Any = None  # IndexSyncer


def get_state(request: Request) -> AppState:
    return request.app.state.documind


def get_db(request: Request):
    db: Session = request.app.state.documind.session_factory()
    try:
        yield db
    finally:
        db.close()


def audit(db: Session, org_id: str, actor_type: str, actor_id: str, action: str, payload: dict | None = None):
    from app.db.models import AuditLog

    db.add(AuditLog(org_id=org_id, actor_type=actor_type, actor_id=actor_id, action=action, payload=payload or {}))
