from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AppState, audit, get_db, get_state
from app.api.schemas import QueryRequest, QueryResponse
from app.core.cache import cache_key
from app.core.metrics import FINDINGS_TOTAL, QUERY_LATENCY, QUERY_TOTAL
from app.core.security import Principal, require_role
from app.db.models import Document, RiskFinding

router = APIRouter(prefix="/v1/query", tags=["query"])


def _persist_findings(db: Session, org_id: str, document_id: str | None, findings: list[dict]) -> list[dict]:
    """Store findings, de-duplicating repeat queries that surface the same clause + rule."""
    out = []
    for f in findings:
        existing = db.scalars(
            select(RiskFinding).where(RiskFinding.org_id == org_id, RiskFinding.rule_id == f["rule_id"])
        ).all()
        row = next((e for e in existing if e.source_chunk_ids == f["source_chunk_ids"]), None)
        if row is None:
            row = RiskFinding(
                org_id=org_id,
                document_id=document_id,
                rule_id=f["rule_id"],
                rule_name=f["rule_name"],
                severity=f["severity"],
                rationale=f["rationale"],
                source_chunk_ids=f["source_chunk_ids"],
                confidence=f["confidence"],
                status=f["status"],
            )
            db.add(row)
            db.flush()
            audit(
                db,
                org_id,
                "agent",
                "langgraph",
                "finding.created",
                {"finding_id": row.id, "rule_id": row.rule_id, "status": row.status},
            )
            FINDINGS_TOTAL.labels(row.status, row.severity).inc()
        out.append(
            {
                "id": row.id,
                "rule_id": row.rule_id,
                "rule_name": row.rule_name,
                "severity": row.severity,
                "confidence": row.confidence,
                "rationale": row.rationale,
                "source_chunk_ids": row.source_chunk_ids,
                "status": row.status,
            }
        )
    return out


@router.post("", response_model=QueryResponse)
def query(
    req: QueryRequest,
    user: Principal = Depends(require_role("viewer")),
    db: Session = Depends(get_db),
    state: AppState = Depends(get_state),
):
    t0 = time.perf_counter()
    if req.document_id:
        owned = db.scalar(select(Document.id).where(Document.id == req.document_id, Document.org_id == user.org_id))
        if not owned:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")

    key = cache_key(user.org_id, req.document_id, req.question)
    cached = state.cache.get(key)
    if cached:
        QUERY_TOTAL.labels("hit").inc()
        QUERY_LATENCY.observe(time.perf_counter() - t0)
        return {**cached, "cached": True, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}

    state.syncer.maybe_sync()  # pick up documents uploaded via other replicas
    result = state.graph.invoke({"question": req.question, "org_id": user.org_id, "document_id": req.document_id})
    findings = _persist_findings(db, user.org_id, req.document_id, result.get("findings", []))
    audit(
        db,
        user.org_id,
        "agent",
        "langgraph",
        "query.answered",
        {
            "user_id": user.user_id,
            "document_id": req.document_id,
            "grounded": result.get("grounded", False),
            "n_findings": len(findings),
        },
    )
    db.commit()

    body = {
        "answer": result.get("answer", ""),
        "grounded": result.get("grounded", False),
        "citations": result.get("valid_citations", []),
        "invalid_citations": result.get("invalid_citations", []),
        "retrieved": result.get("retrieved", []),
        "findings": findings,
        "review_required": result.get("needs_review", False),
        "llm_provider": state.settings.llm_provider,
        "llm_model": result.get("llm_model", ""),
    }
    state.cache.set(key, body)
    QUERY_TOTAL.labels("miss").inc()
    QUERY_LATENCY.observe(time.perf_counter() - t0)
    return {**body, "cached": False, "latency_ms": round((time.perf_counter() - t0) * 1000, 2)}
