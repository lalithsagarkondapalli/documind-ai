from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AppState, audit, get_db, get_state
from app.api.schemas import ReviewDecision, ReviewItem
from app.core.security import Principal, require_role
from app.db.models import RiskFinding

router = APIRouter(prefix="/v1/reviews", tags=["reviews"])


@router.get("/queue", response_model=list[ReviewItem])
def review_queue(user: Principal = Depends(require_role("reviewer")), db: Session = Depends(get_db)):
    return db.scalars(
        select(RiskFinding)
        .where(RiskFinding.org_id == user.org_id, RiskFinding.status == "pending_review")
        .order_by(RiskFinding.created_at)
    ).all()


@router.post("/{finding_id}/decision", response_model=ReviewItem)
def decide(
    finding_id: str,
    body: ReviewDecision,
    user: Principal = Depends(require_role("reviewer")),
    db: Session = Depends(get_db),
    state: AppState = Depends(get_state),
):
    f = db.scalar(select(RiskFinding).where(RiskFinding.id == finding_id, RiskFinding.org_id == user.org_id))
    if not f:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Finding not found")
    if f.status != "pending_review":
        raise HTTPException(status.HTTP_409_CONFLICT, f"Finding already {f.status}")
    before = {"status": f.status, "severity": f.severity}
    f.status = "rejected" if body.decision == "reject" else "approved"
    if body.decision == "edit" and body.severity:
        f.severity = body.severity
    f.reviewed_by, f.reviewed_at, f.reviewer_note = user.user_id, datetime.now(UTC), body.note
    audit(
        db,
        user.org_id,
        "human",
        user.user_id,
        f"finding.{body.decision}",
        {
            "finding_id": f.id,
            "before": before,
            "after": {"status": f.status, "severity": f.severity},
            "note": body.note,
        },
    )
    db.commit()
    state.cache.invalidate_prefix(f"q:{user.org_id}:")  # cached answers embed finding status
    return f
