from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: str
    page_count: int
    n_chunks: int
    created_at: datetime


class QueryRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_id: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    page: int
    score: float


class FindingOut(BaseModel):
    id: str
    rule_id: str
    rule_name: str
    severity: str
    confidence: float
    rationale: str
    source_chunk_ids: list[str]
    status: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    citations: list[str]
    invalid_citations: list[str]
    retrieved: list[RetrievedChunk]
    findings: list[FindingOut]
    review_required: bool
    llm_provider: str
    llm_model: str
    cached: bool
    latency_ms: float


class ReviewDecision(BaseModel):
    decision: str = Field(pattern="^(approve|reject|edit)$")
    note: Optional[str] = Field(default=None, max_length=2000)
    severity: Optional[str] = Field(default=None, pattern="^(low|medium|high)$")


class ReviewItem(FindingOut):
    document_id: Optional[str]
    created_at: datetime
    reviewed_by: Optional[str] = None
    reviewer_note: Optional[str] = None
