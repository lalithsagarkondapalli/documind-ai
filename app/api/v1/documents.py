from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import AppState, audit, get_db, get_state
from app.api.schemas import DocumentOut
from app.core.security import Principal, require_role
from app.db.models import ChunkRow, Document
from app.services.ingestion import UnsupportedDocument, chunk_pages, extract_pages
from app.services.retrieval import Chunk

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile,
    user: Principal = Depends(require_role("reviewer")),
    db: Session = Depends(get_db),
    state: AppState = Depends(get_state),
):
    data = await file.read()
    try:
        pages = extract_pages(file.filename or "upload", data)
    except UnsupportedDocument as e:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(e)) from e
    parsed = chunk_pages(pages)
    if not parsed:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "No text could be extracted")

    doc = Document(
        org_id=user.org_id,
        filename=file.filename or "upload",
        page_count=len(pages),
        n_chunks=len(parsed),
        uploaded_by=user.user_id,
    )
    db.add(doc)
    db.flush()
    chunks = [
        Chunk(
            chunk_id=f"{doc.id[:8]}_c{p.chunk_index:03d}",
            document_id=doc.id,
            text=p.text,
            org_id=user.org_id,
            page_number=p.page_number,
            chunk_index=p.chunk_index,
        )
        for p in parsed
    ]
    for c in chunks:
        db.add(
            ChunkRow(
                id=c.chunk_id,
                document_id=doc.id,
                org_id=c.org_id,
                page_number=c.page_number,
                chunk_index=c.chunk_index,
                text=c.text,
            )
        )
    audit(
        db,
        user.org_id,
        "human",
        user.user_id,
        "document.uploaded",
        {"document_id": doc.id, "filename": doc.filename, "n_chunks": len(chunks)},
    )
    db.commit()
    state.engine.add_chunks(chunks)
    state.cache.invalidate_prefix(f"q:{user.org_id}:")
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(user: Principal = Depends(require_role("viewer")), db: Session = Depends(get_db)):
    return db.scalars(select(Document).where(Document.org_id == user.org_id).order_by(Document.created_at.desc())).all()


@router.get("/{document_id}", response_model=DocumentOut)
def get_document(document_id: str, user: Principal = Depends(require_role("viewer")), db: Session = Depends(get_db)):
    doc = db.scalar(select(Document).where(Document.id == document_id, Document.org_id == user.org_id))
    if not doc:  # same 404 whether it doesn't exist or belongs to another tenant
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    return doc
