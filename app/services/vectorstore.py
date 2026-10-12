"""Qdrant-backed dense index. Uses a remote server when QDRANT_URL is set, otherwise
Qdrant's embedded local mode (the same client API, in-process), so it runs with no infra."""

from __future__ import annotations

import uuid
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient, models

from app.services.retrieval import EMBEDDING_DIM


class QdrantBackend:
    def __init__(
        self,
        url: Optional[str] = None,
        collection: str = "documind_chunks",
        dim: int = EMBEDDING_DIM,
        client: Optional[QdrantClient] = None,
    ):
        self.client = client or (QdrantClient(url=url) if url else QdrantClient(":memory:"))
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
                hnsw_config=models.HnswConfigDiff(m=16, ef_construct=128),
            )
            if url:  # payload indexes only apply to a server; embedded mode ignores them
                for field in ("document_id", "org_id"):
                    self.client.create_payload_index(collection, field, models.PayloadSchemaType.KEYWORD)

    @staticmethod
    def point_id(chunk_id: str) -> str:
        """Deterministic UUID so every replica maps the same chunk to the same Qdrant point."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"documind/{chunk_id}"))

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        self.client.upsert(
            self.collection,
            points=[
                models.PointStruct(id=self.point_id(cid), vector=v.tolist(), payload={**p, "chunk_id": cid})
                for cid, v, p in zip(chunk_ids, vectors, payloads, strict=True)
            ],
        )

    def search(
        self, vector: np.ndarray, top_k: int, document_id: Optional[str], org_id: Optional[str]
    ) -> list[tuple[str, float]]:
        must = []
        if document_id:
            must.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)))
        if org_id:
            must.append(models.FieldCondition(key="org_id", match=models.MatchValue(value=org_id)))
        res = self.client.query_points(
            self.collection, query=vector.tolist(), limit=top_k, query_filter=models.Filter(must=must) if must else None
        )
        return [(p.payload["chunk_id"], float(p.score)) for p in res.points]
