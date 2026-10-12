"""
Hybrid retrieval engine: dense (hashed lexical embedding) + BM25, fused with
reciprocal rank fusion (RRF).

This is the single retrieval implementation. The API serves queries with it and
tests/evaluation scores it, so the numbers in the README describe the code that runs.

Embedding note: `embed()` is a deterministic hashed bag-of-words vectoriser
(the "hashing trick"), not a semantic model. It keeps the whole system runnable
and benchmarkable offline. Swap in a real embedding model behind `embed()` for
production-quality semantic recall; the retrieval mechanics do not change.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

EMBEDDING_DIM = 256
EMBEDDING_PROVIDER = "local-hashed-bow"
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "and",
    "or",
    "shall",
    "this",
    "for",
    "be",
    "is",
    "with",
    "under",
    "by",
    "on",
    "as",
    "at",
    "any",
    "all",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9%$']+", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def embed(text: str, dim: int = EMBEDDING_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokenize(text):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0 if (h // dim) % 2 == 0 else -1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def bm25_score(query_tokens: list[str], doc_tokens: list[str], avgdl: float, k1: float = 1.5, b: float = 0.75) -> float:
    """Okapi BM25 with a simplified constant IDF (fine for relative ranking at this scale)."""
    counts = Counter(doc_tokens)
    dl = len(doc_tokens)
    score = 0.0
    for qt in set(query_tokens):
        f = counts.get(qt, 0)
        if f:
            score += (f * (k1 + 1)) / max(f + k1 * (1 - b + b * dl / max(avgdl, 1e-6)), 1e-6)
    return score


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    org_id: str = "default"
    page_number: int = 1
    chunk_index: int = 0
    meta: dict = field(default_factory=dict)


class DenseBackend(Protocol):
    """Vector index keyed by the STABLE chunk_id (never a per-process list position), so several
    replicas can share one Qdrant server without ID collisions."""

    def upsert(self, chunk_ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None: ...
    def search(
        self, vector: np.ndarray, top_k: int, document_id: Optional[str], org_id: Optional[str]
    ) -> list[tuple[str, float]]: ...


class RetrievalEngine:
    def __init__(self, chunks: list[Chunk], dense_backend: Optional[DenseBackend] = None):
        self.chunks: list[Chunk] = []
        self.embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
        self.doc_tokens: list[list[str]] = []
        self._index: dict[str, int] = {}  # chunk_id -> position in self.chunks
        self.avgdl = 1.0
        self.dense_backend = dense_backend
        self._lock = threading.Lock()
        self._build_time_s: Optional[float] = None
        if chunks:
            self.add_chunks(chunks)

    # ---- indexing -------------------------------------------------------
    def add_chunks(self, new_chunks: list[Chunk], upsert_dense: bool = True) -> float:
        """Idempotent: chunks whose chunk_id is already indexed are skipped. `upsert_dense=False`
        is used when the vectors already live in a shared Qdrant server (another replica wrote them)."""
        t0 = time.perf_counter()
        with self._lock:
            new_chunks = [c for c in new_chunks if c.chunk_id not in self._index]
            start = len(self.chunks)
            new_vecs = (
                np.stack([embed(c.text) for c in new_chunks])
                if new_chunks
                else np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
            )
            self.chunks.extend(new_chunks)
            self._index.update({c.chunk_id: start + i for i, c in enumerate(new_chunks)})
            self.embeddings = np.vstack([self.embeddings, new_vecs])
            self.doc_tokens.extend(tokenize(c.text) for c in new_chunks)
            self.avgdl = sum(len(t) for t in self.doc_tokens) / max(len(self.doc_tokens), 1)
            if self.dense_backend is not None and upsert_dense and new_chunks:
                self.dense_backend.upsert(
                    [c.chunk_id for c in new_chunks],
                    new_vecs,
                    [{"document_id": c.document_id, "org_id": c.org_id} for c in new_chunks],
                )
        self._build_time_s = time.perf_counter() - t0
        return self._build_time_s

    def build_index(self) -> float:
        """Kept for the eval harness: returns the time spent indexing."""
        return self._build_time_s or 0.0

    # ---- search ---------------------------------------------------------
    def _candidates(self, document_id: Optional[str], org_id: Optional[str]) -> list[int]:
        return [
            i
            for i, c in enumerate(self.chunks)
            if (document_id is None or c.document_id == document_id) and (org_id is None or c.org_id == org_id)
        ]

    def dense_search(
        self, query: str, top_k: int = 20, document_id: Optional[str] = None, org_id: Optional[str] = None
    ) -> list[tuple[int, float]]:
        qv = embed(query)
        if self.dense_backend is not None:
            hits = self.dense_backend.search(qv, top_k, document_id, org_id)
            # a hit can reference a chunk this replica has not synced yet; skip it rather than fail
            return [(self._index[cid], score) for cid, score in hits if cid in self._index]
        cand = self._candidates(document_id, org_id)
        if not cand:
            return []
        sims = self.embeddings[cand] @ qv
        order = np.argsort(-sims)[:top_k]
        return [(cand[i], float(sims[i])) for i in order]

    def bm25_search(
        self, query: str, top_k: int = 20, document_id: Optional[str] = None, org_id: Optional[str] = None
    ) -> list[tuple[int, float]]:
        q_tokens = tokenize(query)
        scored = [
            (i, bm25_score(q_tokens, self.doc_tokens[i], self.avgdl)) for i in self._candidates(document_id, org_id)
        ]
        scored.sort(key=lambda kv: -kv[1])
        return scored[:top_k]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        fusion_k: int = 20,
        rrf_k: int = 60,
        document_id: Optional[str] = None,
        org_id: Optional[str] = None,
    ) -> list[tuple[int, float]]:
        """Reciprocal rank fusion of dense and BM25 rankings. `document_id` scopes the
        search to one document (the usual case: a reviewer asking about the open file);
        `org_id` enforces tenant isolation and is always set by the API from the JWT."""
        fused: dict[int, float] = {}
        for ranking in (
            self.dense_search(query, fusion_k, document_id, org_id),
            self.bm25_search(query, fusion_k, document_id, org_id),
        ):
            for rank, (idx, _) in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (rrf_k + rank + 1)
        return sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]

    def get_chunk(self, idx: int) -> Chunk:
        return self.chunks[idx]


def load_engine(data_dir: Path) -> RetrievalEngine:
    """Load the synthetic eval corpus (used by tests/evaluation)."""
    raw = json.loads((Path(data_dir) / "chunks.json").read_text())
    return RetrievalEngine([Chunk(c["chunk_id"], c["document_id"], c["text"]) for c in raw])
