"""Keeps a replica's in-memory search index in step with the database.

The database is the source of truth; each replica's RetrievalEngine is a derived index.
Chunks get a monotonically increasing `seq`, so a replica only reads rows newer than its
high-water mark. A small overlap window is re-read on each pass because, on PostgreSQL,
concurrent transactions can commit slightly out of seq order; the engine de-duplicates by
chunk_id, so re-reading is harmless."""

from __future__ import annotations

import threading
import time

from sqlalchemy import select

from app.db.models import ChunkRow
from app.services.retrieval import Chunk, RetrievalEngine

OVERLAP_ROWS = 500


class IndexSyncer:
    def __init__(self, session_factory, engine: RetrievalEngine, min_interval: float = 2.0, upsert_dense: bool = True):
        self.session_factory, self.engine = session_factory, engine
        self.min_interval, self.upsert_dense = min_interval, upsert_dense
        self.last_seq = 0
        self._last_run = 0.0
        self._lock = threading.Lock()

    def maybe_sync(self) -> int:
        """Cheap to call on every request: does real work at most once per `min_interval`."""
        if time.monotonic() - self._last_run < self.min_interval:
            return 0
        return self.sync()

    def sync(self) -> int:
        with self._lock:
            self._last_run = time.monotonic()
            with self.session_factory() as db:
                rows = db.scalars(
                    select(ChunkRow).where(ChunkRow.seq > max(0, self.last_seq - OVERLAP_ROWS)).order_by(ChunkRow.seq)
                ).all()
            if not rows:
                return 0
            before = len(self.engine.chunks)
            self.engine.add_chunks(
                [Chunk(r.id, r.document_id, r.text, r.org_id, r.page_number, r.chunk_index) for r in rows],
                upsert_dense=self.upsert_dense,
            )
            self.last_seq = max(self.last_seq, rows[-1].seq)
            return len(self.engine.chunks) - before
