"""Query-result cache. Redis when REDIS_URL is set, otherwise an in-process TTL dict."""

from __future__ import annotations

import hashlib
import json
import time


def cache_key(org_id: str, document_id: str | None, question: str) -> str:
    norm = " ".join(question.lower().split())
    digest = hashlib.sha256(norm.encode()).hexdigest()[:24]
    return f"q:{org_id}:{document_id or '*'}:{digest}"


class QueryCache:
    def __init__(self, redis_url: str | None, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._local: dict[str, tuple[float, str]] = {}
        self._redis = None
        self.backend = "memory"
        if redis_url:
            import redis

            self._redis = redis.from_url(redis_url, socket_connect_timeout=1)
            self._redis.ping()  # fail loudly at startup if configured but unreachable
            self.backend = "redis"

    def get(self, key: str) -> dict | None:
        if self._redis is not None:
            raw = self._redis.get(key)
            return json.loads(raw) if raw else None
        item = self._local.get(key)
        if item and item[0] > time.time():
            return json.loads(item[1])
        self._local.pop(key, None)
        return None

    def set(self, key: str, value: dict) -> None:
        raw = json.dumps(value)
        if self._redis is not None:
            self._redis.set(key, raw, ex=self.ttl)
        else:
            self._local[key] = (time.time() + self.ttl, raw)

    def invalidate_prefix(self, prefix: str) -> None:
        """Called after a document is (re)indexed so stale answers are not served."""
        if self._redis is not None:
            for k in self._redis.scan_iter(f"{prefix}*"):
                self._redis.delete(k)
        else:
            for k in [k for k in self._local if k.startswith(prefix)]:
                del self._local[k]
