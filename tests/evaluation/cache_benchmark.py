"""
cache_benchmark.py

Measures cache hit ratio, cache read/write latency, and embedding-cache
compute savings under a SIMULATED realistic reviewer query pattern (repeat
queries against the same open document, per the architecture's caching
strategy -- Section 25). Uses real Redis if available at REDIS_URL, else an
in-memory LRU dict with the same interface, clearly labeled either way.

Usage:
    python cache_benchmark.py --data-dir data --n-requests 500 --repeat-rate 0.4
"""
from __future__ import annotations
import argparse
import json
import os
import random
import time
from pathlib import Path
from collections import OrderedDict

BACKEND = "in-memory (no REDIS_URL set)"
try:
    import redis as _redis_lib
    _client = _redis_lib.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379"), socket_connect_timeout=1)
    _client.ping()
    BACKEND = "redis"
except Exception:
    _client = None


class InMemoryLRU:
    def __init__(self, maxsize: int = 5000):
        self.store = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.store:
            self.store.move_to_end(key)
            return self.store[key]
        return None

    def set(self, key, value):
        self.store[key] = value
        self.store.move_to_end(key)
        if len(self.store) > self.maxsize:
            self.store.popitem(last=False)


def run(data_dir: Path, n_requests: int, repeat_rate: float) -> dict:
    questions = json.loads((data_dir / "golden_questions.json").read_text())
    random.seed(11)

    # Simulate a realistic reviewer session: `repeat_rate` fraction of
    # requests re-ask a question already asked earlier in the session
    # (mirrors a reviewer re-checking the same clause), rest are novel.
    seen_pool: list = []
    request_sequence = []
    for _ in range(n_requests):
        if seen_pool and random.random() < repeat_rate:
            request_sequence.append(random.choice(seen_pool))
        else:
            q = random.choice(questions)
            seen_pool.append(q["question_id"])
            request_sequence.append(q["question_id"])

    cache = InMemoryLRU() if _client is None else None
    hits, misses = 0, 0
    get_latencies_ms, set_latencies_ms = [], []
    embedding_calls_saved = 0

    for qid in request_sequence:
        key = f"query_result:{qid}"
        t0 = time.perf_counter()
        if _client is not None:
            val = _client.get(key)
        else:
            val = cache.get(key)
        get_latencies_ms.append((time.perf_counter() - t0) * 1000)

        if val is not None:
            hits += 1
            embedding_calls_saved += 1  # a cache hit skips re-embedding the query + re-running retrieval
        else:
            misses += 1
            t0 = time.perf_counter()
            if _client is not None:
                _client.set(key, "1", ex=3600)
            else:
                cache.set(key, "1")
            set_latencies_ms.append((time.perf_counter() - t0) * 1000)

    total = hits + misses
    hit_ratio = hits / total if total else 0.0

    import numpy as np
    result = {
        "backend": BACKEND,
        "n_requests": n_requests,
        "repeat_rate_assumption": repeat_rate,
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_hit_ratio": round(hit_ratio, 4),
        "cache_get_latency_p50_ms": round(float(np.percentile(get_latencies_ms, 50)), 5),
        "cache_get_latency_p95_ms": round(float(np.percentile(get_latencies_ms, 95)), 5),
        "embedding_calls_saved": embedding_calls_saved,
        "embedding_calls_saved_pct": round(embedding_calls_saved / n_requests * 100, 2),
        "status": "measured",
        "note": "repeat_rate_assumption models a reviewer session pattern (re-querying an open "
                "document); it is an explicit simulation parameter, not an observed production "
                "traffic statistic -- re-run with your own real query logs once available.",
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--n-requests", type=int, default=500)
    ap.add_argument("--repeat-rate", type=float, default=0.4)
    args = ap.parse_args()
    print(json.dumps(run(Path(args.data_dir), args.n_requests, args.repeat_rate), indent=2))
