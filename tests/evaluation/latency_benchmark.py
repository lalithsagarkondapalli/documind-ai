"""
latency_benchmark.py

Measures p50/p90/p95/p99 latency and throughput (requests/sec) for the
retrieval pipeline by actually calling it repeatedly and timing each call
with time.perf_counter (monotonic, high-resolution -- correct tool for
latency measurement, unlike time.time()).

Usage:
    python latency_benchmark.py --data-dir data --n-requests 300
"""
from __future__ import annotations
import argparse
import json
import random
import time
from pathlib import Path
from typing import List

import numpy as np

from retrieval_engine import load_engine


def percentile(values: List[float], p: float) -> float:
    return float(np.percentile(values, p))


def run_benchmark(data_dir: Path, n_requests: int, top_k: int = 5) -> dict:
    engine = load_engine(data_dir)
    questions = json.loads((data_dir / "golden_questions.json").read_text())
    random.seed(7)
    sample_questions = [random.choice(questions) for _ in range(n_requests)]

    latencies_ms = []
    t_start = time.perf_counter()
    for q in sample_questions:
        t0 = time.perf_counter()
        _ = engine.hybrid_search(q["question"], top_k=top_k, document_id=q["document_id"])
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)
    wall_time_s = time.perf_counter() - t_start

    result = {
        "n_requests": n_requests,
        "p50_ms": round(percentile(latencies_ms, 50), 3),
        "p90_ms": round(percentile(latencies_ms, 90), 3),
        "p95_ms": round(percentile(latencies_ms, 95), 3),
        "p99_ms": round(percentile(latencies_ms, 99), 3),
        "min_ms": round(min(latencies_ms), 3),
        "max_ms": round(max(latencies_ms), 3),
        "mean_ms": round(float(np.mean(latencies_ms)), 3),
        "throughput_rps": round(n_requests / wall_time_s, 2),
        "wall_time_s": round(wall_time_s, 3),
        "status": "measured",
        "scope": "retrieval-only latency (dense+BM25 hybrid search against in-memory index); "
                 "does NOT include LLM generation time or network latency -- run this against a "
                 "live deployed API endpoint (see load_test.py) for full end-to-end request latency.",
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--n-requests", type=int, default=300)
    args = ap.parse_args()

    result = run_benchmark(Path(args.data_dir), args.n_requests)
    print(json.dumps(result, indent=2))
