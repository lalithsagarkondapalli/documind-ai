"""
vector_search_benchmark.py

Measures index build time, query latency, and memory usage of the in-memory
vector index. This uses the same local embedder as the rest of the
framework -- see retrieval_engine.py's EMBEDDING_PROVIDER note. Swap in a
real Qdrant/FAISS client here for infra-representative numbers; the timing
methodology (perf_counter, RSS memory via psutil) stays the same either way.

Usage:
    python vector_search_benchmark.py --data-dir data --n-queries 200
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import psutil

from retrieval_engine import load_engine, embed, Chunk, RetrievalEngine


def run(data_dir: Path, n_queries: int) -> dict:
    chunks_raw = json.loads((data_dir / "chunks.json").read_text())
    chunks = [Chunk(c["chunk_id"], c["document_id"], c["text"]) for c in chunks_raw]

    process = psutil.Process()
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    engine = RetrievalEngine(chunks)
    t0 = time.perf_counter()
    build_time_s = engine.build_index()
    mem_after_mb = process.memory_info().rss / (1024 * 1024)

    questions = json.loads((data_dir / "golden_questions.json").read_text())
    import random
    random.seed(3)
    sample = [random.choice(questions) for _ in range(n_queries)]

    query_latencies_ms = []
    for q in sample:
        t0 = time.perf_counter()
        _ = engine.dense_search(q["question"], top_k=5)
        query_latencies_ms.append((time.perf_counter() - t0) * 1000)

    embeddings_bytes = engine.embeddings.nbytes

    result = {
        "n_vectors": len(chunks),
        "embedding_dim": engine.embeddings.shape[1],
        "index_build_time_s": round(build_time_s, 4),
        "index_memory_mb": round(embeddings_bytes / (1024 * 1024), 3),
        "process_rss_delta_mb": round(mem_after_mb - mem_before_mb, 3),
        "n_queries": n_queries,
        "query_latency_p50_ms": round(float(np.percentile(query_latencies_ms, 50)), 4),
        "query_latency_p95_ms": round(float(np.percentile(query_latencies_ms, 95)), 4),
        "query_latency_p99_ms": round(float(np.percentile(query_latencies_ms, 99)), 4),
        "status": "measured",
        "scope": "in-memory numpy brute-force cosine search (no ANN index, no disk I/O, no network). "
                 "This is a lower bound for a real Qdrant/FAISS HNSW deployment under network+disk "
                 "conditions, and an upper bound for raw compute cost at this corpus size -- report both "
                 "boundaries rather than implying this equals production vector-DB latency.",
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--n-queries", type=int, default=200)
    args = ap.parse_args()
    print(json.dumps(run(Path(args.data_dir), args.n_queries), indent=2))
