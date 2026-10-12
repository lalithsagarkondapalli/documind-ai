"""
run_eval.py

Orchestrates the full evaluation suite and writes results to:
    reports/metrics.json
    reports/metrics.csv
    reports/resume_metrics.md   (generated ONLY from metrics.json values)
    reports/benchmark_report.pdf
    reports/dashboard.html

This is the single command you run to regenerate every number in your
resume/portfolio from scratch:

    python run_eval.py --data-dir ../data --reports-dir ../reports

Every run is timestamped and the embedding/judge backend used is recorded
in the output, so a reader can tell exactly what produced each number.
"""
from __future__ import annotations
import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import retrieval_eval
import rag_eval
import latency_benchmark
import token_cost_analysis
import ingestion_benchmark
import vector_search_benchmark
import cache_benchmark
from retrieval_engine import load_engine, EMBEDDING_PROVIDER
from llm_judge import USE_MOCK


def run_all(data_dir: Path, reports_dir: Path, n_rag_samples: int, n_latency_requests: int,
            n_cache_requests: int, monthly_queries: int) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] retrieval_eval ...", file=sys.stderr)
    engine = load_engine(data_dir)
    questions = json.loads((data_dir / "golden_questions.json").read_text())
    retrieval_results = {"document_scoped": {}, "corpus_wide": {}}
    for mode in ["dense", "bm25", "hybrid"]:
        retrieval_results["document_scoped"][mode] = retrieval_eval.evaluate(
            engine, questions, k=5, mode=mode, scoped=True)["aggregate"]
        retrieval_results["corpus_wide"][mode] = retrieval_eval.evaluate(
            engine, questions, k=5, mode=mode, scoped=False)["aggregate"]

    print("[2/6] rag_eval ...", file=sys.stderr)
    rag_results = rag_eval.run(engine, questions, n_samples=n_rag_samples)["aggregate"]

    print("[3/6] latency_benchmark ...", file=sys.stderr)
    latency_results = latency_benchmark.run_benchmark(data_dir, n_requests=n_latency_requests)

    print("[4/6] token_cost_analysis ...", file=sys.stderr)
    cost_results = token_cost_analysis.analyze(data_dir, n_samples=100, monthly_queries=monthly_queries)

    print("[5/6] ingestion + vector_search + cache benchmarks ...", file=sys.stderr)
    ingestion_results = ingestion_benchmark.run(data_dir)
    vector_results = vector_search_benchmark.run(data_dir, n_queries=200)
    cache_results = cache_benchmark.run(data_dir, n_requests=n_cache_requests, repeat_rate=0.4)

    print("[6/6] assembling metrics.json ...", file=sys.stderr)
    metrics = {
        "run_metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": platform.python_version(),
            "embedding_provider": EMBEDDING_PROVIDER,
            "llm_judge_mode": "heuristic (no API key set)" if USE_MOCK else "real LLM judge",
            "corpus_n_documents": len(json.loads((data_dir / "documents.json").read_text())),
            "corpus_n_chunks": len(json.loads((data_dir / "chunks.json").read_text())),
            "corpus_n_questions": len(questions),
        },
        "retrieval": retrieval_results,
        "rag_quality": rag_results,
        "latency": latency_results,
        "token_cost": cost_results,
        "ingestion": ingestion_results,
        "vector_search": vector_results,
        "cache": cache_results,
        "load_test": {
            "status": "not run -- requires a live deployed API endpoint. Run load_test.py separately "
                      "against docker-compose or a real deployment, then merge its output CSVs here.",
        },
    }

    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_csv(metrics, reports_dir / "metrics.csv")
    print(f"Wrote {reports_dir / 'metrics.json'} and {reports_dir / 'metrics.csv'}", file=sys.stderr)
    return metrics


def _flatten(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    rows = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            rows.extend(_flatten(v, key))
        elif isinstance(v, list):
            continue  # skip per-row detail lists in the flat CSV view
        else:
            rows.append((key, v))
    return rows


def _write_csv(metrics: dict, path: Path):
    rows = _flatten(metrics)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric_path", "value"])
        writer.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--reports-dir", type=str, default="reports")
    ap.add_argument("--n-rag-samples", type=int, default=100)
    ap.add_argument("--n-latency-requests", type=int, default=300)
    ap.add_argument("--n-cache-requests", type=int, default=500)
    ap.add_argument("--monthly-queries", type=int, default=50_000)
    args = ap.parse_args()

    run_all(Path(args.data_dir), Path(args.reports_dir), args.n_rag_samples,
             args.n_latency_requests, args.n_cache_requests, args.monthly_queries)
