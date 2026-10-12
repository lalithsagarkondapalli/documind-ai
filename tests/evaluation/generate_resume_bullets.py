"""
generate_resume_bullets.py

Generates resume_metrics.md using ONLY values read out of metrics.json.
No number in this file is typed by a human or an LLM guess -- every bullet
is an f-string templated directly against a dict key. If a metric is
missing or marked "not measured", the corresponding bullet is skipped
entirely rather than filled with a placeholder, so nothing ungrounded can
end up in the output.

Usage:
    python generate_resume_bullets.py --metrics-file ../reports/metrics.json \
        --out ../reports/resume_metrics.md
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _get(d: dict, path: str, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_bullets(m: dict) -> list[str]:
    bullets = []
    meta = m["run_metadata"]

    recall = _get(m, "retrieval.document_scoped.hybrid.recall_at_k")
    ndcg = _get(m, "retrieval.document_scoped.hybrid.ndcg_at_k")
    mrr = _get(m, "retrieval.document_scoped.hybrid.mrr")
    k = _get(m, "retrieval.document_scoped.hybrid.k")
    n_q = _get(m, "retrieval.document_scoped.hybrid.n_questions")
    if recall is not None:
        bullets.append(
            f"Built a hybrid (dense + BM25 with reciprocal rank fusion) retrieval pipeline achieving "
            f"{recall*100:.1f}% Recall@{k} and {ndcg:.3f} NDCG@{k} on a {n_q}-question self-built "
            f"evaluation set (embedding provider: {meta['embedding_provider']})."
        )

    corpus_recall = _get(m, "retrieval.corpus_wide.hybrid.recall_at_k")
    if corpus_recall is not None and recall is not None:
        bullets.append(
            f"Measured and documented the retrieval-scope trade-off between document-scoped queries "
            f"({recall*100:.1f}% Recall@{k}) and full corpus-wide search ({corpus_recall*100:.1f}% "
            f"Recall@{k}), informing the production API's document-scoped query design."
        )

    faithfulness = _get(m, "rag_quality.faithfulness")
    citation_acc = _get(m, "rag_quality.citation_accuracy")
    hallu_rate = _get(m, "rag_quality.hallucination_rate_observed")
    hallu_precision = _get(m, "rag_quality.hallucination_detector_precision")
    judge_status = _get(m, "rag_quality.judge_status", "")
    if citation_acc is not None:
        bullets.append(
            f"Implemented deterministic citation validation (every cited chunk id checked against the "
            f"retrieved set); on {_get(m, 'rag_quality.n_samples')} synthetic queries it flagged "
            f"{(1 - citation_acc)*100:.1f}% of answers as lacking a valid citation, matching the "
            f"{hallu_rate*100:.1f}% that were deliberately fabricated (a validator test, not a model-accuracy claim)."
        )
    if hallu_precision is not None:
        bullets.append(
            f"Built a hallucination-detection layer combining faithfulness scoring with citation "
            f"validation, achieving {hallu_precision*100:.1f}% precision against a controlled set of "
            f"known-injected hallucinations used to validate the detector itself."
        )

    p95 = _get(m, "latency.p95_ms")
    rps = _get(m, "latency.throughput_rps")
    n_req = _get(m, "latency.n_requests")
    if p95 is not None:
        bullets.append(
            f"Benchmarked retrieval-pipeline latency at {p95:.2f}ms p95 across {n_req} requests "
            f"({rps:.0f} req/s throughput), with full p50/p90/p95/p99 profiling "
            f"(retrieval-only scope -- see metrics.json for boundary conditions)."
        )

    savings = _get(m, "token_cost.cost_savings_pct_from_tiering")
    monthly_savings = _get(m, "token_cost.monthly_savings_usd")
    monthly_q = _get(m, "token_cost.monthly_queries_assumption")
    if savings is not None:
        bullets.append(
            f"Designed a model-tiering strategy (small model for synthesis, frontier model reserved for "
            f"high-stakes risk scoring), reducing blended per-query LLM token cost by {savings:.1f}% "
            f"versus an all-frontier-model baseline (projected ${monthly_savings:,.2f}/month at "
            f"{monthly_q:,} queries/month -- explicitly a projection, not observed production spend)."
        )

    cache_ratio = _get(m, "cache.cache_hit_ratio")
    if cache_ratio is not None:
        bullets.append(
            f"Implemented query-result caching achieving a {cache_ratio*100:.1f}% hit ratio under a "
            f"simulated reviewer session pattern (repeat-query rate: "
            f"{_get(m, 'cache.repeat_rate_assumption')*100:.0f}%), backend: {_get(m, 'cache.backend')}."
        )

    embed_speed = _get(m, "ingestion.embedding_chunks_per_sec")
    if embed_speed is not None:
        bullets.append(
            f"Profiled document-ingestion pipeline throughput at {embed_speed:,.0f} chunks/sec for "
            f"embedding generation across {_get(m, 'ingestion.n_documents')} synthetic documents "
            f"({_get(m, 'ingestion.total_pages')} total pages) -- see metrics.json for full pipeline "
            f"scope notes (excludes PDF extraction/OCR time)."
        )

    vec_p95 = _get(m, "vector_search.query_latency_p95_ms")
    n_vec = _get(m, "vector_search.n_vectors")
    if vec_p95 is not None:
        bullets.append(
            f"Benchmarked in-memory vector search at {vec_p95:.3f}ms p95 query latency over "
            f"{n_vec} indexed chunks, with index build time of "
            f"{_get(m, 'vector_search.index_build_time_s'):.3f}s (see metrics.json for production-"
            f"scale extrapolation caveats)."
        )

    return bullets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-file", type=str, default="../reports/metrics.json")
    ap.add_argument("--out", type=str, default="../reports/resume_metrics.md")
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics_file).read_text())
    bullets = build_bullets(metrics)
    meta = metrics["run_metadata"]

    lines = [
        "# DocuMind AI -- Resume Metrics (auto-generated, do not hand-edit numbers)",
        "",
        f"Generated from `{Path(args.metrics_file).name}` at {meta['timestamp_utc']}.",
        f"Embedding provider: `{meta['embedding_provider']}` | LLM judge mode: `{meta['llm_judge_mode']}`",
        f"Corpus: {meta['corpus_n_documents']} documents, {meta['corpus_n_chunks']} chunks, "
        f"{meta['corpus_n_questions']} golden questions.",
        "",
        "**Every number below was computed directly from metrics.json by generate_resume_bullets.py. "
        "No number was hand-typed. If you re-run run_eval.py, this file regenerates and numbers may "
        "shift slightly (that's expected -- they're measured, not fixed).**",
        "",
        "## Resume bullets",
        "",
    ]
    for b in bullets:
        lines.append(f"- {b}")

    lines += [
        "",
        "## Before publishing",
        "- These bullets describe a personal portfolio project, not production deployment -- keep that "
        "framing in your resume/interview answers.",
        "- Several metrics are explicitly labeled projections or simulations (cache repeat-rate, monthly "
        "cost) -- say so if asked; the interview-questions doc covers exactly this.",
        "- Re-run with a real embedding model / LLM judge (set EMBEDDING_PROVIDER and an API key) before "
        "using these numbers as production-representative rather than framework-validation numbers.",
    ]

    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out} ({len(bullets)} bullets)")


if __name__ == "__main__":
    main()
