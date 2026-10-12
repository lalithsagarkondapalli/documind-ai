"""
benchmark_dashboard.py

Builds a single self-contained HTML dashboard (Plotly, no server required)
from reports/metrics.json. Open the output file directly in a browser.

Usage:
    python benchmark_dashboard.py --metrics-file ../reports/metrics.json \
        --out ../reports/dashboard.html
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio


def _get(d, path, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_dashboard(m: dict) -> str:
    meta = m["run_metadata"]

    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            "Retrieval Quality (doc-scoped, hybrid)", "Retrieval Scope Comparison (Recall@5)",
            "RAG Quality", "Latency Percentiles (ms)", "Cache Hit Ratio",
            "Token Cost per Query (USD)", "Vector Search Latency (ms)",
            "Ingestion Throughput", "Model Cost Comparison",
        ),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "indicator"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "indicator"}, {"type": "bar"}]],
    )

    # 1. Retrieval quality
    hyb = _get(m, "retrieval.document_scoped.hybrid", {})
    metrics_names = ["Recall@5", "Precision@5", "MRR", "NDCG@5", "Hit Rate"]
    metrics_vals = [hyb.get("recall_at_k", 0), hyb.get("precision_at_k", 0), hyb.get("mrr", 0),
                     hyb.get("ndcg_at_k", 0), hyb.get("hit_rate", 0)]
    fig.add_trace(go.Bar(x=metrics_names, y=metrics_vals, marker_color="#2563eb"), row=1, col=1)

    # 2. Scope comparison
    doc_recall = _get(m, "retrieval.document_scoped.hybrid.recall_at_k", 0)
    corpus_recall = _get(m, "retrieval.corpus_wide.hybrid.recall_at_k", 0)
    fig.add_trace(go.Bar(x=["Document-scoped", "Corpus-wide"], y=[doc_recall, corpus_recall],
                          marker_color=["#16a34a", "#dc2626"]), row=1, col=2)

    # 3. RAG quality
    rag = m.get("rag_quality", {})
    rag_names = ["Faithfulness", "Context Precision", "Context Recall", "Answer Relevancy", "Citation Acc."]
    rag_vals = [rag.get("faithfulness", 0), rag.get("context_precision", 0), rag.get("context_recall", 0),
                rag.get("answer_relevancy", 0), rag.get("citation_accuracy", 0)]
    fig.add_trace(go.Bar(x=rag_names, y=rag_vals, marker_color="#7c3aed"), row=1, col=3)

    # 4. Latency percentiles
    lat = m.get("latency", {})
    lat_names = ["p50", "p90", "p95", "p99"]
    lat_vals = [lat.get("p50_ms", 0), lat.get("p90_ms", 0), lat.get("p95_ms", 0), lat.get("p99_ms", 0)]
    fig.add_trace(go.Bar(x=lat_names, y=lat_vals, marker_color="#ea580c"), row=2, col=1)

    # 5. Cache hit ratio (indicator gauge)
    cache_ratio = _get(m, "cache.cache_hit_ratio", 0) * 100
    fig.add_trace(go.Indicator(mode="gauge+number", value=cache_ratio,
                                gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#0891b2"}},
                                title={"text": "Cache Hit %"}), row=2, col=2)

    # 6. Token cost
    cost_names = ["Baseline (frontier-only)", "Tiered"]
    cost_vals = [_get(m, "token_cost.baseline_cost_per_query_usd", 0),
                 _get(m, "token_cost.tiered_cost_per_query_usd", 0)]
    fig.add_trace(go.Bar(x=cost_names, y=cost_vals, marker_color=["#dc2626", "#16a34a"]), row=2, col=3)

    # 7. Vector search latency
    vec_names = ["p50", "p95", "p99"]
    vec_vals = [_get(m, "vector_search.query_latency_p50_ms", 0), _get(m, "vector_search.query_latency_p95_ms", 0),
                _get(m, "vector_search.query_latency_p99_ms", 0)]
    fig.add_trace(go.Bar(x=vec_names, y=vec_vals, marker_color="#4338ca"), row=3, col=1)

    # 8. Ingestion throughput (indicator)
    chunks_per_sec = _get(m, "ingestion.embedding_chunks_per_sec", 0)
    fig.add_trace(go.Indicator(mode="number", value=chunks_per_sec,
                                title={"text": "Chunks embedded / sec"}), row=3, col=2)

    # 9. Model cost comparison per 1M tokens
    pricing = _get(m, "token_cost.pricing_table_used", {})
    tiers = list(pricing.keys())
    input_costs = [pricing[t]["input_per_1m"] for t in tiers]
    output_costs = [pricing[t]["output_per_1m"] for t in tiers]
    fig.add_trace(go.Bar(name="Input $/1M", x=tiers, y=input_costs, marker_color="#94a3b8"), row=3, col=3)
    fig.add_trace(go.Bar(name="Output $/1M", x=tiers, y=output_costs, marker_color="#334155"), row=3, col=3)

    fig.update_layout(
        height=1100, width=1400, showlegend=False,
        title_text=(f"DocuMind AI -- Benchmark Dashboard | Generated {meta['timestamp_utc']} | "
                     f"Embedding: {meta['embedding_provider']} | Judge: {meta['llm_judge_mode']}"),
        title_font_size=14,
    )

    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True)

    disclaimer = f"""
    <div style="font-family: -apple-system, sans-serif; max-width: 1400px; margin: 20px auto; padding: 16px;
                background:#fef9c3; border-left: 4px solid #ca8a04; font-size: 14px;">
      <strong>Read before citing these numbers:</strong> corpus = {meta['corpus_n_documents']} synthetic
      documents / {meta['corpus_n_chunks']} chunks / {meta['corpus_n_questions']} golden questions.
      Embedding provider = <code>{meta['embedding_provider']}</code> (a deterministic local lexical
      embedder, not a production semantic model). LLM judge mode = <code>{meta['llm_judge_mode']}</code>.
      See <code>metrics.json</code> for full per-metric "status" and "scope" notes -- several numbers
      here are explicitly labeled as simulations/projections, not observed production behavior.
    </div>
    """
    return html.replace("<body>", f"<body>{disclaimer}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-file", type=str, default="../reports/metrics.json")
    ap.add_argument("--out", type=str, default="../reports/dashboard.html")
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics_file).read_text())
    html = build_dashboard(metrics)
    Path(args.out).write_text(html)
    print(f"Wrote {args.out}")
