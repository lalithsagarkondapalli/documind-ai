"""
pdf_report.py

Generates benchmark_report.pdf from metrics.json using reportlab.

Usage:
    python pdf_report.py --metrics-file ../reports/metrics.json \
        --out ../reports/benchmark_report.pdf
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak)


def _get(d, path, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def build_pdf(m: dict, out_path: Path):
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=20)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontSize=8, textColor=colors.grey)

    doc = SimpleDocTemplate(str(out_path), pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    story = []
    meta = m["run_metadata"]

    story.append(Paragraph("DocuMind AI -- Benchmark Report", title_style))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated: {meta['timestamp_utc']} | Python {meta['python_version']} | "
        f"Embedding provider: {meta['embedding_provider']} | LLM judge: {meta['llm_judge_mode']}",
        small))
    story.append(Paragraph(
        f"Corpus: {meta['corpus_n_documents']} synthetic documents, {meta['corpus_n_chunks']} chunks, "
        f"{meta['corpus_n_questions']} golden questions.", small))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "All figures in this report are computed directly from the evaluation harness run referenced "
        "above and are reproducible by re-running run_eval.py. Numbers marked 'estimated' used a "
        "heuristic fallback (no LLM API key was set for this run) and are explicitly distinguished "
        "from 'measured' numbers throughout. This is a personal portfolio project benchmark, not a "
        "report of production system performance.", body))

    def section(title, rows, headers=("Metric", "Value")):
        story.append(Paragraph(title, h2))
        table_data = [list(headers)] + rows
        t = Table(table_data, colWidths=[3.3 * inch, 3.3 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(t)
        story.append(Spacer(1, 8))

    hyb = _get(m, "retrieval.document_scoped.hybrid", {})
    section("Retrieval Quality (document-scoped, hybrid dense+BM25)", [
        [f"Recall@{hyb.get('k')}", f"{hyb.get('recall_at_k', 0)*100:.1f}%"],
        [f"Precision@{hyb.get('k')}", f"{hyb.get('precision_at_k', 0)*100:.1f}%"],
        ["MRR", f"{hyb.get('mrr', 0):.4f}"],
        [f"NDCG@{hyb.get('k')}", f"{hyb.get('ndcg_at_k', 0):.4f}"],
        ["Hit Rate", f"{hyb.get('hit_rate', 0)*100:.1f}%"],
        ["Corpus-wide Recall@5 (harder task, for comparison)",
         f"{_get(m, 'retrieval.corpus_wide.hybrid.recall_at_k', 0)*100:.1f}%"],
    ])

    rag = m.get("rag_quality", {})
    section("RAG Quality", [
        ["Faithfulness", f"{rag.get('faithfulness', 0):.4f} ({rag.get('judge_status', '')})"],
        ["Context Precision", f"{rag.get('context_precision', 0)*100:.1f}%"],
        ["Context Recall", f"{rag.get('context_recall', 0)*100:.1f}%"],
        ["Answer Relevancy", f"{rag.get('answer_relevancy', 0):.4f}"],
        ["Citation Accuracy (deterministic)", f"{rag.get('citation_accuracy', 0)*100:.1f}%"],
        ["Hallucination Detector Precision", f"{rag.get('hallucination_detector_precision', 0)*100:.1f}%"],
        ["Hallucination Detector Recall", f"{rag.get('hallucination_detector_recall', 0)*100:.1f}%"],
    ])

    lat = m.get("latency", {})
    section("Latency (retrieval-only)", [
        ["p50", f"{lat.get('p50_ms', 0):.3f} ms"],
        ["p90", f"{lat.get('p90_ms', 0):.3f} ms"],
        ["p95", f"{lat.get('p95_ms', 0):.3f} ms"],
        ["p99", f"{lat.get('p99_ms', 0):.3f} ms"],
        ["Throughput", f"{lat.get('throughput_rps', 0):.1f} req/s"],
    ])

    cost = m.get("token_cost", {})
    section("Token Cost", [
        ["Avg input tokens/query", f"{cost.get('avg_input_tokens_per_query', 0):.1f}"],
        ["Avg output tokens/query", f"{cost.get('avg_output_tokens_per_query', 0):.1f}"],
        ["Baseline cost/query (frontier-only)", f"${cost.get('baseline_cost_per_query_usd', 0):.6f}"],
        ["Tiered cost/query", f"${cost.get('tiered_cost_per_query_usd', 0):.6f}"],
        ["Cost savings from tiering", f"{cost.get('cost_savings_pct_from_tiering', 0):.1f}%"],
        [f"Projected monthly cost @ {cost.get('monthly_queries_assumption', 0):,} queries",
         f"${cost.get('tiered_monthly_cost_usd', 0):,.2f}"],
    ])

    cache = m.get("cache", {})
    section("Cache Performance", [
        ["Backend", cache.get("backend", "")],
        ["Hit ratio", f"{cache.get('cache_hit_ratio', 0)*100:.1f}%"],
        ["Simulated repeat-query rate", f"{cache.get('repeat_rate_assumption', 0)*100:.0f}%"],
        ["Embedding calls saved", f"{cache.get('embedding_calls_saved_pct', 0):.1f}%"],
    ])

    story.append(PageBreak())
    story.append(Paragraph("Scope Notes & Honesty Disclosures", h2))
    for section_name, path in [("Retrieval", "retrieval.document_scoped.hybrid"),
                                 ("Latency", "latency"), ("Ingestion", "ingestion"),
                                 ("Vector Search", "vector_search"), ("Cache", "cache")]:
        note = _get(m, f"{path}.scope") or _get(m, f"{path}.note") or _get(m, f"{path}.method")
        if note:
            story.append(Paragraph(f"<b>{section_name}:</b> {note}", body))
            story.append(Spacer(1, 6))

    doc.build(story)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-file", type=str, default="../reports/metrics.json")
    ap.add_argument("--out", type=str, default="../reports/benchmark_report.pdf")
    args = ap.parse_args()

    metrics = json.loads(Path(args.metrics_file).read_text())
    build_pdf(metrics, Path(args.out))
    print(f"Wrote {args.out}")
