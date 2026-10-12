"""
ingestion_benchmark.py

Times the actual local ingestion steps (chunk-text prep + embedding
generation) over the synthetic corpus and reports pages/minute, docs/hour,
and embedding throughput.

OCR NOTE: this framework doesn't include real scanned-image OCR (the
synthetic corpus is text-native), so OCR speed here is reported as
"not measured -- requires real scanned samples" rather than a fabricated
number. If you want a real OCR speed metric, run pytesseract over a small
set of actual scanned PDFs and plug the result into metrics.json directly --
see the README section on OCR benchmarking.

Usage:
    python ingestion_benchmark.py --data-dir data
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

from retrieval_engine import embed


def run(data_dir: Path) -> dict:
    documents = json.loads((data_dir / "documents.json").read_text())
    chunks = json.loads((data_dir / "chunks.json").read_text())

    total_pages = sum(d["page_count"] for d in documents)
    n_docs = len(documents)

    # Time chunk-level "parsing" (here: text already extracted, so this times
    # the tokenization/normalization step that would follow real PDF text
    # extraction) + embedding generation, which is the real per-chunk cost.
    t0 = time.perf_counter()
    for c in chunks:
        _ = embed(c["text"])
    embed_time_s = time.perf_counter() - t0

    pages_per_minute = total_pages / (embed_time_s / 60) if embed_time_s > 0 else float("inf")
    docs_per_hour = n_docs / (embed_time_s / 3600) if embed_time_s > 0 else float("inf")
    chunks_per_sec = len(chunks) / embed_time_s if embed_time_s > 0 else float("inf")

    result = {
        "n_documents": n_docs,
        "total_pages": total_pages,
        "n_chunks": len(chunks),
        "embedding_time_s": round(embed_time_s, 4),
        "embedding_chunks_per_sec": round(chunks_per_sec, 2),
        "pages_per_minute": round(pages_per_minute, 2),
        "documents_per_hour": round(docs_per_hour, 2),
        "status": "measured",
        "scope": "measures local embedding-generation throughput only (text already extracted). "
                 "Does NOT include: PDF text-extraction time (PyMuPDF), network latency to an "
                 "embeddings API if using a hosted model, or OCR time for scanned documents. "
                 "OCR speed/accuracy requires real scanned samples and is intentionally not "
                 "fabricated here -- see README.",
        "ocr_speed_pages_per_min": None,
        "ocr_accuracy_cer": None,
        "ocr_status": "not measured -- requires a real scanned-document sample set (see README)",
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    args = ap.parse_args()
    print(json.dumps(run(Path(args.data_dir)), indent=2))
