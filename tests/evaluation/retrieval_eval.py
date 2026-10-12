"""
retrieval_eval.py

Computes standard IR evaluation metrics against the golden question set:
Recall@K, Precision@K, MRR, NDCG@K, Hit Rate.

Methodology matches BEIR / MTEB conventions (see:
https://github.com/beir-cellar/beir) so the numbers here mean the same thing
they would in published retrieval benchmarks.

Usage:
    python retrieval_eval.py --data-dir data --k 5 --mode hybrid
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import List, Dict

from retrieval_engine import load_engine, RetrievalEngine


def _relevant_set(question: Dict) -> set:
    return set(question["relevant_chunk_ids"])


def recall_at_k(retrieved_ids: List[str], relevant: set) -> float:
    if not relevant:
        return 0.0
    hit = len(set(retrieved_ids) & relevant)
    return hit / len(relevant)


def precision_at_k(retrieved_ids: List[str], relevant: set) -> float:
    if not retrieved_ids:
        return 0.0
    hit = len(set(retrieved_ids) & relevant)
    return hit / len(retrieved_ids)


def reciprocal_rank(retrieved_ids: List[str], relevant: set) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: List[str], relevant: set) -> float:
    dcg = 0.0
    for i, cid in enumerate(retrieved_ids):
        rel = 1.0 if cid in relevant else 0.0
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    ideal_hits = min(len(relevant), len(retrieved_ids))
    idcg = sum((2 ** 1 - 1) / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit(retrieved_ids: List[str], relevant: set) -> float:
    return 1.0 if set(retrieved_ids) & relevant else 0.0


def evaluate(engine: RetrievalEngine, questions: List[Dict], k: int, mode: str,
             scoped: bool = True) -> Dict:
    """scoped=True mirrors the production API's document-scoped query mode
    (the realistic case: a reviewer asking about the document they have open).
    scoped=False searches the full corpus, which is the harder, more general
    case (a reviewer asking a question across their whole document library)."""
    per_question = []
    for q in questions:
        doc_id = q["document_id"] if scoped else None
        if mode == "dense":
            results = engine.dense_search(q["question"], top_k=k, document_id=doc_id)
        elif mode == "bm25":
            results = engine.bm25_search(q["question"], top_k=k, document_id=doc_id)
        else:
            results = engine.hybrid_search(q["question"], top_k=k, document_id=doc_id)

        retrieved_ids = [engine.get_chunk(idx).chunk_id for idx, _ in results]
        relevant = _relevant_set(q)

        per_question.append({
            "question_id": q["question_id"],
            "recall": recall_at_k(retrieved_ids, relevant),
            "precision": precision_at_k(retrieved_ids, relevant),
            "rr": reciprocal_rank(retrieved_ids, relevant),
            "ndcg": ndcg_at_k(retrieved_ids, relevant),
            "hit": hit(retrieved_ids, relevant),
        })

    n = len(per_question)
    agg = {
        "mode": mode,
        "scope": "document" if scoped else "corpus",
        "k": k,
        "n_questions": n,
        "recall_at_k": round(sum(p["recall"] for p in per_question) / n, 4),
        "precision_at_k": round(sum(p["precision"] for p in per_question) / n, 4),
        "mrr": round(sum(p["rr"] for p in per_question) / n, 4),
        "ndcg_at_k": round(sum(p["ndcg"] for p in per_question) / n, 4),
        "hit_rate": round(sum(p["hit"] for p in per_question) / n, 4),
        "status": "measured",
        "method": "Recall/Precision/MRR/NDCG computed against synthetic golden set with known "
                  "relevant-chunk labels; methodology follows BEIR/MTEB IR evaluation conventions.",
    }
    return {"aggregate": agg, "per_question": per_question}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--mode", type=str, default="hybrid", choices=["dense", "bm25", "hybrid"])
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    engine = load_engine(data_dir)
    questions = json.loads((data_dir / "golden_questions.json").read_text())

    results = {"document_scoped": {}, "corpus_wide": {}}
    for mode in ["dense", "bm25", "hybrid"]:
        results["document_scoped"][mode] = evaluate(engine, questions, args.k, mode, scoped=True)["aggregate"]
        results["corpus_wide"][mode] = evaluate(engine, questions, args.k, mode, scoped=False)["aggregate"]

    print(json.dumps(results, indent=2))
