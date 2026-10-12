"""
rag_eval.py

End-to-end RAG quality metrics: Faithfulness, Context Precision, Context
Recall, Answer Relevancy, Hallucination Rate, Citation Accuracy.

This simulates the full query path (retrieve -> synthesize answer -> judge)
using a template-based "synthesizer" in place of a live LLM call, so the
framework runs offline. Swap `synthesize_answer()` for a real LLM call
(OpenAI/Anthropic) to get production-representative synthesis quality --
the metric *computation* code does not change either way.

Usage:
    python rag_eval.py --data-dir data --n-samples 100
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from retrieval_engine import load_engine, RetrievalEngine
from llm_judge import judge_faithfulness, judge_answer_relevancy, judge_citation_correctness, USE_MOCK

random.seed(42)


def synthesize_answer(question: str, retrieved_texts: List[str], retrieved_ids: List[str],
                       correct_answer: str, hallucination_injection_rate: float = 0.1) -> Dict:
    """
    Template synthesizer standing in for a real LLM call. Injects a controlled
    hallucination on a known fraction of samples so hallucination-DETECTION
    can itself be validated against ground truth (does the eval correctly
    catch the cases we know are wrong?) rather than trusting the judge blindly.
    """
    inject_hallucination = random.random() < hallucination_injection_rate

    if inject_hallucination or not retrieved_texts:
        fabricated = "approximately $10,000,000 (unverified)"
        return {
            "answer": f"Based on the document, the answer is {fabricated}.",
            "cited_chunk_ids": [],  # a real hallucination typically cites nothing real, or cites wrong chunk
            "ground_truth_hallucination": True,
        }

    best_text = retrieved_texts[0]
    cited = [retrieved_ids[0]]
    answer = f"According to the document, {best_text.strip().rstrip('.')}."
    return {"answer": answer, "cited_chunk_ids": cited, "ground_truth_hallucination": False}


def context_precision(retrieved_ids: List[str], relevant_ids: set) -> float:
    if not retrieved_ids:
        return 0.0
    return len(set(retrieved_ids) & relevant_ids) / len(retrieved_ids)


def context_recall(retrieved_ids: List[str], relevant_ids: set) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids) & relevant_ids) / len(relevant_ids)


def run(engine: RetrievalEngine, questions: List[Dict], n_samples: int, top_k: int = 5) -> Dict:
    sample = questions if len(questions) <= n_samples else random.sample(questions, n_samples)
    rows = []

    for q in sample:
        results = engine.hybrid_search(q["question"], top_k=top_k, document_id=q["document_id"])
        retrieved_ids = [engine.get_chunk(i).chunk_id for i, _ in results]
        retrieved_texts = [engine.get_chunk(i).text for i, _ in results]
        relevant = set(q["relevant_chunk_ids"])

        synth = synthesize_answer(q["question"], retrieved_texts, retrieved_ids, q["answer"])

        faithfulness = judge_faithfulness(synth["answer"], retrieved_texts)
        relevancy = judge_answer_relevancy(q["question"], synth["answer"])
        citation = judge_citation_correctness(synth["answer"], synth["cited_chunk_ids"], retrieved_ids)

        # hallucination flag: low faithfulness AND/OR fabricated citation
        predicted_hallucination = (faithfulness.score < 0.4) or (citation.label == "fabricated")

        rows.append({
            "question_id": q["question_id"],
            "context_precision": round(context_precision(retrieved_ids, relevant), 4),
            "context_recall": round(context_recall(retrieved_ids, relevant), 4),
            "faithfulness": faithfulness.score,
            "answer_relevancy": relevancy.score,
            "citation_accuracy": citation.score,
            "predicted_hallucination": predicted_hallucination,
            "ground_truth_hallucination": synth["ground_truth_hallucination"],
        })

    n = len(rows)
    tp = sum(1 for r in rows if r["predicted_hallucination"] and r["ground_truth_hallucination"])
    fp = sum(1 for r in rows if r["predicted_hallucination"] and not r["ground_truth_hallucination"])
    fn = sum(1 for r in rows if not r["predicted_hallucination"] and r["ground_truth_hallucination"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    judge_status = "estimated (heuristic judge, no LLM API key set)" if USE_MOCK else "measured (real LLM judge)"

    aggregate = {
        "n_samples": n,
        "context_precision": round(sum(r["context_precision"] for r in rows) / n, 4),
        "context_recall": round(sum(r["context_recall"] for r in rows) / n, 4),
        "faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 4),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in rows) / n, 4),
        "citation_accuracy": round(sum(r["citation_accuracy"] for r in rows) / n, 4),
        "hallucination_rate_observed": round(sum(1 for r in rows if r["ground_truth_hallucination"]) / n, 4),
        "hallucination_detector_precision": round(precision, 4),
        "hallucination_detector_recall": round(recall, 4),
        "judge_status": judge_status,
        "note": "context_precision/context_recall/citation_accuracy are deterministic and always 'measured'. "
                "faithfulness/answer_relevancy depend on the judge backend -- see judge_status.",
    }
    return {"aggregate": aggregate, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--n-samples", type=int, default=100)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    engine = load_engine(data_dir)
    questions = json.loads((data_dir / "golden_questions.json").read_text())

    result = run(engine, questions, args.n_samples)
    print(json.dumps(result["aggregate"], indent=2))
