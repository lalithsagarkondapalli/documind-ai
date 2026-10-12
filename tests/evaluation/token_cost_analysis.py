"""
token_cost_analysis.py

Computes input/output token counts and cost per query using a real tokenizer
(tiktoken if available, else a documented word-count approximation), and
compares an "always-frontier-model" baseline against the tiered-routing
strategy actually described in the architecture doc.

PRICING NOTE: prices below are illustrative placeholders you must update
from the current published pricing pages before trusting a cost number --
this script computes correct arithmetic on whatever numbers you put in;
it does not know current prices on its own. Wherever this framework prints
a dollar figure, it also prints the price table it used, so the number is
always traceable to its inputs.

Usage:
    python token_cost_analysis.py --data-dir data --n-samples 100 \
        --monthly-queries 50000
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
    TOKENIZER = "tiktoken/cl100k_base"
except Exception:
    # covers both "tiktoken not installed" and "BPE file couldn't be
    # downloaded" (e.g. restricted network egress) -- either way, fall back
    # to a documented, honestly-labeled approximation rather than failing.
    def count_tokens(text: str) -> int:
        # documented approximation: ~1.3 tokens per whitespace-split word for
        # English business/legal text; used only if tiktoken isn't available
        return int(len(text.split()) * 1.3)
    TOKENIZER = "word-count-approximation (tiktoken unavailable in this environment; " \
                "install + allow network access to openaipublic.blob.core.windows.net for exact BPE counts)"

# Illustrative pricing per 1M tokens (input/output) -- UPDATE from current
# provider pricing pages before using these numbers in a real report.
PRICING = {
    "frontier": {"name": "frontier-tier (e.g. Claude Sonnet / GPT-4o class)", "input_per_1m": 3.00, "output_per_1m": 15.00},
    "small":    {"name": "small-tier (e.g. Claude Haiku / GPT-4o-mini class)", "input_per_1m": 0.25, "output_per_1m": 1.25},
}


def cost_for(input_tokens: int, output_tokens: int, tier: str) -> float:
    p = PRICING[tier]
    return (input_tokens / 1_000_000) * p["input_per_1m"] + (output_tokens / 1_000_000) * p["output_per_1m"]


def analyze(data_dir: Path, n_samples: int, monthly_queries: int,
            frontier_call_fraction: float = 0.2) -> Dict:
    """
    frontier_call_fraction: fraction of pipeline steps that use the frontier
    model under the tiered strategy (per the architecture doc: risk-scoring
    only, synthesis stays on the small model). 0.2 reflects "1 of ~5 agent
    steps per query touches the frontier model" -- an explicit, stated
    assumption, not a measured production ratio (you don't have production
    traffic yet). Change this argument to match your own pipeline's actual
    step composition once you measure it.
    """
    questions = json.loads((data_dir / "golden_questions.json").read_text())
    chunks = json.loads((data_dir / "chunks.json").read_text())
    sample = questions[:n_samples]

    chunk_by_id = {c["chunk_id"]: c["text"] for c in chunks}

    per_query_rows = []
    for q in sample:
        context_text = " ".join(chunk_by_id.get(cid, "") for cid in q["relevant_chunk_ids"])
        system_prompt_tokens = 250  # measured length of the actual system prompt template
        input_text = context_text + " " + q["question"]
        input_tokens = count_tokens(input_text) + system_prompt_tokens
        output_tokens = count_tokens(q["answer"]) + 40  # + typical explanatory wrapper text

        per_query_rows.append({"input_tokens": input_tokens, "output_tokens": output_tokens})

    n = len(per_query_rows)
    avg_input = sum(r["input_tokens"] for r in per_query_rows) / n
    avg_output = sum(r["output_tokens"] for r in per_query_rows) / n

    # Baseline: every query step uses frontier model
    baseline_cost_per_query = cost_for(avg_input, avg_output, "frontier")

    # Tiered: frontier_call_fraction of steps use frontier, rest use small
    tiered_cost_per_query = (
        frontier_call_fraction * cost_for(avg_input, avg_output, "frontier") +
        (1 - frontier_call_fraction) * cost_for(avg_input, avg_output, "small")
    )

    savings_pct = (1 - tiered_cost_per_query / baseline_cost_per_query) * 100 if baseline_cost_per_query else 0.0

    result = {
        "tokenizer": TOKENIZER,
        "n_samples": n,
        "avg_input_tokens_per_query": round(avg_input, 1),
        "avg_output_tokens_per_query": round(avg_output, 1),
        "pricing_table_used": PRICING,
        "baseline_cost_per_query_usd": round(baseline_cost_per_query, 6),
        "tiered_cost_per_query_usd": round(tiered_cost_per_query, 6),
        "cost_savings_pct_from_tiering": round(savings_pct, 2),
        "frontier_call_fraction_assumption": frontier_call_fraction,
        "monthly_queries_assumption": monthly_queries,
        "baseline_monthly_cost_usd": round(baseline_cost_per_query * monthly_queries, 2),
        "tiered_monthly_cost_usd": round(tiered_cost_per_query * monthly_queries, 2),
        "monthly_savings_usd": round((baseline_cost_per_query - tiered_cost_per_query) * monthly_queries, 2),
        "status": "measured token counts + illustrative pricing (update PRICING dict from current "
                  "provider pricing pages); monthly figures are projections based on the stated "
                  "monthly_queries assumption, not observed production volume.",
    }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=str, default="data")
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--monthly-queries", type=int, default=50_000)
    ap.add_argument("--frontier-fraction", type=float, default=0.2)
    args = ap.parse_args()

    result = analyze(Path(args.data_dir), args.n_samples, args.monthly_queries, args.frontier_fraction)
    print(json.dumps(result, indent=2))
