"""
llm_judge.py

LLM-as-judge abstraction used by rag_eval.py for faithfulness, correctness,
completeness, groundedness, toxicity, and citation-correctness scoring.

TWO MODES:
1. API mode (real): set ANTHROPIC_API_KEY or OPENAI_API_KEY and pass
   --judge-model <model-name>. Calls the real API and returns the model's
   structured judgment. This is what you'd use for real resume numbers.
2. Mock/heuristic mode (default, no key required): uses a transparent,
   deterministic lexical-overlap heuristic as a stand-in judge, so the whole
   framework can be run and validated with zero API cost/keys. Every score
   produced in mock mode is tagged "status": "estimated (heuristic judge,
   not LLM)" -- it is NEVER reported as a measured LLM-judged number. This
   distinction is enforced in the output schema, not just documentation,
   specifically so it can't accidentally leak into a resume bullet as if it
   came from a real model judgment.
"""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Optional

USE_MOCK = not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))


@dataclass
class JudgeResult:
    score: float          # 0.0 - 1.0
    label: str
    rationale: str
    status: str            # "measured" (real API judge) | "estimated" (heuristic fallback)
    judge_model: str


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9%$']+", text.lower()))


def _heuristic_groundedness(answer: str, context_chunks: list[str]) -> JudgeResult:
    """Lexical-overlap proxy for faithfulness/groundedness: what fraction of
    the answer's content tokens appear in the retrieved context. This is a
    coarse stand-in for a real LLM judge -- it will over- and under-score
    paraphrase in both directions -- which is exactly why it's labeled
    'estimated' and never allowed into a 'measured' resume claim."""
    ans_tokens = _tokens(answer)
    ctx_tokens = set()
    for c in context_chunks:
        ctx_tokens |= _tokens(c)
    if not ans_tokens:
        return JudgeResult(0.0, "ungrounded", "empty answer", "estimated", "heuristic-v1")
    overlap = len(ans_tokens & ctx_tokens) / len(ans_tokens)
    label = "grounded" if overlap >= 0.6 else ("partially_grounded" if overlap >= 0.3 else "ungrounded")
    return JudgeResult(round(overlap, 4), label,
                        f"{len(ans_tokens & ctx_tokens)}/{len(ans_tokens)} answer tokens found in retrieved context",
                        "estimated", "heuristic-v1")


def _heuristic_relevancy(question: str, answer: str) -> JudgeResult:
    q_tokens = _tokens(question)
    a_tokens = _tokens(answer)
    if not q_tokens:
        return JudgeResult(0.0, "irrelevant", "empty question", "estimated", "heuristic-v1")
    overlap = len(q_tokens & a_tokens) / len(q_tokens)
    label = "relevant" if overlap >= 0.3 else "off_topic"
    return JudgeResult(round(min(overlap * 1.5, 1.0), 4), label,
                        "keyword overlap between question and answer", "estimated", "heuristic-v1")


def judge_faithfulness(answer: str, context_chunks: list[str], judge_model: Optional[str] = None) -> JudgeResult:
    if USE_MOCK:
        return _heuristic_groundedness(answer, context_chunks)
    return _call_real_judge_api(
        task="faithfulness",
        prompt=_faithfulness_prompt(answer, context_chunks),
        judge_model=judge_model or "claude-sonnet-5",
    )


def judge_answer_relevancy(question: str, answer: str, judge_model: Optional[str] = None) -> JudgeResult:
    if USE_MOCK:
        return _heuristic_relevancy(question, answer)
    return _call_real_judge_api(
        task="answer_relevancy",
        prompt=_relevancy_prompt(question, answer),
        judge_model=judge_model or "claude-sonnet-5",
    )


def judge_citation_correctness(answer: str, cited_chunk_ids: list[str],
                                retrieved_chunk_ids: list[str]) -> JudgeResult:
    """This one does NOT need an LLM at all -- it's a hard, deterministic
    check: was every cited chunk ID actually part of the retrieved set?
    A citation to a chunk ID that was never retrieved is unambiguously wrong,
    so this is always 'measured', even in mock mode."""
    if not cited_chunk_ids:
        return JudgeResult(0.0, "no_citations", "answer made no citations", "measured", "n/a")
    valid = sum(1 for cid in cited_chunk_ids if cid in retrieved_chunk_ids)
    score = valid / len(cited_chunk_ids)
    label = "correct" if score == 1.0 else ("partial" if score > 0 else "fabricated")
    return JudgeResult(round(score, 4), label,
                        f"{valid}/{len(cited_chunk_ids)} citations reference chunks actually in the retrieved set",
                        "measured", "n/a")


def judge_toxicity(text: str) -> JudgeResult:
    """Simple deny-list heuristic; a real deployment should use a dedicated
    moderation model/API. This exists mainly to complete the metric surface
    the eval framework reports on."""
    deny_terms = {"hate", "kill", "attack", "racist", "slur"}
    hit = any(t in text.lower() for t in deny_terms)
    return JudgeResult(1.0 if not hit else 0.0, "clean" if not hit else "flagged",
                        "deny-list keyword scan (not a substitute for a real moderation model)",
                        "estimated", "denylist-v1")


def _faithfulness_prompt(answer: str, context_chunks: list[str]) -> str:
    ctx = "\n---\n".join(context_chunks)
    return (f"Context:\n{ctx}\n\nAnswer to evaluate:\n{answer}\n\n"
            "On a scale of 0.0 to 1.0, how well is every claim in the Answer supported by the Context? "
            "Respond as JSON: {\"score\": float, \"label\": str, \"rationale\": str}")


def _relevancy_prompt(question: str, answer: str) -> str:
    return (f"Question: {question}\nAnswer: {answer}\n\n"
            "On a scale of 0.0 to 1.0, how directly does the Answer address the Question? "
            "Respond as JSON: {\"score\": float, \"label\": str, \"rationale\": str}")


def _call_real_judge_api(task: str, prompt: str, judge_model: str) -> JudgeResult:
    """Real API call path. Implemented for Anthropic; swap client for OpenAI
    if OPENAI_API_KEY is set instead. Requires `pip install anthropic`."""
    import json
    try:
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=judge_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text
        parsed = json.loads(text[text.find("{"):text.rfind("}") + 1])
        return JudgeResult(float(parsed["score"]), parsed.get("label", ""),
                            parsed.get("rationale", ""), "measured", judge_model)
    except Exception as e:
        # Fail loud into an explicitly-labeled degraded result rather than
        # silently returning a fabricated score.
        return JudgeResult(0.0, "judge_error", f"real judge call failed: {e}", "error", judge_model)
