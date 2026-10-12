"""LangGraph agent pipeline:

    retrieve -> synthesize -> validate_citations -> score_risk -> (needs review?) -> human_review -> finalize
                                                             -> (no)            -> finalize

Each node is a plain function over a typed state, so it can be tested and evaluated in
isolation, and each is wrapped in an OpenTelemetry span."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.core.tracing import traced
from app.services.llm import LLM, extract_citations
from app.services.retrieval import RetrievalEngine
from app.services.rules import DEFAULT_RULES, match_rule

RULE_MATCH_MIN = 0.5  # a chunk must cover at least half of a rule's keywords to count


class AgentState(TypedDict, total=False):
    question: str
    org_id: str
    document_id: Optional[str]
    retrieved: list[dict]  # [{chunk_id, text, page, score}]
    answer: str
    citations: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    grounded: bool
    findings: list[dict]
    needs_review: bool
    llm_model: str
    tokens: dict


@dataclass
class Deps:
    engine: RetrievalEngine
    llm: LLM
    review_threshold: float = 0.85
    top_k: int = 5
    rules: tuple = tuple(DEFAULT_RULES)


def build_graph(deps: Deps):
    @traced("agent.retrieve")
    def retrieve(state: AgentState) -> AgentState:
        hits = deps.engine.hybrid_search(
            state["question"], top_k=deps.top_k, document_id=state.get("document_id"), org_id=state["org_id"]
        )
        retrieved = []
        for idx, score in hits:
            c = deps.engine.get_chunk(idx)
            retrieved.append(
                {
                    "chunk_id": c.chunk_id,
                    "document_id": c.document_id,
                    "text": c.text,
                    "page": c.page_number,
                    "score": round(score, 5),
                }
            )
        return {"retrieved": retrieved}

    @traced("agent.synthesize")
    def synthesize(state: AgentState) -> AgentState:
        ctx = [(r["chunk_id"], r["text"]) for r in state["retrieved"]]
        res = deps.llm.synthesize(state["question"], ctx)
        return {
            "answer": res.text,
            "citations": extract_citations(res.text),
            "llm_model": res.model,
            "tokens": {"input": res.input_tokens, "output": res.output_tokens},
        }

    @traced("agent.validate_citations")
    def validate_citations(state: AgentState) -> AgentState:
        """Deterministic: every cited chunk id must be in the retrieved set. No LLM involved."""
        retrieved_ids = {r["chunk_id"] for r in state["retrieved"]}
        cites = state.get("citations", [])
        valid = [c for c in cites if c in retrieved_ids]
        invalid = [c for c in cites if c not in retrieved_ids]
        return {"valid_citations": valid, "invalid_citations": invalid, "grounded": bool(cites) and not invalid}

    @traced("agent.score_risk")
    def score_risk(state: AgentState) -> AgentState:
        findings = []
        for r in state["retrieved"]:
            for rule in deps.rules:
                cov = match_rule(rule, r["text"])
                if cov < RULE_MATCH_MIN:
                    continue
                confidence = round(min(1.0, 0.5 + 0.5 * cov), 3)
                review = rule.always_review or confidence < deps.review_threshold
                findings.append(
                    {
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "severity": rule.severity,
                        "confidence": confidence,
                        "source_chunk_ids": [r["chunk_id"]],
                        "rationale": f"Clause matches rule '{rule.name}' ({cov:.0%} keyword coverage).",
                        "status": "pending_review" if review else "auto_resolved",
                    }
                )
        needs = any(f["status"] == "pending_review" for f in findings) or (
            bool(state.get("citations")) and not state.get("grounded", False)
        )
        return {"findings": findings, "needs_review": needs}

    @traced("agent.human_review")
    def human_review(state: AgentState) -> AgentState:
        # The graph only marks items for review; the API layer persists them to the review queue.
        return {}

    @traced("agent.finalize")
    def finalize(state: AgentState) -> AgentState:
        if not state.get("retrieved"):
            return {"answer": "No relevant content found for this question.", "grounded": False}
        return {}

    g = StateGraph(AgentState)
    for name, fn in [
        ("retrieve", retrieve),
        ("synthesize", synthesize),
        ("validate_citations", validate_citations),
        ("score_risk", score_risk),
        ("human_review", human_review),
        ("finalize", finalize),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "synthesize")
    g.add_edge("synthesize", "validate_citations")
    g.add_edge("validate_citations", "score_risk")
    g.add_conditional_edges(
        "score_risk",
        lambda s: "human_review" if s.get("needs_review") else "finalize",
        {"human_review": "human_review", "finalize": "finalize"},
    )
    g.add_edge("human_review", "finalize")
    g.add_edge("finalize", END)
    return g.compile()
