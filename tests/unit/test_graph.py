from app.agents.graph import Deps, build_graph
from app.services.llm import LLMResult, MockLLM
from app.services.retrieval import Chunk, RetrievalEngine

INDEMNITY = (
    "The Indemnifying Party's aggregate liability under this Section shall not exceed $1,000,000, "
    "except in cases of gross negligence."
)
GOVLAW = "This Agreement shall be governed by the laws of the State of Delaware."


def _engine():
    return RetrievalEngine([Chunk("c1", "d1", INDEMNITY, org_id="o"), Chunk("c2", "d1", GOVLAW, org_id="o")])


def _run(llm, q, engine=None, **kw):
    g = build_graph(Deps(engine or _engine(), llm, **kw))
    return g.invoke({"question": q, "org_id": "o", "document_id": "d1"})


def test_grounded_answer_with_valid_citation():
    out = _run(MockLLM("s", "f"), "which law governs the agreement")
    assert out["grounded"] and out["valid_citations"] == ["c2"] and not out["invalid_citations"]


def test_fabricated_citation_is_detected_and_forces_review():
    class Fabricating:
        def synthesize(self, q, ctx):
            return LLMResult("The cap is $50M. [ghost_chunk_999]", "fake", "small", 1, 1)

    out = _run(Fabricating(), "what is the cap")
    assert out["invalid_citations"] == ["ghost_chunk_999"]
    assert out["grounded"] is False and out["needs_review"] is True


def test_answer_without_any_citation_is_not_grounded():
    class NoCite:
        def synthesize(self, q, ctx):
            return LLMResult("The cap is probably large.", "fake", "small", 1, 1)

    assert _run(NoCite(), "what is the cap")["grounded"] is False


def test_always_review_rule_routes_to_human_even_at_full_confidence():
    out = _run(MockLLM("s", "f"), "what is the indemnification liability cap")
    f = next(f for f in out["findings"] if f["rule_id"] == "R-INDEMNITY")
    assert f["confidence"] == 1.0 and f["status"] == "pending_review" and out["needs_review"]


def test_low_risk_high_confidence_finding_auto_resolves():
    audit = Chunk(
        "c9",
        "d1",
        "The Company may audit Vendor compliance upon 15 days prior notice, no more than twice per calendar year.",
        org_id="o",
    )
    out = _run(MockLLM("s", "f"), "audit rights notice period", engine=RetrievalEngine([audit]))
    f = next(f for f in out["findings"] if f["rule_id"] == "R-AUDIT")
    assert f["status"] == "auto_resolved" and not out["needs_review"]


def test_threshold_controls_review_routing():
    audit = Chunk("c9", "d1", "The Company may audit Vendor compliance upon prior notice.", org_id="o")
    strict = _run(MockLLM("s", "f"), "audit rights", engine=RetrievalEngine([audit]), review_threshold=0.99)
    assert any(f["status"] == "pending_review" for f in strict["findings"])


def test_no_match_returns_empty_findings_and_no_review():
    out = _run(
        MockLLM("s", "f"),
        "completely unrelated zebra question",
        engine=RetrievalEngine([Chunk("c3", "d1", "Boilerplate severability clause text.", org_id="o")]),
    )
    assert out["findings"] == [] and not out["needs_review"]


def test_other_tenant_chunks_are_invisible_to_graph():
    e = RetrievalEngine([Chunk("x1", "d1", INDEMNITY, org_id="other-org")])
    g = build_graph(Deps(e, MockLLM("s", "f")))
    out = g.invoke({"question": "liability cap", "org_id": "o", "document_id": "d1"})
    assert out["retrieved"] == []
