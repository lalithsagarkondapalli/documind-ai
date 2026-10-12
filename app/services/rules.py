"""Configurable risk rulebook. Each rule matches a clause by keyword coverage.
Rules marked always_review always go to a human, whatever the confidence."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.retrieval import tokenize


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    keywords: tuple
    severity: str  # low | medium | high
    always_review: bool = False


DEFAULT_RULES: list[Rule] = [
    Rule(
        "R-INDEMNITY",
        "Indemnification liability cap",
        ("indemnifying", "aggregate", "liability", "exceed"),
        "high",
        True,
    ),
    Rule("R-LIABILITY", "Limitation of liability", ("total", "liability", "exceed", "fees", "preceding"), "high", True),
    Rule("R-RENEWAL", "Automatic renewal", ("automatically", "renew", "successive", "non-renewal"), "medium"),
    Rule("R-DATA", "Data residency", ("customer", "data", "stored", "processed", "transferred", "region"), "high"),
    Rule("R-TERMINATION", "Termination for convenience", ("terminate", "convenience", "written", "notice"), "medium"),
    Rule("R-AUDIT", "Audit rights", ("audit", "compliance", "prior", "notice", "calendar", "year"), "low"),
    Rule(
        "R-CONFIDENTIALITY",
        "Confidentiality survival",
        ("confidentiality", "obligations", "survive", "termination"),
        "low",
    ),
    Rule("R-INTEREST", "Interest rate", ("interest", "rate", "per", "annum", "principal"), "medium"),
]


def match_rule(rule: Rule, text: str) -> float:
    """Fraction of the rule's keywords present in the text (0..1)."""
    toks = set(tokenize(text))
    return sum(1 for k in rule.keywords if k in toks) / len(rule.keywords)
