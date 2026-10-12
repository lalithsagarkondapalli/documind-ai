"""Prometheus metrics exposed at /metrics."""

from prometheus_client import Counter, Histogram

QUERY_LATENCY = Histogram(
    "documind_query_seconds", "End-to-end /v1/query latency", buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)
QUERY_TOTAL = Counter("documind_queries_total", "Queries processed", ["cache"])
FINDINGS_TOTAL = Counter("documind_findings_total", "Risk findings produced", ["status", "severity"])
LLM_TOKENS = Counter("documind_llm_tokens_total", "Approximate LLM tokens", ["tier", "direction"])
