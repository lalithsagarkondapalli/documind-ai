import fakeredis
import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.agents.graph import Deps, build_graph
from app.core.cache import QueryCache
from app.services.llm import MockLLM
from app.services.retrieval import Chunk, RetrievalEngine

# ---------------------------------------------------------------- Redis cache path (against fakeredis)


@pytest.fixture
def redis_cache(monkeypatch):
    import redis

    server = fakeredis.FakeServer()
    monkeypatch.setattr(redis, "from_url", lambda *a, **k: fakeredis.FakeRedis(server=server, decode_responses=False))
    return QueryCache("redis://fake:6379/0", ttl_seconds=60), server


def test_redis_backend_roundtrip_and_ttl(redis_cache):
    cache, server = redis_cache
    assert cache.backend == "redis"
    cache.set("q:acme:d:1", {"answer": "x"})
    assert cache.get("q:acme:d:1") == {"answer": "x"}
    assert 0 < fakeredis.FakeRedis(server=server).ttl("q:acme:d:1") <= 60


def test_redis_invalidate_prefix_is_tenant_scoped(redis_cache):
    cache, _ = redis_cache
    cache.set("q:acme:d:1", {"v": 1})
    cache.set("q:acme:d:2", {"v": 2})
    cache.set("q:globex:d:1", {"v": 3})
    cache.invalidate_prefix("q:acme:")
    assert cache.get("q:acme:d:1") is None and cache.get("q:acme:d:2") is None
    assert cache.get("q:globex:d:1") == {"v": 3}


def test_replicas_share_one_redis_cache(monkeypatch):
    """The reason REDIS_URL is required for multi-replica: invalidation on one pod reaches the others."""
    import redis

    server = fakeredis.FakeServer()
    monkeypatch.setattr(redis, "from_url", lambda *a, **k: fakeredis.FakeRedis(server=server))
    pod_a, pod_b = QueryCache("redis://x", 60), QueryCache("redis://x", 60)
    pod_a.set("q:acme:d:1", {"v": 1})
    assert pod_b.get("q:acme:d:1") == {"v": 1}
    pod_b.invalidate_prefix("q:acme:")
    assert pod_a.get("q:acme:d:1") is None


# ---------------------------------------------------------------- OpenTelemetry spans from the agent graph


@pytest.fixture(scope="module")
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


def test_every_agent_node_emits_a_span(spans):
    engine = RetrievalEngine(
        [Chunk("c1", "d1", "Aggregate liability shall not exceed $1,000,000 indemnifying.", org_id="o")]
    )
    spans.clear()
    build_graph(Deps(engine, MockLLM("s", "f"))).invoke(
        {"question": "liability cap", "org_id": "o", "document_id": "d1"}
    )
    names = {s.name for s in spans.get_finished_spans()}
    assert {
        "agent.retrieve",
        "agent.synthesize",
        "agent.validate_citations",
        "agent.score_risk",
        "agent.finalize",
    } <= names
