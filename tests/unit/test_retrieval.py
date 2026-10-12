from app.services.retrieval import Chunk, RetrievalEngine
from app.services.vectorstore import QdrantBackend


def _chunks():
    return [
        Chunk("a1", "docA", "Aggregate liability shall not exceed $1,000,000.", org_id="org1"),
        Chunk("a2", "docA", "Governed by the laws of the State of Delaware.", org_id="org1"),
        Chunk("a3", "docA", "Customer data stored within the European Economic Area.", org_id="org1"),
        Chunk("b1", "docB", "Aggregate liability shall not exceed $9,000,000.", org_id="org2"),
    ]


def test_hybrid_finds_relevant_chunk_first():
    e = RetrievalEngine(_chunks())
    # NB: the offline embedder is lexical (no stemming), so the query shares exact words with the clause.
    hits = e.hybrid_search("laws of the State of Delaware", top_k=3, document_id="docA", org_id="org1")
    assert e.get_chunk(hits[0][0]).chunk_id == "a2"


def test_lexical_embedder_has_no_stemming_known_limitation():
    """Documents a real limitation: 'law governs' does not match 'laws governed'.
    A real embedding model (or a stemmer) is the fix; see README."""
    e = RetrievalEngine(_chunks())
    scores = dict(e.dense_search("which law governs", top_k=4, document_id="docA", org_id="org1"))
    assert all(s == 0.0 for s in scores.values())


def test_tenant_isolation_never_returns_other_org_chunks():
    e = RetrievalEngine(_chunks())
    hits = e.hybrid_search("aggregate liability cap", top_k=10, org_id="org1")
    assert {e.get_chunk(i).chunk_id for i, _ in hits}.isdisjoint({"b1"})


def test_document_scope_restricts_results():
    e = RetrievalEngine(_chunks())
    hits = e.hybrid_search("liability", top_k=10, document_id="docB", org_id="org2")
    assert [e.get_chunk(i).chunk_id for i, _ in hits] == ["b1"]


def test_incremental_indexing_makes_new_chunks_searchable():
    e = RetrievalEngine(_chunks())
    e.add_chunks([Chunk("a4", "docA", "Either party may terminate on 90 days notice.", org_id="org1")])
    hits = e.hybrid_search("terminate notice", top_k=1, document_id="docA", org_id="org1")
    assert e.get_chunk(hits[0][0]).chunk_id == "a4"


def test_empty_scope_returns_nothing():
    assert RetrievalEngine(_chunks()).hybrid_search("liability", document_id="nope", org_id="org1") == []


def test_qdrant_backend_matches_in_memory_ranking():
    mem = RetrievalEngine(_chunks())
    qd = RetrievalEngine(_chunks(), dense_backend=QdrantBackend())
    for q in ("aggregate liability cap", "laws of the State of Delaware", "European Economic Area data"):
        m = [mem.get_chunk(i).chunk_id for i, _ in mem.hybrid_search(q, top_k=3, org_id="org1")]
        d = [qd.get_chunk(i).chunk_id for i, _ in qd.hybrid_search(q, top_k=3, org_id="org1")]
        assert m[0] == d[0], q  # same best match
        assert set(m) == set(d), q  # order among zero-score ties is unspecified


def test_qdrant_backend_enforces_org_filter():
    qd = RetrievalEngine(_chunks(), dense_backend=QdrantBackend())
    hits = qd.hybrid_search("aggregate liability", top_k=10, org_id="org2")
    assert {qd.get_chunk(i).chunk_id for i, _ in hits} == {"b1"}


def test_add_chunks_is_idempotent():
    e = RetrievalEngine(_chunks())
    n = len(e.chunks)
    e.add_chunks(_chunks())  # same chunk_ids again
    assert len(e.chunks) == n and e.embeddings.shape[0] == n


def test_two_replicas_share_one_qdrant_without_id_collisions():
    """Replica A writes vectors; replica B only loads chunk text (upsert_dense=False) and still finds them.
    Point ids derive from chunk_id, not from a per-process list position, so nothing collides."""
    from qdrant_client import QdrantClient

    shared = QdrantClient(":memory:")
    a = RetrievalEngine([], dense_backend=QdrantBackend(client=shared))
    a.add_chunks([Chunk("x1", "d", "Aggregate liability shall not exceed $1,000,000.", org_id="o")])

    b = RetrievalEngine([], dense_backend=QdrantBackend(client=shared))
    # B ingests different chunks FIRST, so its local positions differ from A's
    b.add_chunks([Chunk("y1", "d", "Unrelated severability boilerplate language.", org_id="o")])
    b.add_chunks([Chunk("x1", "d", "Aggregate liability shall not exceed $1,000,000.", org_id="o")], upsert_dense=False)

    hits = b.hybrid_search("aggregate liability", top_k=2, document_id="d", org_id="o")
    assert b.get_chunk(hits[0][0]).chunk_id == "x1"
    assert shared.count("documind_chunks").count == 2  # x1 written once, y1 once; no overwrite
