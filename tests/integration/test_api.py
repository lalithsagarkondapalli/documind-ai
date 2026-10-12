import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core.config import Settings
from app.db.models import AuditLog
from app.main import create_app
from tests.conftest import CONTRACT, login


def _audit_actions(client, org="acme"):
    with client.app.state.documind.session_factory() as db:
        return [
            a.action
            for a in db.scalars(select(AuditLog).where(AuditLog.org_id == org).order_by(AuditLog.created_at)).all()
        ]


# ---------- auth ----------
def test_login_bad_password_rejected(client):
    r = client.post("/v1/auth/token", data={"username": "admin", "password": "nope"})
    assert r.status_code == 401


def test_endpoints_require_token(client):
    for method, path in [("get", "/v1/documents"), ("get", "/v1/reviews/queue"), ("post", "/v1/query")]:
        assert getattr(client, method)(path).status_code == 401


def test_health_ready_metrics_are_public(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["status"] == "ready"
    m = client.get("/metrics")
    assert m.status_code == 200 and b"documind_queries_total" in m.content


# ---------- RBAC ----------
def test_viewer_cannot_upload_or_see_review_queue(client, auth):
    up = client.post("/v1/documents", headers=auth["viewer"], files={"file": ("a.txt", CONTRACT, "text/plain")})
    assert up.status_code == 403
    assert client.get("/v1/reviews/queue", headers=auth["viewer"]).status_code == 403


def test_viewer_can_query_and_list(client, auth, uploaded_doc):
    assert client.get("/v1/documents", headers=auth["viewer"]).status_code == 200
    r = client.post(
        "/v1/query",
        headers=auth["viewer"],
        json={"question": "which State of Delaware law governs", "document_id": uploaded_doc["id"]},
    )
    assert r.status_code == 200


# ---------- ingestion ----------
def test_upload_indexes_and_reports_chunks(uploaded_doc):
    assert uploaded_doc["status"] == "indexed" and uploaded_doc["n_chunks"] >= 4


def test_upload_rejects_unsupported_type(client, auth):
    r = client.post(
        "/v1/documents", headers=auth["reviewer"], files={"file": ("x.exe", b"MZ\x90\x00", "application/octet-stream")}
    )
    assert r.status_code == 415


# ---------- query + citations ----------
def test_query_returns_grounded_answer_with_valid_citation(client, auth, uploaded_doc):
    r = client.post(
        "/v1/query",
        headers=auth["viewer"],
        json={"question": "What is the governing law State of Delaware?", "document_id": uploaded_doc["id"]},
    )
    body = r.json()
    assert body["grounded"] and body["citations"] and not body["invalid_citations"]
    assert "Delaware" in body["answer"] and body["llm_provider"] == "mock"
    retrieved_ids = {c["chunk_id"] for c in body["retrieved"]}
    assert set(body["citations"]) <= retrieved_ids


def test_query_validation_rejects_short_question(client, auth):
    assert client.post("/v1/query", headers=auth["viewer"], json={"question": "x"}).status_code == 422


# ---------- human-in-the-loop ----------
def test_high_risk_clause_goes_to_review_queue_and_can_be_approved(client, auth, uploaded_doc):
    q = client.post(
        "/v1/query",
        headers=auth["viewer"],
        json={"question": "What is the indemnification aggregate liability cap?", "document_id": uploaded_doc["id"]},
    ).json()
    assert q["review_required"]
    queue = client.get("/v1/reviews/queue", headers=auth["reviewer"]).json()
    item = next(i for i in queue if i["rule_id"] == "R-INDEMNITY")
    assert item["status"] == "pending_review" and item["severity"] == "high"

    d = client.post(
        f"/v1/reviews/{item['id']}/decision",
        headers=auth["reviewer"],
        json={"decision": "approve", "note": "Cap matches negotiated terms"},
    )
    assert d.status_code == 200 and d.json()["status"] == "approved"
    assert d.json()["reviewed_by"] and d.json()["reviewer_note"] == "Cap matches negotiated terms"
    assert all(i["id"] != item["id"] for i in client.get("/v1/reviews/queue", headers=auth["reviewer"]).json())


def test_edit_decision_can_change_severity(client, auth, uploaded_doc):
    client.post(
        "/v1/query",
        headers=auth["viewer"],
        json={"question": "indemnification aggregate liability cap", "document_id": uploaded_doc["id"]},
    )
    item = client.get("/v1/reviews/queue", headers=auth["reviewer"]).json()[0]
    d = client.post(
        f"/v1/reviews/{item['id']}/decision", headers=auth["reviewer"], json={"decision": "edit", "severity": "low"}
    ).json()
    assert d["severity"] == "low" and d["status"] == "approved"


def test_cannot_decide_twice(client, auth, uploaded_doc):
    client.post(
        "/v1/query",
        headers=auth["viewer"],
        json={"question": "indemnification aggregate liability cap", "document_id": uploaded_doc["id"]},
    )
    fid = client.get("/v1/reviews/queue", headers=auth["reviewer"]).json()[0]["id"]
    assert (
        client.post(f"/v1/reviews/{fid}/decision", headers=auth["reviewer"], json={"decision": "reject"}).status_code
        == 200
    )
    assert (
        client.post(f"/v1/reviews/{fid}/decision", headers=auth["reviewer"], json={"decision": "approve"}).status_code
        == 409
    )


def test_repeat_queries_do_not_duplicate_review_items(client, auth, uploaded_doc):
    for q in ("indemnification aggregate liability cap", "what is the aggregate liability exceed indemnifying"):
        client.post("/v1/query", headers=auth["viewer"], json={"question": q, "document_id": uploaded_doc["id"]})
    rules = [i["rule_id"] for i in client.get("/v1/reviews/queue", headers=auth["reviewer"]).json()]
    assert rules.count("R-INDEMNITY") == 1


def test_audit_trail_records_agent_and_human_actions(client, auth, uploaded_doc):
    client.post(
        "/v1/query",
        headers=auth["viewer"],
        json={"question": "indemnification aggregate liability cap", "document_id": uploaded_doc["id"]},
    )
    fid = client.get("/v1/reviews/queue", headers=auth["reviewer"]).json()[0]["id"]
    client.post(f"/v1/reviews/{fid}/decision", headers=auth["reviewer"], json={"decision": "approve"})
    actions = _audit_actions(client)
    for expected in ("document.uploaded", "query.answered", "finding.created", "finding.approve"):
        assert expected in actions


# ---------- tenant isolation ----------
def test_other_tenant_cannot_read_document(client, auth, uploaded_doc):
    assert client.get(f"/v1/documents/{uploaded_doc['id']}", headers=auth["globex-admin"]).status_code == 404
    assert client.get("/v1/documents", headers=auth["globex-admin"]).json() == []


def test_other_tenant_cannot_query_document_by_id(client, auth, uploaded_doc):
    r = client.post(
        "/v1/query",
        headers=auth["globex-admin"],
        json={"question": "indemnification liability cap", "document_id": uploaded_doc["id"]},
    )
    assert r.status_code == 404


def test_corpus_wide_query_never_leaks_other_tenants_text(client, auth, uploaded_doc):
    r = client.post("/v1/query", headers=auth["globex-admin"], json={"question": "indemnification liability cap"})
    assert r.status_code == 200 and r.json()["retrieved"] == []
    assert "2,500,000" not in r.json()["answer"]


def test_other_tenant_cannot_see_or_decide_findings(client, auth, uploaded_doc):
    client.post(
        "/v1/query",
        headers=auth["viewer"],
        json={"question": "indemnification aggregate liability cap", "document_id": uploaded_doc["id"]},
    )
    fid = client.get("/v1/reviews/queue", headers=auth["reviewer"]).json()[0]["id"]
    assert client.get("/v1/reviews/queue", headers=auth["globex-admin"]).json() == []
    assert (
        client.post(
            f"/v1/reviews/{fid}/decision", headers=auth["globex-admin"], json={"decision": "approve"}
        ).status_code
        == 404
    )


# ---------- cache ----------
def test_second_identical_query_is_served_from_cache(client, auth, uploaded_doc):
    body = {"question": "Which State of Delaware law governs?", "document_id": uploaded_doc["id"]}
    first = client.post("/v1/query", headers=auth["viewer"], json=body).json()
    second = client.post("/v1/query", headers=auth["viewer"], json=body).json()
    assert first["cached"] is False and second["cached"] is True and second["answer"] == first["answer"]


def test_cache_invalidated_after_review_decision(client, auth, uploaded_doc):
    body = {"question": "indemnification aggregate liability cap", "document_id": uploaded_doc["id"]}
    client.post("/v1/query", headers=auth["viewer"], json=body)
    fid = client.get("/v1/reviews/queue", headers=auth["reviewer"]).json()[0]["id"]
    client.post(f"/v1/reviews/{fid}/decision", headers=auth["reviewer"], json={"decision": "approve"})
    again = client.post("/v1/query", headers=auth["viewer"], json=body).json()
    assert again["cached"] is False
    assert next(f for f in again["findings"] if f["id"] == fid)["status"] == "approved"


# ---------- persistence / startup safety ----------
def test_index_is_rebuilt_from_database_after_restart(settings, uploaded_doc, client, auth):
    with TestClient(create_app(settings)) as c2:  # a fresh process, same database
        assert c2.get("/ready").json()["chunks_indexed"] >= uploaded_doc["n_chunks"]
        r = c2.post(
            "/v1/query",
            headers=login(c2, "viewer"),
            json={"question": "Which State of Delaware law governs?", "document_id": uploaded_doc["id"]},
        )
        assert r.json()["grounded"]


def test_production_refuses_default_jwt_secret(tmp_path):
    s = Settings(
        database_url=f"sqlite:///{tmp_path / 'p.db'}", environment="production", jwt_secret="change-me-in-production"
    )
    with pytest.raises(RuntimeError, match="default JWT secret"):
        with TestClient(create_app(s)):
            pass


def test_production_does_not_seed_demo_users(tmp_path):
    s = Settings(
        database_url=f"sqlite:///{tmp_path / 'p.db'}",
        environment="production",
        jwt_secret="a-sufficiently-long-production-secret-0123456789",
    )
    with TestClient(create_app(s)) as c:
        assert c.post("/v1/auth/token", data={"username": "admin", "password": "documind-demo"}).status_code == 401


def test_qdrant_backend_serves_queries_end_to_end(tmp_path):
    s = Settings(
        database_url=f"sqlite:///{tmp_path / 'q.db'}",
        environment="test",
        vector_backend="qdrant",
        jwt_secret="test-secret-not-for-production-use-0123456789",
    )
    with TestClient(create_app(s)) as c:
        h = login(c, "reviewer")
        doc = c.post("/v1/documents", headers=h, files={"file": ("msa.txt", CONTRACT, "text/plain")}).json()
        r = c.post(
            "/v1/query", headers=h, json={"question": "State of Delaware governing law", "document_id": doc["id"]}
        )
        assert r.json()["grounded"] and "Delaware" in r.json()["answer"]


# ---------- multi-replica correctness ----------
def test_document_uploaded_on_one_replica_is_searchable_on_another(settings):
    """Two app instances share one database, like two Kubernetes pods. A document uploaded to
    replica A must be answerable by replica B without restarting B."""
    settings = settings.model_copy(update={"index_sync_interval_seconds": 0})
    with TestClient(create_app(settings)) as a, TestClient(create_app(settings)) as b:
        doc = a.post(
            "/v1/documents", headers=login(a, "reviewer"), files={"file": ("msa.txt", CONTRACT, "text/plain")}
        ).json()
        r = b.post(
            "/v1/query",
            headers=login(b, "viewer"),
            json={"question": "Which State of Delaware law governs?", "document_id": doc["id"]},
        )
        assert r.status_code == 200 and r.json()["grounded"], r.text


def test_demo_user_seeding_survives_a_startup_race(tmp_path):
    """Two replicas can both see 'user missing' and both insert. The loser must not crash startup."""
    from sqlalchemy.orm import Session, sessionmaker

    from app.db.models import User, make_session_factory
    from app.main import _seed_demo_users

    engine, normal = make_session_factory(f"sqlite:///{tmp_path / 'race.db'}")
    _seed_demo_users(normal, "pw")  # replica A seeds first

    class BlindSession(Session):  # replica B's existence check ran BEFORE A committed
        def scalar(self, *a, **k):
            return None

    blind = sessionmaker(bind=engine, class_=BlindSession, expire_on_commit=False)
    _seed_demo_users(blind, "pw")  # previously raised IntegrityError and aborted startup

    with normal() as db:
        assert db.query(User).count() == 4


def test_production_rejects_short_jwt_secret(tmp_path):
    s = Settings(database_url=f"sqlite:///{tmp_path / 'p.db'}", environment="production", jwt_secret="secret123")
    with pytest.raises(RuntimeError, match="at least 32 bytes"):
        with TestClient(create_app(s)):
            pass
