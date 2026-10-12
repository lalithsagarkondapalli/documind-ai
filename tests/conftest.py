import os
import sys
from pathlib import Path

os.environ.setdefault("JWT_SECRET", "test-secret-not-for-production-use-0123456789")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.main import create_app

get_settings.cache_clear()

CONTRACT = b"""MASTER SERVICES AGREEMENT between Acme Corp and Vendor Inc.

The Indemnifying Party's aggregate liability under this Section shall not exceed $2,500,000, except in cases of gross negligence or willful misconduct.

This Agreement shall automatically renew for successive 12-month periods unless either party provides written notice of non-renewal at least 60 days prior to the renewal date.

All Customer Data processed under this Agreement shall be stored and processed exclusively within the European Economic Area, and shall not be transferred outside this region without prior written consent.

This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware.
"""


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        jwt_secret="test-secret-not-for-production-use-0123456789",
        environment="test",
        llm_provider="mock",
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c


def login(client, username: str, password: str = "documind-demo") -> dict:
    r = client.post("/v1/auth/token", data={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture
def auth(client):
    return {u: login(client, u) for u in ("admin", "reviewer", "viewer", "globex-admin")}


@pytest.fixture
def uploaded_doc(client, auth):
    r = client.post("/v1/documents", headers=auth["reviewer"], files={"file": ("msa.txt", CONTRACT, "text/plain")})
    assert r.status_code == 201, r.text
    return r.json()
