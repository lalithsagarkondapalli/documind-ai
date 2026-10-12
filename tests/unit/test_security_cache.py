import time

import jwt
import pytest
from fastapi import HTTPException

from app.core.cache import QueryCache, cache_key
from app.core.security import (
    ROLE_RANK,
    create_access_token,
    get_principal,
    hash_password,
    require_role,
    verify_password,
)


def test_password_hash_roundtrip_and_salting():
    h = hash_password("s3cret")
    assert verify_password("s3cret", h) and not verify_password("wrong", h)
    assert hash_password("s3cret") != h  # per-hash random salt


def test_token_roundtrip_carries_org_and_role():
    p = get_principal(create_access_token("u1", "acme", "reviewer"))
    assert (p.user_id, p.org_id, p.role) == ("u1", "acme", "reviewer")


def test_tampered_or_garbage_token_rejected():
    with pytest.raises(HTTPException) as e:
        get_principal("not.a.jwt")
    assert e.value.status_code == 401
    forged = jwt.encode(
        {"sub": "u", "org_id": "acme", "role": "admin"},
        "a-different-but-long-enough-secret-key-0123",
        algorithm="HS256",
    )
    with pytest.raises(HTTPException):
        get_principal(forged)


def test_role_hierarchy():
    from app.core.security import Principal

    dep = require_role("reviewer")
    assert dep(Principal("u", "o", "admin")).role == "admin"
    with pytest.raises(HTTPException) as e:
        dep(Principal("u", "o", "viewer"))
    assert e.value.status_code == 403
    assert ROLE_RANK["admin"] > ROLE_RANK["reviewer"] > ROLE_RANK["viewer"]


def test_cache_key_normalises_whitespace_and_case_and_scopes_by_org():
    assert cache_key("a", "d", "What  IS the cap?") == cache_key("a", "d", "what is the cap?")
    assert cache_key("a", "d", "q") != cache_key("b", "d", "q")


def test_cache_ttl_and_prefix_invalidation():
    c = QueryCache(None, ttl_seconds=1)
    c.set("q:acme:d:1", {"x": 1})
    c.set("q:globex:d:1", {"x": 2})
    assert c.get("q:acme:d:1") == {"x": 1}
    c.invalidate_prefix("q:acme:")
    assert c.get("q:acme:d:1") is None and c.get("q:globex:d:1") == {"x": 2}
    time.sleep(1.1)
    assert c.get("q:globex:d:1") is None
