"""JWT auth + role-based access control.

org_id and role come ONLY from the verified token, never from request bodies, which is
what enforces tenant isolation at the API boundary."""

from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.core.config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/v1/auth/token")
ROLE_RANK = {"viewer": 1, "reviewer": 2, "admin": 3}


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)
    return f"{salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    salt_hex, _ = stored.split("$")
    return hmac.compare_digest(hash_password(password, bytes.fromhex(salt_hex)), stored)


def create_access_token(user_id: str, org_id: str, role: str) -> str:
    s = get_settings()
    exp = datetime.now(UTC) + timedelta(minutes=s.jwt_expire_minutes)
    return jwt.encode(
        {"sub": user_id, "org_id": org_id, "role": role, "exp": exp}, s.jwt_secret, algorithm=s.jwt_algorithm
    )


@dataclass
class Principal:
    user_id: str
    org_id: str
    role: str


def get_principal(token: str = Depends(oauth2_scheme)) -> Principal:
    s = get_settings()
    try:
        data = jwt.decode(token, s.jwt_secret, algorithms=[s.jwt_algorithm])
    except jwt.PyJWTError:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid or expired token", headers={"WWW-Authenticate": "Bearer"}
        ) from None
    return Principal(data["sub"], data["org_id"], data["role"])


def require_role(min_role: str):
    def dep(p: Principal = Depends(get_principal)) -> Principal:
        if ROLE_RANK.get(p.role, 0) < ROLE_RANK[min_role]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires role '{min_role}' or higher")
        return p

    return dep
