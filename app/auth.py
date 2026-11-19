"""JWT verification. The company comes ONLY from the signed token; no request field can change it."""
import time
from dataclasses import dataclass

import jwt
from fastapi import Header, HTTPException, Request

from .store import valid_company


@dataclass(frozen=True)
class Principal:
    subject: str
    company: str
    role: str  # employee | admin


def issue_token(settings, subject: str, company: str, role: str = "employee", ttl_s: int = 3600) -> str:
    """The mock identity provider. In production a real IdP does this; we only verify."""
    now = int(time.time())
    return jwt.encode({"sub": subject, "company": company, "role": role, "iss": settings.jwt_issuer,
                       "aud": settings.jwt_audience, "iat": now, "exp": now + ttl_s},
                      settings.jwt_secret, algorithm=settings.jwt_algorithm)


def verify_token(settings, token: str) -> Principal:
    try:
        claims = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm],
                            issuer=settings.jwt_issuer, audience=settings.jwt_audience,
                            options={"require": ["exp", "sub", "iss", "aud"]})
    except jwt.PyJWTError:
        raise HTTPException(401, "invalid or expired token") from None
    company, role = claims.get("company"), claims.get("role", "employee")
    if not valid_company(company) or role not in ("employee", "admin"):
        raise HTTPException(401, "token has no valid company")
    return Principal(str(claims["sub"]), company, role)


def current_principal(request: Request, authorization: str = Header(default="")) -> Principal:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(401, "missing bearer token")
    return verify_token(request.app.state.settings, token)
