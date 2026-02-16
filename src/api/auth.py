"""
HighwayLens Authentication
==========================

Shared-password authentication with HMAC-signed session cookies.

Environment variables:
    APP_PASSWORD    – shared password for access
    SESSION_SECRET  – HMAC signing key for session tokens
"""

import hmac
import hashlib
import logging
import os

from fastapi import Request, Response
from fastapi.responses import JSONResponse, RedirectResponse

logger = logging.getLogger(__name__)

APP_PASSWORD = os.getenv("APP_PASSWORD", "")
if not APP_PASSWORD:
    logger.warning("APP_PASSWORD is not set — login will always fail")
SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me")

COOKIE_NAME = "session"
# No max_age → session cookie (expires when browser closes)

# Paths that don't require authentication
PUBLIC_PATHS = {"/login", "/api/v1/auth/login", "/health", "/docs", "/redoc", "/openapi.json"}


def _make_token() -> str:
    """Generate HMAC-signed session token from password + secret."""
    return hmac.new(
        SESSION_SECRET.encode(), APP_PASSWORD.encode(), hashlib.sha256
    ).hexdigest()[:32]


def _verify_token(token: str) -> bool:
    """Constant-time comparison of session token."""
    expected = _make_token()
    return hmac.compare_digest(token, expected)


def verify_password(password: str) -> bool:
    """Check if the provided password matches APP_PASSWORD."""
    if not APP_PASSWORD:
        return False
    return hmac.compare_digest(password, APP_PASSWORD)


async def auth_middleware(request: Request, call_next):
    """Authentication middleware.

    - Public paths are allowed through.
    - API paths (/api/*) return 401 JSON if unauthenticated.
    - Page paths redirect to /login if unauthenticated.
    """
    path = request.url.path

    # Allow public paths
    if path in PUBLIC_PATHS:
        return await call_next(request)

    # Check session cookie
    token = request.cookies.get(COOKIE_NAME)
    if token and _verify_token(token):
        return await call_next(request)

    # Unauthenticated
    if path.startswith("/api/"):
        return JSONResponse(status_code=401, content={"detail": "Not authenticated"})

    # Page request → redirect to login
    return RedirectResponse(url="/login", status_code=302)


def set_session_cookie(response: Response) -> Response:
    """Set session cookie (expires when browser closes)."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=_make_token(),
        httponly=True,
        samesite="lax",
    )
    return response


def clear_session_cookie(response: Response) -> Response:
    """Remove session cookie."""
    response.delete_cookie(key=COOKIE_NAME)
    return response
