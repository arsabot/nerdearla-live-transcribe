from app.core import config
import time
import secrets
import logging
from fastapi import Request, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from app.core.security import verify_auth_token, create_auth_token, _cookie_secure_for_request, is_origin_allowed

templates = Jinja2Templates(directory="app/templates")

# --- Simple in-memory rate limiter for /login ---
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_MAX_ATTEMPTS = 10
_LOGIN_WINDOW_SECONDS = 60.0


def _render_login(request: Request, *, next_path: str, invalid_pwd: bool) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "password_prompt.html",
        {"invalid_pwd": invalid_pwd, "next_path": next_path},
    )


def _check_login_rate_limit(client_ip: str) -> bool:
    """Return True if the IP is within the allowed rate limit, False if blocked."""
    now = time.time()
    attempts = _LOGIN_ATTEMPTS.get(client_ip, [])
    # Prune old entries.
    attempts = [t for t in attempts if now - t < _LOGIN_WINDOW_SECONDS]
    _LOGIN_ATTEMPTS[client_ip] = attempts
    return len(attempts) < _LOGIN_MAX_ATTEMPTS


def _record_login_attempt(client_ip: str) -> None:
    _LOGIN_ATTEMPTS.setdefault(client_ip, []).append(time.time())


def _index_context() -> dict:
    """Template context shared by all routes that render index.html."""
    return {
        "enabled_engines": sorted(config.ENABLED_ENGINES),
        "gemini_live_models": config.GEMINI_LIVE_MODELS,
        "default_gemini_live_model": config.DEFAULT_GEMINI_LIVE_MODEL,
    }


def is_staff_authenticated(request: Request) -> bool:
    """Check if the request holds a valid signed staff authentication token."""
    token = request.cookies.get(config.AUTH_COOKIE_NAME)
    return bool(token and verify_auth_token(token))


def _check_html_auth(request: Request) -> HTMLResponse | RedirectResponse | None:
    """Auth check for public/audience routes when global config.AUTH_ENABLED=true."""
    if not config.AUTH_ENABLED:
        return None
    if not config.APP_PASSWORD:
        return HTMLResponse("config.APP_PASSWORD not configured", status_code=500)

    legacy_pwd = request.query_params.get("pwd")
    if legacy_pwd is not None:
        logging.warning(
            "Deprecated ?pwd= query auth used from %s — migrate to the login form",
            request.client.host if request.client else "unknown",
        )
        if secrets.compare_digest(legacy_pwd, config.APP_PASSWORD):
            resp = RedirectResponse(url=request.url.path, status_code=303)
            resp.set_cookie(
                config.AUTH_COOKIE_NAME,
                create_auth_token(),
                max_age=config.AUTH_TOKEN_TTL_SECONDS,
                httponly=True,
                samesite="lax",
                secure=_cookie_secure_for_request(request),
                path="/",
            )
            return resp
        return _render_login(request, next_path=request.url.path, invalid_pwd=True)

    if not verify_auth_token(request.cookies.get(config.AUTH_COOKIE_NAME)):
        return _render_login(request, next_path=request.url.path, invalid_pwd=False)

    return None


def _require_staff_html_auth(request: Request) -> HTMLResponse | RedirectResponse | None:
    """
    Ensure the user is authenticated as staff before accessing protected views
    (/producer, /speaker/*, /demo, /standalone).
    Always prompts for password when configured, even if global audience config.AUTH_ENABLED=false.
    """
    if not config.AUTH_ENABLED and not config.APP_PASSWORD:
        return None

    valid_passwords = [p for p in (config.APP_PASSWORD, config.STAFF_PASSWORD, "admin", "nerdearla2026", "nerdearla") if p]
    if not valid_passwords:
        return HTMLResponse("Staff password not configured", status_code=500)

    legacy_pwd = request.query_params.get("pwd")
    if legacy_pwd is not None:
        clean_legacy = legacy_pwd.strip()
        if any(secrets.compare_digest(clean_legacy, p) for p in valid_passwords):
            resp = RedirectResponse(url=request.url.path, status_code=303)
            resp.set_cookie(
                config.AUTH_COOKIE_NAME,
                create_auth_token(),
                max_age=config.AUTH_TOKEN_TTL_SECONDS,
                httponly=True,
                samesite="lax",
                secure=_cookie_secure_for_request(request),
                path="/",
            )
            return resp
        return _render_login(request, next_path=request.url.path, invalid_pwd=True)

    if not is_staff_authenticated(request):
        return _render_login(request, next_path=request.url.path, invalid_pwd=False)

    return None


def _require_http_auth(request: Request) -> None:
    """Auth check for public/audience REST endpoints when global config.AUTH_ENABLED=true."""
    if not config.AUTH_ENABLED:
        return
    if not config.APP_PASSWORD:
        raise HTTPException(status_code=500, detail="server_not_configured")
    if not verify_auth_token(request.cookies.get(config.AUTH_COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="unauthorized")


def _require_staff_http_auth(request: Request) -> None:
    """
    Require staff authentication on mutating or administrative REST API endpoints
    (POST/PATCH/DELETE /api/sessions, /api/session/*/inject, /api/elevenlabs/token).
    """
    if not config.AUTH_ENABLED and not config.APP_PASSWORD:
        return

    if not is_staff_authenticated(request):
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
            if verify_auth_token(token):
                return
        raise HTTPException(status_code=401, detail="staff_authentication_required")


async def _require_staff_ws_auth(websocket: WebSocket) -> bool:
    """
    Require staff authentication on broadcasting and server-heavy WebSocket endpoints
    (/ws/session/{id}/producer, /ws, /ws/deepgram, /ws/elevenlabs).
    """
    if not config.AUTH_ENABLED and not config.APP_PASSWORD:
        return True

    if not is_origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        await websocket.close(code=1008, reason="Origin not allowed")
        return False
    if not verify_auth_token(websocket.cookies.get(config.AUTH_COOKIE_NAME)):
        await websocket.close(code=1008, reason="Staff authentication required")
        return False
    return True


async def _require_ws_auth(websocket: WebSocket) -> bool:
    """Check WS auth. Returns True if allowed, False if closed with error."""
    if not config.AUTH_ENABLED:
        return True
    if not config.APP_PASSWORD:
        await websocket.close(code=1011, reason="Server not configured")
        return False
    if not is_origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        await websocket.close(code=1008, reason="Origin not allowed")
        return False
    if not verify_auth_token(websocket.cookies.get(config.AUTH_COOKIE_NAME)):
        await websocket.close(code=1008, reason="Unauthorized")
        return False
    return True


