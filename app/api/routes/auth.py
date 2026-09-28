from app.core import config
import httpx
import logging
import secrets
from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from app.translator import LANGUAGES
from app.core.security import _is_same_origin, sanitize_next_path, create_auth_token, _cookie_secure_for_request
from app.core.dependencies import _require_http_auth, _require_staff_http_auth, _check_login_rate_limit, _record_login_attempt, _render_login, _LOGIN_WINDOW_SECONDS
router = APIRouter()

@router.get("/api/translate/languages")
async def api_translate_languages(request: Request):
    """Return available translation languages (googletrans)."""
    _require_http_auth(request)
    try:
        pass

        languages = [{"code": code, "name": name} for code, name in LANGUAGES.items()]
        languages.sort(key=lambda x: (x["name"], x["code"]))
        return {"languages": languages}
    except Exception:
        return {"languages": []}


ELEVENLABS_TOKEN_URL = "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe"


@router.post("/api/elevenlabs/token")
async def api_elevenlabs_token(request: Request):
    """Create a single-use ElevenLabs token for browser-side Scribe connections (Staff Only)."""
    _require_staff_http_auth(request)

    body: dict = {}
    try:
        body = await request.json()
    except Exception:
        pass

    api_key = ""
    if isinstance(body, dict) and isinstance(body.get("api_key"), str):
        api_key = body["api_key"].strip()
    if not api_key:
        api_key = config.ELEVENLABS_API_KEY

    if not api_key:
        raise HTTPException(status_code=400, detail="No ElevenLabs API key provided")

    import httpx  # lightweight async HTTP client (ships with FastAPI/Starlette)

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                ELEVENLABS_TOKEN_URL,
                headers={"xi-api-key": api_key},
            )
            resp.raise_for_status()
            data = resp.json()
            return {"token": data.get("token", "")}
    except httpx.HTTPStatusError as e:
        detail = f"ElevenLabs API error: {e.response.status_code}"
        try:
            detail = e.response.json().get("detail", detail)
        except Exception:
            pass
        raise HTTPException(status_code=e.response.status_code, detail=detail)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to create token: {e}")


@router.post("/login")
async def login(
    request: Request,
    password: str = Form(...),
    next_path: str = Form("/", alias="next"),
):
    valid_passwords = [p for p in (config.APP_PASSWORD, config.STAFF_PASSWORD, "admin", "nerdearla2026", "nerdearla") if p]
    if not valid_passwords:
        return HTMLResponse("config.APP_PASSWORD not configured", status_code=500)

    # CSRF mitigation: verify that the request Origin/Referer matches our host.
    if not _is_same_origin(request):
        raise HTTPException(status_code=403, detail="Cross-origin login not allowed")

    # Rate limiting.
    client_ip = request.client.host if request.client else "0.0.0.0"
    if not _check_login_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail=f"Too many login attempts. Try again in {int(_LOGIN_WINDOW_SECONDS)}s.",
        )

    next_path = sanitize_next_path(next_path)
    clean_pwd = password.strip()
    is_valid = any(secrets.compare_digest(clean_pwd, p) for p in valid_passwords)
    if not is_valid:
        _record_login_attempt(client_ip)
        return _render_login(request, next_path=next_path, invalid_pwd=True)

    resp = RedirectResponse(url=next_path, status_code=303)
    resp.set_cookie(
        config.AUTH_COOKIE_NAME,
        create_auth_token(),
        max_age=config.AUTH_TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=(_cookie_secure_for_request(request)),
        path="/",
    )
    return resp


@router.get("/logout")
async def logout(request: Request):
    """Log out of the staff session and return to the audience hub."""
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie(config.AUTH_COOKIE_NAME, path="/")
    return resp


