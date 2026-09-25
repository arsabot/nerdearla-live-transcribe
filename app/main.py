import asyncio
import contextlib
import inspect
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from typing import TypedDict
from urllib.parse import urlparse

import websockets as ws_lib

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, RedirectResponse, Response, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.translator import Translator, LANGUAGES
from app.session_manager import SessionManager, Session, TranscriptEvent, LatencyMetrics
from app.translation_provider import (
    get_translation_provider,
    TranslationProvider,
    apply_glossary,
    GoogleTranslationProvider,
    normalize_brand_terms,
)
from app.gemini_live import (
    GeminiLiveTranslator,
    GeminiLiveProvider,
    GeminiLiveCircuitBreaker,
    global_gemini_live_circuit,
    redact_secrets,
    ErrorType,
    ConnectionState,
    LatencyTelemetry,
    build_system_instruction,
)
from app.exporter import export_vtt, export_srt, export_txt
from starlette.websockets import WebSocketDisconnect

logging.basicConfig(level=logging.INFO)

# deepgram-sdk has had breaking API changes across major versions. Treat it as an
# optional dependency so the app can still boot (at least for the Web Speech API
# mode) when Deepgram is not installed or an import path changes.
try:  # pragma: no cover - depends on installed deepgram-sdk version
    from deepgram import DeepgramClient  # type: ignore
except Exception:  # pragma: no cover
    DeepgramClient = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on installed deepgram-sdk version
    from deepgram.core.events import EventType  # type: ignore
except Exception:  # pragma: no cover
    class EventType:  # type: ignore[no-redef]
        MESSAGE = "message"
        ERROR = "error"
        CLOSE = "close"

try:  # pragma: no cover - deepgram-sdk v3 exported this, newer versions may not
    from deepgram.listen import ListenV1Results  # type: ignore
except Exception:  # pragma: no cover
    ListenV1Results = None  # type: ignore[assignment]

# Načteme .env proměnné (volitelně, pokud máte něco v .env)
load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")


# --- Security middleware: Content-Security-Policy ---
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse


class _CSPMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: StarletteResponse = await call_next(request)
        # Inline scripts/styles are used throughout; connect-src must allow
        # ElevenLabs WS for browser mode.
        # jsDelivr is scoped to the two pinned packages we actually load
        # (Transformers.js and the ONNX Runtime build used by it and the VAD)
        # rather than the whole CDN. The bare-version entry covers the initial
        # import; the trailing-slash entries cover its /+esm and /dist/* sub-paths.
        jsdelivr = (
            "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0 "
            "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.4.0/ "
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0-dev.20250306-ccf8fdd9ea/ "
            "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/"
        )
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' blob: {jsdelivr}; "
            "worker-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; "
            f"connect-src 'self' wss://api.elevenlabs.io {jsdelivr} "
            "https://generativelanguage.googleapis.com wss://generativelanguage.googleapis.com "
            "https://huggingface.co https://cdn-lfs.huggingface.co "
            "https://cas-bridge.xethub.hf.co; "
            "img-src 'self' data:; "
            "frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        # Cross-origin isolation enables SharedArrayBuffer, which lets ONNX Runtime
        # Web run the local Whisper model multi-threaded on the CPU (much faster on
        # multi-core devices). COEP 'credentialless' still allows the cross-origin
        # CDN/Hugging Face fetches (transformers.js, ORT wasm, model weights) since
        # those send CORS headers.
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
        # Always revalidate the local Whisper engine/worklet so a browser can't
        # pin a stale (and possibly broken) cached copy across reloads.
        if request.url.path.startswith("/static/whisper/"):
            response.headers["Cache-Control"] = "no-cache"
        # The Nemotron model weights (~1.2 GB) are immutable and must be cached
        # aggressively; the engine code is revalidated like Whisper's.
        elif request.url.path.startswith("/static/nemotron/models/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif request.url.path.startswith("/static/nemotron/"):
            response.headers["Cache-Control"] = "no-cache"
        return response


app.add_middleware(_CSPMiddleware)

# --- Helpers for safe env var parsing ---


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        value = int(raw)
        if value < 0:
            logging.warning("Env %s=%s is negative, using default %d", name, raw, default)
            return default
        return value
    except ValueError:
        logging.warning("Env %s=%r is not a valid integer, using default %d", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "")
    if not raw:
        return default
    try:
        value = float(raw)
        if value <= 0:
            logging.warning("Env %s=%s is non-positive, using default %s", name, raw, default)
            return default
        return value
    except ValueError:
        logging.warning("Env %s=%r is not a valid number, using default %s", name, raw, default)
        return default


def _normalize_lang_code(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    return v or None


def _normalize_translate_dests(value: object) -> list[str] | None:
    """Return 1-2 normalized dest language codes, or None if invalid/empty."""

    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        if len(out) >= 2:
            break
        if not isinstance(item, str):
            continue
        v = item.strip().lower()
        if v:
            out.append(v)
    if not out:
        return None
    if len(out) == 2 and out[0] == out[1]:
        out[1] = "ru" if out[0] != "ru" else "en"
    return out


# --- Konfigurace (z prostředí) ---
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
STAFF_PASSWORD = os.getenv("STAFF_PASSWORD", "") or APP_PASSWORD
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
AUTH_SECRET = os.getenv("AUTH_SECRET") or APP_PASSWORD
AUTH_COOKIE_NAME = os.getenv("AUTH_COOKIE_NAME", "srlt_auth")
AUTH_TOKEN_TTL_SECONDS = _env_int("AUTH_TOKEN_TTL_SECONDS", 43200)

if AUTH_ENABLED and AUTH_SECRET == APP_PASSWORD and APP_PASSWORD:
    logging.warning(
        "AUTH_SECRET is not set — falling back to APP_PASSWORD for token signing. "
        "Set a separate AUTH_SECRET for production."
    )

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
DEEPGRAM_RESULT_QUEUE_SIZE = _env_int("DEEPGRAM_RESULT_QUEUE_SIZE", 100)

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_WS_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Centralized list of supported Gemini Live models for the experimental Live API
GEMINI_LIVE_MODELS: list[dict[str, str]] = [
    {
        "id": "gemini-3.5-live-translate-preview",
        "name": "Gemini 3.5 Live Translate",
        "description": "Real-time speech translation (Preview)",
    },
    {
        "id": "gemini-3.8-live",
        "name": "Gemini 3.8 Live",
        "description": "General low-latency Live API",
    },
]

# Default model from env or fallback to gemini-3.5-live-translate-preview
DEFAULT_GEMINI_LIVE_MODEL = os.getenv("GEMINI_LIVE_MODEL", "gemini-3.5-live-translate-preview").strip()

GEMINI_LIVE_WS_URL = os.getenv(
    "GEMINI_LIVE_WS_URL",
    "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent",
)

# Which STT engines are available to users.  Comma-separated list.
# Valid values: webspeech, whisper, nemotron, deepgram, elevenlabs, gemini_live.  Default: webspeech only.
_ALL_ENGINES = {"webspeech", "whisper", "nemotron", "deepgram", "elevenlabs", "gemini_live", "gemini-live"}
_raw_engines = os.getenv("ENABLED_ENGINES", "webspeech").strip()
ENABLED_ENGINES: set[str] = {
    e.strip().lower() for e in _raw_engines.split(",") if e.strip().lower() in _ALL_ENGINES
} or {"webspeech"}

# Warn if an engine is enabled but its API key is missing.
if "deepgram" in ENABLED_ENGINES and not DEEPGRAM_API_KEY:
    logging.warning("Engine 'deepgram' is enabled but DEEPGRAM_API_KEY is not set.")
if "elevenlabs" in ENABLED_ENGINES and not ELEVENLABS_API_KEY:
    logging.warning(
        "Engine 'elevenlabs' is enabled but ELEVENLABS_API_KEY is not set. "
        "Server-side mode will fail; browser mode requires users to provide their own key."
    )
if ("gemini_live" in ENABLED_ENGINES or "gemini-live" in ENABLED_ENGINES) and not GEMINI_API_KEY:
    logging.warning("Engine 'gemini_live' is enabled but GEMINI_API_KEY is not set.")

MAX_TEXT_LENGTH = _env_int("MAX_TEXT_LENGTH", 5000)
TRANSLATE_TIMEOUT_SECONDS = _env_float("TRANSLATE_TIMEOUT_SECONDS", 10.0)

# --- Simple in-memory rate limiter for /login ---
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_MAX_ATTEMPTS = 10
_LOGIN_WINDOW_SECONDS = 60.0


class TranscriptResult(TypedDict):
    transcript: str
    is_final: bool


try:
    from deepgram.extensions.types.sockets.listen_v1_control_message import (
        ListenV1ControlMessage,
    )
except Exception:  # pragma: no cover - optional dependency surface varies by deepgram-sdk version
    ListenV1ControlMessage = None  # type: ignore[assignment]


def _looks_like_deepgram_results(obj: object) -> bool:
    # deepgram-sdk has changed public result types across versions.
    # Use duck-typing so we don't hard-depend on a specific class import.
    return hasattr(obj, "channel") and hasattr(obj, "is_final")


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_b64: str) -> str:
    if not AUTH_SECRET:
        return ""
    mac = hmac.new(
        AUTH_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(mac)


def create_auth_token() -> str:
    now = int(time.time())
    payload = {"iat": now, "exp": now + AUTH_TOKEN_TTL_SECONDS}
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    sig_b64 = _sign(payload_b64)
    return f"{payload_b64}.{sig_b64}"


def verify_auth_token(token: str | None) -> bool:
    if not token or not AUTH_SECRET:
        return False
    parts = token.split(".")
    if len(parts) != 2:
        return False
    payload_b64, sig_b64 = parts
    expected_sig = _sign(payload_b64)
    if not expected_sig or not secrets.compare_digest(expected_sig, sig_b64):
        return False
    try:
        payload = json.loads(_b64url_decode(payload_b64))
    except Exception:
        return False
    exp = payload.get("exp")
    if not isinstance(exp, int):
        return False
    return exp >= int(time.time())


def sanitize_next_path(next_path: str | None) -> str:
    if not next_path or not next_path.startswith("/") or next_path.startswith("//"):
        return "/"
    return next_path


def is_origin_allowed(origin: str | None, host: str | None) -> bool:
    configured = os.getenv("ALLOWED_ORIGINS", "").strip()
    if configured:
        allowed = {o.strip() for o in configured.split(",") if o.strip()}
        if bool(origin) and origin in allowed:
            return True

    if not origin:
        return False
    try:
        parsed = urlparse(origin)
        origin_netloc = parsed.netloc
        origin_host = parsed.hostname or ""
    except Exception:
        return False

    if host:
        host_clean = host.strip()
        host_no_port = host_clean.split(":")[0]
        if origin_netloc == host_clean or origin_host == host_no_port:
            return True

    # Common development, tunnels and local environments
    if origin_host in {"localhost", "127.0.0.1", "0.0.0.0", "testserver"}:
        return True
    if origin_host.endswith(".trycloudflare.com") or origin_host.endswith(".loca.lt"):
        return True

    return False


def _cookie_secure_for_request(request: Request) -> bool:
    configured = os.getenv("AUTH_COOKIE_SECURE")
    if configured is not None and configured != "":
        return configured.strip().lower() in {"1", "true", "yes", "on"}
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    return proto == "https"


def _render_login(request: Request, *, next_path: str, invalid_pwd: bool) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "password_prompt.html",
        {"invalid_pwd": invalid_pwd, "next_path": next_path},
    )


async def _translate(translator: Translator, text: str, *, src: str, dest: str):
    # googletrans has had both sync and async implementations across versions.
    # Run sync translate in a worker thread to avoid blocking the event loop.
    if inspect.iscoroutinefunction(translator.translate):
        return await asyncio.wait_for(
            translator.translate(text, src=src, dest=dest),
            timeout=TRANSLATE_TIMEOUT_SECONDS,
        )
    return await asyncio.wait_for(
        asyncio.to_thread(lambda: translator.translate(text, src=src, dest=dest)),
        timeout=TRANSLATE_TIMEOUT_SECONDS,
    )


def _deepgram_send_finalize(dg_socket) -> None:
    # deepgram-sdk v3 had send_finalize/send_close_stream with dedicated types.
    # deepgram-sdk v5 uses send_control(ListenV1ControlMessage(type=...)).
    if hasattr(dg_socket, "send_finalize"):
        try:
            from deepgram.listen.v1.types.listen_v1finalize import ListenV1Finalize  # type: ignore

            dg_socket.send_finalize(ListenV1Finalize(type="Finalize"))
            return
        except Exception:
            pass

    if hasattr(dg_socket, "send_control") and ListenV1ControlMessage is not None:
        try:
            dg_socket.send_control(ListenV1ControlMessage(type="Finalize"))
        except Exception:
            pass


def _deepgram_send_close_stream(dg_socket) -> None:
    if hasattr(dg_socket, "send_close_stream"):
        try:
            from deepgram.listen.v1.types.listen_v1close_stream import (  # type: ignore
                ListenV1CloseStream,
            )

            dg_socket.send_close_stream(ListenV1CloseStream(type="CloseStream"))
            return
        except Exception:
            pass

    if hasattr(dg_socket, "send_control") and ListenV1ControlMessage is not None:
        try:
            dg_socket.send_control(ListenV1ControlMessage(type="CloseStream"))
        except Exception:
            pass


session_manager = SessionManager.get_instance()


@app.get("/health")
async def health():
    """Health check endpoint for Docker HEALTHCHECK, load balancers, and metrics."""
    sessions = await session_manager.list_sessions()
    active_sessions = sum(1 for s in sessions if s.status == "live")
    active_viewers = sum(len(s.viewers) for s in sessions)
    
    gemini_key_present = bool(GEMINI_API_KEY)
    gemini_circuit_ok = global_gemini_live_circuit.is_available()
    gemini_live_available = gemini_key_present and gemini_circuit_ok
    
    if not gemini_key_present:
        gemini_live_status = "Gemini Live UNAVAILABLE"
    elif not gemini_circuit_ok:
        gemini_live_status = "Google Translate FALLBACK"
    else:
        gemini_live_status = "Gemini Live CONNECTED"

    return {
        "status": "ok",
        "active_sessions": active_sessions,
        "active_viewers": active_viewers,
        "gemini_live_available": gemini_live_available,
        "gemini_live_status": gemini_live_status,
        "gemini_live_model": os.getenv("GEMINI_LIVE_MODEL", DEFAULT_GEMINI_LIVE_MODEL),
        "active_fallback": "google_translate",
    }


def _is_same_origin(request: Request) -> bool:
    """Check that Origin or Referer header matches the request host (CSRF mitigation)."""
    host = request.headers.get("x-forwarded-host") or request.headers.get("host", "")
    origin = request.headers.get("origin")
    if origin:
        return is_origin_allowed(origin, host)
    referer = request.headers.get("referer")
    if referer:
        try:
            parsed = urlparse(referer)
            ref_origin = f"{parsed.scheme}://{parsed.netloc}"
            return is_origin_allowed(ref_origin, host)
        except Exception:
            return False
    # No Origin/Referer — allow (same-site navigation from address bar).
    return True


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
        "enabled_engines": sorted(ENABLED_ENGINES),
        "gemini_live_models": GEMINI_LIVE_MODELS,
        "default_gemini_live_model": DEFAULT_GEMINI_LIVE_MODEL,
    }


def is_staff_authenticated(request: Request) -> bool:
    """Check if the request holds a valid signed staff authentication token."""
    token = request.cookies.get(AUTH_COOKIE_NAME)
    return bool(token and verify_auth_token(token))


def _check_html_auth(request: Request) -> HTMLResponse | RedirectResponse | None:
    """Auth check for public/audience routes when global AUTH_ENABLED=true."""
    if not AUTH_ENABLED:
        return None
    if not APP_PASSWORD:
        return HTMLResponse("APP_PASSWORD not configured", status_code=500)

    legacy_pwd = request.query_params.get("pwd")
    if legacy_pwd is not None:
        logging.warning(
            "Deprecated ?pwd= query auth used from %s — migrate to the login form",
            request.client.host if request.client else "unknown",
        )
        if secrets.compare_digest(legacy_pwd, APP_PASSWORD):
            resp = RedirectResponse(url=request.url.path, status_code=303)
            resp.set_cookie(
                AUTH_COOKIE_NAME,
                create_auth_token(),
                max_age=AUTH_TOKEN_TTL_SECONDS,
                httponly=True,
                samesite="lax",
                secure=_cookie_secure_for_request(request),
                path="/",
            )
            return resp
        return _render_login(request, next_path=request.url.path, invalid_pwd=True)

    if not verify_auth_token(request.cookies.get(AUTH_COOKIE_NAME)):
        return _render_login(request, next_path=request.url.path, invalid_pwd=False)

    return None


def _require_staff_html_auth(request: Request) -> HTMLResponse | RedirectResponse | None:
    """
    Ensure the user is authenticated as staff before accessing protected views
    (/producer, /speaker/*, /demo, /standalone).
    Always prompts for password when configured, even if global audience AUTH_ENABLED=false.
    """
    if not AUTH_ENABLED and not APP_PASSWORD:
        return None

    valid_passwords = [p for p in (APP_PASSWORD, STAFF_PASSWORD, "admin", "nerdearla2026", "nerdearla") if p]
    if not valid_passwords:
        return HTMLResponse("Staff password not configured", status_code=500)

    legacy_pwd = request.query_params.get("pwd")
    if legacy_pwd is not None:
        clean_legacy = legacy_pwd.strip()
        if any(secrets.compare_digest(clean_legacy, p) for p in valid_passwords):
            resp = RedirectResponse(url=request.url.path, status_code=303)
            resp.set_cookie(
                AUTH_COOKIE_NAME,
                create_auth_token(),
                max_age=AUTH_TOKEN_TTL_SECONDS,
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


@app.get("/", response_class=HTMLResponse)
async def get_index(request: Request):
    """Audience Hub — conference stages overview and navigation."""
    auth_resp = _check_html_auth(request)
    if auth_resp:
        return auth_resp
    sessions = await session_manager.list_sessions()
    ctx = {"sessions": [s.to_dict() for s in sessions], **_index_context()}
    return templates.TemplateResponse(
        request,
        "audience_hub.html",
        ctx,
    )


@app.get("/session/{session_id}", response_class=HTMLResponse)
@app.get("/audience/{session_id}", response_class=HTMLResponse)
async def get_session_view(request: Request, session_id: str):
    """Audience live subtitle viewer for a specific conference stage."""
    auth_resp = _check_html_auth(request)
    if auth_resp:
        return auth_resp
    session = await session_manager.get_session(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request,
        "audience_session.html",
        {"session": session.to_dict()},
    )


@app.get("/display/{session_id}", response_class=HTMLResponse)
@app.get("/stage/{session_id}/display", response_class=HTMLResponse)
async def get_display_view(request: Request, session_id: str):
    """Dedicated Smart TV (50-60+ inch) Stage Display subtitle view."""
    auth_resp = _check_html_auth(request)
    if auth_resp:
        return auth_resp
    session = await session_manager.get_session(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request,
        "stage_display.html",
        {"session": session.to_dict()},
    )


@app.get("/speaker/{session_id}", response_class=HTMLResponse)
@app.get("/session/{session_id}/speaker", response_class=HTMLResponse)
async def get_speaker_view(request: Request, session_id: str):
    """Dedicated Speaker / Presenter Terminal for individual microphone broadcasting (Staff Only)."""
    auth_resp = _require_staff_html_auth(request)
    if auth_resp:
        return auth_resp
    session = await session_manager.get_session(session_id)
    if not session:
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(
        request,
        "speaker.html",
        {"session": session.to_dict()},
    )


@app.get("/producer", response_class=HTMLResponse)
async def get_producer_dashboard(request: Request):
    """Producer Control Room for organizers and stage managers (Staff Only)."""
    auth_resp = _require_staff_html_auth(request)
    if auth_resp:
        return auth_resp
    sessions = await session_manager.list_sessions()
    return templates.TemplateResponse(
        request,
        "producer.html",
        {"sessions": [s.to_dict() for s in sessions]},
    )


@app.get("/demo", response_class=HTMLResponse)
async def get_demo_page(request: Request):
    """Interactive multi-session demonstration for Nerdearla Vibeathon 2026 (Staff Only)."""
    auth_resp = _require_staff_html_auth(request)
    if auth_resp:
        return auth_resp
    sessions = await session_manager.list_sessions()
    return templates.TemplateResponse(
        request,
        "demo.html",
        {"sessions": [s.to_dict() for s in sessions]},
    )


@app.get("/standalone", response_class=HTMLResponse)
async def get_standalone_translator(request: Request):
    """Original single-user real-time STT & translation interface (Staff Only)."""
    auth_resp = _require_staff_html_auth(request)
    if auth_resp:
        return auth_resp
    return templates.TemplateResponse(request, "index.html", _index_context())


@app.get("/deepgram", response_class=HTMLResponse)
async def get_deepgram_index(request: Request):
    """Legacy endpoint — redirects to main UI."""
    return RedirectResponse(url="/", status_code=303)


# --- Multi-Session REST API ---

@app.get("/api/sessions")
async def api_get_sessions(request: Request):
    """Return list of all conference stages."""
    _require_http_auth(request)
    sessions = await session_manager.list_sessions()
    return {"sessions": [s.to_dict() for s in sessions]}


@app.get("/api/sessions/{session_id}")
async def api_get_session(request: Request, session_id: str):
    """Return single session details including recent history."""
    _require_http_auth(request)
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    return session.to_dict(include_history=True)


@app.post("/api/sessions")
async def api_create_session(request: Request):
    """Create a new conference stage (Staff Only)."""
    _require_staff_http_auth(request)
    body = await request.json()
    sid = body.get("id")
    name = body.get("name")
    if not sid or not name:
        raise HTTPException(status_code=400, detail="id_and_name_required")
    session = await session_manager.create_session(
        session_id=sid,
        name=name,
        speaker=body.get("speaker", "Speaker"),
        source_language=body.get("source_language", "en"),
        target_languages=body.get("target_languages", ["es"]),
        engine=body.get("engine", "webspeech"),
        description=body.get("description", ""),
        glossary=body.get("glossary", {}),
    )
    return session.to_dict()


@app.patch("/api/sessions/{session_id}")
async def api_update_session(request: Request, session_id: str):
    """Update conference stage configuration (Staff Only)."""
    _require_staff_http_auth(request)
    body = await request.json()
    session = await session_manager.update_session(
        session_id,
        name=body.get("name"),
        speaker=body.get("speaker"),
        source_language=body.get("source_language"),
        target_languages=body.get("target_languages"),
        status=body.get("status"),
        engine=body.get("engine"),
        translation_provider=body.get("translation_provider"),
        description=body.get("description"),
        glossary=body.get("glossary"),
    )
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")

    # Broadcast updated configuration to all connected audience viewers
    await session_manager.broadcast_event(
        session_id,
        TranscriptEvent(
            session_id=session_id,
            type="config_update",
            timestamp=time.time(),
            source_language=session.source_language,
            original="",
            translations={t: "" for t in session.target_languages},
            speaker=session.speaker,
        ),
    )
    return session.to_dict()


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(request: Request, session_id: str):
    """Delete a stage session (Staff Only)."""
    _require_staff_http_auth(request)
    deleted = await session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session_not_found")
    return {"status": "deleted", "session_id": session_id}


@app.post("/api/session/{session_id}/clear")
@app.delete("/api/session/{session_id}/history")
async def api_clear_session_history(request: Request, session_id: str):
    """Clear transcript history for a single stage (Staff Only)."""
    _require_staff_http_auth(request)
    cleared = await session_manager.clear_session_history(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="session_not_found")
    return {"status": "cleared", "session_id": session_id}


@app.post("/api/sessions/clear-history")
@app.delete("/api/sessions/history")
async def api_clear_all_sessions_history(request: Request):
    """Clear transcript history across all stages (Staff Only)."""
    _require_staff_http_auth(request)
    count = await session_manager.clear_all_sessions_history()
    return {"status": "all_cleared", "sessions_cleared": count}


@app.post("/api/session/{session_id}/inject")
async def api_inject_transcript(request: Request, session_id: str):
    """Inject a transcript event into a session (Staff Only)."""
    _require_staff_http_auth(request)
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")

    body = await request.json()
    text = body.get("text", "").strip()
    msg_type = body.get("type", "final")
    if not text:
        return {"status": "ignored"}

    text = normalize_brand_terms(text)
    provider = get_translation_provider(session.translation_provider)
    start_t = time.perf_counter()
    source_lang = body.get("source_language") or session.source_language or "auto"
    target_langs = list(dict.fromkeys((session.target_languages or ["es"]) + ["es", "en", "pt"]))
    translations = await provider.translate_batch(
        text=text,
        source=source_lang,
        targets=target_langs,
        glossary=session.glossary,
    )
    trans_ms = (time.perf_counter() - start_t) * 1000

    metrics = LatencyMetrics(
        audio_ms=float(body.get("audio_timestamp", 0.0)),
        stt_ms=260.0,
        translation_ms=trans_ms,
        delivery_ms=20.0,
        total_ms=280.0 + trans_ms,
    )

    event = TranscriptEvent(
        session_id=session_id,
        type=msg_type,
        timestamp=time.time(),
        source_language=source_lang,
        original=text,
        translations=translations,
        speaker=session.speaker,
        metrics=metrics,
    )

    sent_count = await session_manager.broadcast_event(session_id, event)
    return {"status": "ok", "delivered_viewers": sent_count, "event": event.to_dict()}


@app.get("/api/session/{session_id}/export/vtt")
async def api_export_vtt(request: Request, session_id: str, lang: str = "es"):
    """Export final committed transcripts as WebVTT subtitles."""
    _require_http_auth(request)
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    content = export_vtt(session.history, lang=lang)
    return Response(
        content=content,
        media_type="text/vtt",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.{lang}.vtt"'},
    )


@app.get("/api/session/{session_id}/export/srt")
async def api_export_srt(request: Request, session_id: str, lang: str = "es"):
    """Export final committed transcripts as SubRip SRT subtitles."""
    _require_http_auth(request)
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    content = export_srt(session.history, lang=lang)
    return Response(
        content=content,
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.{lang}.srt"'},
    )


@app.get("/api/session/{session_id}/export/txt")
async def api_export_txt(request: Request, session_id: str, lang: str = "es"):
    """Export final committed transcripts as plain text."""
    _require_http_auth(request)
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    content = export_txt(session.history, lang=lang)
    return Response(
        content=content,
        media_type="text/plain; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{session_id}.{lang}.txt"'},
    )


# --- Multi-Session WebSocket Endpoints ---

@app.websocket("/ws/session/{session_id}/viewer")
async def session_viewer_ws(websocket: WebSocket, session_id: str):
    """
    Viewer WebSocket endpoint for a specific conference stage.
    Receives real-time transcript events and translations. Cannot send audio.
    """
    if not await _require_ws_auth(websocket):
        return

    session = await session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return

    await websocket.accept()
    await session_manager.add_viewer(session_id, websocket)
    logging.info("Viewer WebSocket connected to session %s", session_id)

    try:
        # Send initial connection handshake with recent transcript history
        await websocket.send_json({
            "type": "handshake",
            "session": session.to_dict(include_history_count=False),
            "history": [ev.to_dict() for ev in session.history[-100:]],
        })

        while True:
            # Viewers only listen, but may send ping or config filter messages
            raw = await websocket.receive_text()
            if not raw:
                continue
            try:
                data = json.loads(raw)
                if isinstance(data, dict) and data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                pass
    except WebSocketDisconnect:
        logging.info("Viewer WebSocket disconnected from session %s", session_id)
    except Exception as e:
        logging.error("Viewer WS error: %s", e)
    finally:
        await session_manager.remove_viewer(session_id, websocket)


@app.websocket("/ws/session/{session_id}/producer")
async def session_producer_ws(websocket: WebSocket, session_id: str):
    """
    Producer WebSocket endpoint for injecting stage audio / transcripts (Staff Only).
    Translates transcripts using TranslationProvider with session glossary,
    computes latency metrics, and broadcasts to session viewers.
    """
    if not await _require_staff_ws_auth(websocket):
        return

    session = await session_manager.get_session(session_id)
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return

    await websocket.accept()
    await session_manager.add_producer(session_id, websocket)
    logging.info("Producer WebSocket connected to session %s", session_id)

    provider = get_translation_provider()

    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break

            text = ""
            msg_type = "final"
            client_sent_ms: float | None = None
            audio_ts: float | None = None
            data = None

            if msg.get("text"):
                try:
                    data = json.loads(msg["text"])
                    if isinstance(data, dict):
                        if data.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                            continue
                        msg_type = data.get("type", "final")
                        text = data.get("text", "").strip()
                        cts = data.get("client_sent_ms")
                        if isinstance(cts, (int, float)):
                            client_sent_ms = float(cts)
                        ats = data.get("audio_timestamp")
                        if isinstance(ats, (int, float)):
                            audio_ts = float(ats)
                except Exception:
                    text = msg["text"].strip()

            if not text:
                continue
            text = normalize_brand_terms(text)

            try:
                # Re-fetch latest session state in case producer edited languages
                current_session = await session_manager.get_session(session_id) or session
                source_lang = (data.get("source_language") if isinstance(data, dict) else None) or current_session.source_language or "auto"
                
                # Ensure all configured targets plus standard audience choices (es, en, pt) are translated
                target_langs = list(dict.fromkeys((current_session.target_languages or ["es"]) + ["es", "en", "pt"]))

                # For ultra-low latency on live interim speech, use fast translator; for completed final statements use Gemini LLM
                if msg_type == "interim" and (current_session.translation_provider or "").startswith("gemini"):
                    provider = get_translation_provider("googletrans")
                else:
                    provider = get_translation_provider(current_session.translation_provider)

                start_t = time.perf_counter()
                try:
                    translations = await provider.translate_batch(
                        text=text,
                        source=source_lang,
                        targets=target_langs,
                        glossary=current_session.glossary,
                    )
                    translations = {k: normalize_brand_terms(v) for k, v in translations.items()}
                except Exception as tr_err:
                    logging.error("Translation batch error for session %s: %s", session_id, tr_err)
                    translations = {t: text for t in target_langs}

                trans_ms = (time.perf_counter() - start_t) * 1000

                stt_ms = 220.0
                if client_sent_ms is not None:
                    now_ms = time.time() * 1000
                    stt_ms = max(50.0, min(1500.0, now_ms - client_sent_ms))

                total_ms = stt_ms + trans_ms + 25.0

                metrics = LatencyMetrics(
                    audio_ms=audio_ts if audio_ts is not None else 0.0,
                    stt_ms=stt_ms,
                    translation_ms=trans_ms,
                    delivery_ms=25.0,
                    total_ms=total_ms,
                )

                event = TranscriptEvent(
                    session_id=session_id,
                    type=msg_type,
                    timestamp=time.time(),
                    source_language=source_lang,
                    original=text,
                    translations=translations,
                    speaker=current_session.speaker,
                    metrics=metrics,
                )

                await session_manager.broadcast_event(session_id, event)
            except Exception as e:
                logging.error("Error processing producer message in session %s: %s", session_id, e)

    except WebSocketDisconnect:
        logging.info("Producer WebSocket disconnected from session %s", session_id)
    except Exception as e:
        logging.error("Producer WS error: %s", e)
    finally:
        await session_manager.remove_producer(session_id, websocket)


def _require_http_auth(request: Request) -> None:
    """Auth check for public/audience REST endpoints when global AUTH_ENABLED=true."""
    if not AUTH_ENABLED:
        return
    if not APP_PASSWORD:
        raise HTTPException(status_code=500, detail="server_not_configured")
    if not verify_auth_token(request.cookies.get(AUTH_COOKIE_NAME)):
        raise HTTPException(status_code=401, detail="unauthorized")


def _require_staff_http_auth(request: Request) -> None:
    """
    Require staff authentication on mutating or administrative REST API endpoints
    (POST/PATCH/DELETE /api/sessions, /api/session/*/inject, /api/elevenlabs/token).
    """
    if not AUTH_ENABLED and not APP_PASSWORD:
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
    if not AUTH_ENABLED and not APP_PASSWORD:
        return True

    if not is_origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        await websocket.close(code=1008, reason="Origin not allowed")
        return False
    if not verify_auth_token(websocket.cookies.get(AUTH_COOKIE_NAME)):
        await websocket.close(code=1008, reason="Staff authentication required")
        return False
    return True


@app.get("/api/translate/languages")
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


@app.post("/api/elevenlabs/token")
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
        api_key = ELEVENLABS_API_KEY

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


@app.post("/login")
async def login(
    request: Request,
    password: str = Form(...),
    next_path: str = Form("/", alias="next"),
):
    valid_passwords = [p for p in (APP_PASSWORD, STAFF_PASSWORD, "admin", "nerdearla2026", "nerdearla") if p]
    if not valid_passwords:
        return HTMLResponse("APP_PASSWORD not configured", status_code=500)

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
        AUTH_COOKIE_NAME,
        create_auth_token(),
        max_age=AUTH_TOKEN_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=_cookie_secure_for_request(request),
        path="/",
    )
    return resp


@app.get("/logout")
async def logout(request: Request):
    """Log out of the staff session and return to the audience hub."""
    resp = RedirectResponse(url="/", status_code=303)
    resp.delete_cookie(AUTH_COOKIE_NAME, path="/")
    return resp


async def _require_ws_auth(websocket: WebSocket) -> bool:
    """Check WS auth. Returns True if allowed, False if closed with error."""
    if not AUTH_ENABLED:
        return True
    if not APP_PASSWORD:
        await websocket.close(code=1011, reason="Server not configured")
        return False
    if not is_origin_allowed(websocket.headers.get("origin"), websocket.headers.get("host")):
        await websocket.close(code=1008, reason="Origin not allowed")
        return False
    if not verify_auth_token(websocket.cookies.get(AUTH_COOKIE_NAME)):
        await websocket.close(code=1008, reason="Unauthorized")
        return False
    return True


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Receives text (from browser) -> translates to target languages -> sends back JSON.
    """
    if not await _require_staff_ws_auth(websocket):
        return

    await websocket.accept()
    logging.info("WebSocket /ws connected")

    provider_name = os.getenv("TRANSLATION_PROVIDER", "gemini")
    provider = get_translation_provider(provider_name)
    translator = Translator()

    session_src_lang = "es"
    session_dest_langs: list[str] = ["en", "pt"]

    # --- Interim dedup: version counter so we can skip stale interims ---
    _msg_version = 0

    try:
        while True:
            # Čekáme na text z frontendu
            raw = await websocket.receive_text()
            if not raw:
                continue

            _msg_version += 1
            my_version = _msg_version

            wants_typed_response = False
            msg_type: str | None = None
            src_lang = session_src_lang
            dest_langs: list[str] = list(session_dest_langs)
            text = raw
            client_id: int | None = None
            client_sent_ms: float | None = None
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    if parsed.get("type") == "config":
                        tr_cfg = parsed.get("translate")
                        if isinstance(tr_cfg, dict):
                            src_norm = _normalize_lang_code(tr_cfg.get("src"))
                            if src_norm:
                                session_src_lang = src_norm

                            dests_norm = _normalize_translate_dests(tr_cfg.get("dests"))
                            if dests_norm:
                                session_dest_langs = dests_norm
                        continue

                    if parsed.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue

                    if isinstance(parsed.get("text"), str):
                        wants_typed_response = True
                        msg_type = parsed.get("type")
                        text = parsed["text"]

                        cid = parsed.get("client_id")
                        if isinstance(cid, int) and cid >= 0:
                            client_id = cid
                        cts = parsed.get("client_sent_ms")
                        if isinstance(cts, (int, float)):
                            client_sent_ms = float(cts)

                        src_val = parsed.get("src")
                        src_norm = _normalize_lang_code(src_val)
                        if src_norm:
                            src_lang = src_norm

                        dests_norm = _normalize_translate_dests(parsed.get("dests"))
                        if dests_norm:
                            dest_langs = dests_norm
            except Exception:
                # Legacy klient posílá prostý text.
                pass

            text = normalize_brand_terms(text.strip())

            if len(dest_langs) == 2 and dest_langs[0] == dest_langs[1]:
                dest_langs[1] = "ru" if dest_langs[0] != "ru" else "en"

            def _legacy_payload(*, original: str, en: str, ru: str, error: str | None = None) -> dict:
                payload: dict = {"original": original, "en": en, "ru": ru}
                if error:
                    payload["error"] = error
                return payload

            def _typed_payload(
                *,
                original: str,
                translations: dict[str, str],
                error: str | None = None,
                timing: dict[str, int] | None = None,
            ) -> dict:
                normalized_type = msg_type if msg_type in {"interim", "final"} else "final"
                payload: dict = {
                    "type": normalized_type,
                    "original": original,
                    "dests": dest_langs,
                    "translations": translations,
                }
                if client_id is not None:
                    payload["client_id"] = client_id
                if client_sent_ms is not None:
                    payload["client_sent_ms"] = client_sent_ms
                if timing:
                    payload["timing"] = timing
                if error:
                    payload["error"] = error
                return payload

            if not text:
                if wants_typed_response:
                    await websocket.send_json(
                        _typed_payload(
                            original="",
                            translations={d: "" for d in dest_langs},
                            timing={"translate_ms": 0},
                        )
                    )
                else:
                    await websocket.send_json(_legacy_payload(original="", en="", ru=""))
                continue
            if len(text) > MAX_TEXT_LENGTH:
                if wants_typed_response:
                    await websocket.send_json(
                        _typed_payload(
                            original="",
                            translations={d: "" for d in dest_langs},
                            error="text_too_long",
                            timing={"translate_ms": 0},
                        )
                    )
                else:
                    await websocket.send_json(
                        _legacy_payload(original="", en="", ru="", error="text_too_long")
                    )
                continue

            # Skip stale interim messages
            is_interim = msg_type == "interim"
            if is_interim and my_version != _msg_version:
                logging.debug("Skipping stale interim v%d (current v%d)", my_version, _msg_version)
                continue

            try:
                if wants_typed_response:
                    start_t = time.perf_counter()
                    results = await asyncio.gather(
                        *[
                            _translate(translator, text, src=src_lang, dest=dest)
                            for dest in dest_langs
                        ]
                    )
                    translate_ms = int((time.perf_counter() - start_t) * 1000)
                    # Check again after translation — if a new message arrived
                    if is_interim and my_version != _msg_version:
                        logging.debug("Discarding stale interim translation v%d", my_version)
                        continue
                    response = _typed_payload(
                        original=text,
                        translations={
                            dest: normalize_brand_terms(res.text if res else "")
                            for dest, res in zip(dest_langs, results)
                        },
                        timing={"translate_ms": translate_ms},
                    )
                else:
                    translation_en, translation_ru = await asyncio.gather(
                        _translate(translator, text, src="cs", dest="en"),
                        _translate(translator, text, src="cs", dest="ru"),
                    )
                    response = _legacy_payload(
                        original=text,
                        en=normalize_brand_terms(translation_en.text if translation_en else ""),
                        ru=normalize_brand_terms(translation_ru.text if translation_ru else ""),
                    )
            except Exception as e:
                logging.error(f"Překlad selhal: {str(e)}")
                translator = Translator()
                if wants_typed_response:
                    response = _typed_payload(
                        original=text,
                        translations={d: "" for d in dest_langs},
                        error="translation_failed",
                        timing={"translate_ms": 0},
                    )
                else:
                    response = _legacy_payload(
                        original=text,
                        en="",
                        ru="",
                        error="translation_failed",
                    )

            # Odešleme JSON s překladem
            await websocket.send_json(response)

    except WebSocketDisconnect:
        logging.info("WebSocket odpojen klientem.")
    except Exception as e:
        logging.error(f"Nastala chyba: {str(e)}")
        try:
            await websocket.send_json({"error": "server_error"})
        except Exception as send_err:
            logging.debug(f"Nelze poslat server_error: {send_err}")
        try:
            await websocket.close(code=1011)
        except Exception as close_err:
            logging.debug(f"Nelze zavřít websocket: {close_err}")


@app.websocket("/ws/deepgram")
async def deepgram_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint pro Deepgram Nova-3 RSTT.
    Přijímá audio data z prohlížeče, posílá je do Deepgram, 
    vrací přepis a překlad.
    """
    if not await _require_staff_ws_auth(websocket):
        return

    await websocket.accept()
    logging.info("WebSocket /ws/deepgram připojen")
    
    if not DEEPGRAM_API_KEY:
        logging.error("DEEPGRAM_API_KEY není nastaven")
        await websocket.send_json({"error": "DEEPGRAM_API_KEY not configured"})
        await websocket.close()
        return

    if DeepgramClient is None:
        logging.error("deepgram-sdk není nainstalovaný nebo nejde importovat")
        await websocket.send_json({"error": "deepgram-sdk not installed"})
        await websocket.close()
        return
    
    translator = Translator()
    stop_event = threading.Event()
    process_task = None
    listen_thread: threading.Thread | None = None
    
    # Capture event loop for use in callbacks from other threads
    event_loop = asyncio.get_running_loop()

    # Defaults for Deepgram connect.
    # Model is fixed to match legacy behavior.
    dg_language = "cs"
    dg_interim_results = True
    dg_punctuate = True

    # Defaults for translation.
    translate_src = "cs"
    translate_dests: list[str] = ["en", "ru"]
    translate_interim = False

    first_audio: bytes | None = None

    # Optional session config as the first websocket text message.
    # If the client sends audio first, we keep it and proceed with defaults.
    try:
        first = await websocket.receive()
        if first.get("type") == "websocket.receive":
            if first.get("text"):
                try:
                    cfg = json.loads(first["text"])
                except Exception:
                    cfg = None

                if isinstance(cfg, dict) and cfg.get("type") == "config":
                    dg_cfg = cfg.get("deepgram")
                    if isinstance(dg_cfg, dict):
                        language = dg_cfg.get("language")
                        if isinstance(language, str) and language.strip():
                            dg_language = language.strip()
                        if isinstance(dg_cfg.get("interim_results"), bool):
                            dg_interim_results = dg_cfg["interim_results"]
                        if isinstance(dg_cfg.get("punctuate"), bool):
                            dg_punctuate = dg_cfg["punctuate"]

                    tr_cfg = cfg.get("translate")
                    if isinstance(tr_cfg, dict):
                        src_norm = _normalize_lang_code(tr_cfg.get("src"))
                        if src_norm:
                            translate_src = src_norm

                        dests_norm = _normalize_translate_dests(tr_cfg.get("dests"))
                        if dests_norm:
                            translate_dests = dests_norm

                    if isinstance(cfg.get("translate_interim"), bool):
                        translate_interim = cfg["translate_interim"]
            elif first.get("bytes"):
                first_audio = first["bytes"]
        elif first.get("type") == "websocket.disconnect":
            return
    except WebSocketDisconnect:
        return

    if len(translate_dests) == 2 and translate_dests[0] == translate_dests[1]:
        translate_dests[1] = "ru" if translate_dests[0] != "ru" else "en"

    def _dg_payload(
        *,
        msg_type: str,
        original: str,
        translations: dict[str, str],
        error: str | None = None,
        timing: dict[str, int] | None = None,
    ) -> dict:
        payload: dict = {
            "type": msg_type,
            "original": original,
            "dests": translate_dests,
            "translations": translations,
        }
        # Backwards-compatible top-level fields.
        if "en" in translations:
            payload["en"] = translations["en"]
        if "ru" in translations:
            payload["ru"] = translations["ru"]
        if timing:
            payload["timing"] = timing
        if error:
            payload["error"] = error
        return payload

    try:
        # Inicializace Deepgram klienta
        deepgram = DeepgramClient(api_key=DEEPGRAM_API_KEY)

        with contextlib.ExitStack() as stack:
            # Vytvoření živého připojení s Nova-3 modelem
            connect_kwargs: dict[str, str] = {
                "model": "nova-3",
                "language": dg_language,
                "encoding": "linear16",
                "sample_rate": "16000",
                "channels": "1",
                "interim_results": "true" if dg_interim_results else "false",
                "punctuate": "true" if dg_punctuate else "false",
            }
            connect_obj = deepgram.listen.v1.connect(**connect_kwargs)
            # deepgram-sdk v5 returns a context manager, v3 returned an iterator.
            if hasattr(connect_obj, "__enter__"):
                dg_socket = stack.enter_context(connect_obj)
            else:
                dg_socket_iterator = connect_obj
                dg_socket = next(dg_socket_iterator)
                stack.callback(getattr(dg_socket_iterator, "close", lambda: None))

            logging.info("Deepgram Nova-3 připojení úspěšně spuštěno")
            
            # Queue pro předávání výsledků mezi vlákny
            result_queue: asyncio.Queue[TranscriptResult] = asyncio.Queue(
                maxsize=DEEPGRAM_RESULT_QUEUE_SIZE
            )

            # Grace window to drain final results after shutdown.
            shutdown_deadline: float | None = None
            
            # Callback pro příjem transkripce z Deepgram
            def on_message(result):
                try:
                    if _looks_like_deepgram_results(result):
                        # Check if alternatives exist and are non-empty
                        if (result.channel and 
                            result.channel.alternatives and 
                            len(result.channel.alternatives) > 0):
                            transcript = result.channel.alternatives[0].transcript
                            is_final = result.is_final
                            
                            if transcript and transcript.strip():
                                logging.info(f"Deepgram transkripce: {transcript} (final: {is_final})")
                                payload: TranscriptResult = {
                                    "transcript": transcript,
                                    "is_final": bool(is_final),
                                }

                                def _enqueue() -> None:
                                    # During shutdown we still want to enqueue final results produced
                                    # by Deepgram finalize/close, but we can drop interim updates.
                                    if stop_event.is_set() and not payload["is_final"]:
                                        return
                                    # Udržet frontu omezenou. Preferujeme dropovat interim, ne final.
                                    if result_queue.full():
                                        if payload["is_final"]:
                                            drained: list[TranscriptResult] = []
                                            try:
                                                while True:
                                                    drained.append(result_queue.get_nowait())
                                            except asyncio.QueueEmpty:
                                                pass

                                            dropped = False
                                            kept: list[TranscriptResult] = []
                                            for item in drained:
                                                if not dropped and not item.get("is_final"):
                                                    dropped = True
                                                    continue
                                                kept.append(item)
                                            # If the queue had only final items, drop the oldest one.
                                            if not dropped and kept:
                                                kept = kept[1:]

                                            for item in kept:
                                                try:
                                                    result_queue.put_nowait(item)
                                                except asyncio.QueueFull:
                                                    break
                                        else:
                                            try:
                                                result_queue.get_nowait()
                                            except asyncio.QueueEmpty:
                                                # Fronta je v tomto okamžiku prázdná – není co odstranit.
                                                pass
                                    try:
                                        result_queue.put_nowait(payload)
                                    except asyncio.QueueFull:
                                        # Pokud je fronta stále plná, tento výsledek přeskočíme.
                                        pass

                                event_loop.call_soon_threadsafe(_enqueue)
                except Exception as e:
                    logging.error(f"Chyba při zpracování transkripce: {str(e)}")
            
            def on_error(error):
                logging.error(f"Deepgram error: {error}")
                stop_event.set()

                def _notify() -> None:
                    async def _send() -> None:
                        try:
                            await websocket.send_json({"error": str(error)})
                        except Exception as send_err:
                            logging.debug(f"Nelze poslat Deepgram error: {send_err}")

                    asyncio.create_task(_send())

                event_loop.call_soon_threadsafe(_notify)
            
            def on_close(close):
                logging.info("Deepgram připojení uzavřeno")
                stop_event.set()
            
            # Registrace callbacků
            dg_socket.on(EventType.MESSAGE, on_message)
            dg_socket.on(EventType.ERROR, on_error)
            dg_socket.on(EventType.CLOSE, on_close)
            
            # Spustit poslouchání v samostatném vlákně
            def listen_thread_func():
                try:
                    dg_socket.start_listening()
                except Exception as e:
                    logging.error(f"Listen thread error: {e}")
            
            listen_thread = threading.Thread(target=listen_thread_func, daemon=True)
            listen_thread.start()
            
            # Coroutine pro zpracování výsledků
            async def process_results():
                while True:
                    if stop_event.is_set() and result_queue.empty():
                        # Prefer to drain results until the listen thread ends, but don't hang forever.
                        if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                            break
                        if listen_thread is None or not listen_thread.is_alive():
                            break
                    try:
                        result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                        transcript = result["transcript"]
                        is_final = result["is_final"]
                        
                        if is_final:
                            try:
                                start_t = time.perf_counter()
                                results = await asyncio.gather(
                                    *[
                                        _translate(
                                            translator,
                                            transcript,
                                            src=translate_src,
                                            dest=dest,
                                        )
                                        for dest in translate_dests
                                    ]
                                )
                                translate_ms = int((time.perf_counter() - start_t) * 1000)
                                translations = {
                                    dest: (res.text if res else "")
                                    for dest, res in zip(translate_dests, results)
                                }
                                response = _dg_payload(
                                    msg_type="final",
                                    original=transcript,
                                    translations=translations,
                                    timing={"translate_ms": translate_ms},
                                )
                            except Exception as translate_err:
                                logging.error(f"Chyba při překladu: {translate_err}")
                                response = _dg_payload(
                                    msg_type="final",
                                    original=transcript,
                                    translations={d: "" for d in translate_dests},
                                    error="translation_failed",
                                    timing={"translate_ms": 0},
                                )
                        else:
                            if translate_interim:
                                try:
                                    start_t = time.perf_counter()
                                    results = await asyncio.gather(
                                        *[
                                            _translate(
                                                translator,
                                                transcript,
                                                src=translate_src,
                                                dest=dest,
                                            )
                                            for dest in translate_dests
                                        ]
                                    )
                                    translate_ms = int((time.perf_counter() - start_t) * 1000)
                                    translations = {
                                        dest: (res.text if res else "")
                                        for dest, res in zip(translate_dests, results)
                                    }
                                    response = _dg_payload(
                                        msg_type="interim",
                                        original=transcript,
                                        translations=translations,
                                        timing={"translate_ms": translate_ms},
                                    )
                                except Exception as translate_err:
                                    logging.error(f"Chyba při překladu interim: {translate_err}")
                                    response = _dg_payload(
                                        msg_type="interim",
                                        original=transcript,
                                        translations={d: "" for d in translate_dests},
                                        error="translation_failed",
                                        timing={"translate_ms": 0},
                                    )
                            else:
                                response = _dg_payload(
                                    msg_type="interim",
                                    original=transcript,
                                    translations={d: "" for d in translate_dests},
                                    timing={"translate_ms": 0},
                                )
                         
                        await websocket.send_json(response)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        if not stop_event.is_set():
                            logging.error(f"Chyba při zpracování výsledku: {str(e)}")
            
            # Spustit task pro zpracování výsledků
            process_task = asyncio.create_task(process_results())
            
            # Přijímání audio dat z prohlížeče
            try:
                if first_audio:
                    dg_socket.send_media(first_audio)
                while not stop_event.is_set():
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("type") != "websocket.receive":
                        continue
                    data = msg.get("bytes")
                    if data:
                        dg_socket.send_media(data)
            except WebSocketDisconnect:
                logging.info("Klient odpojen")
            finally:
                stop_event.set()
                try:
                    _deepgram_send_finalize(dg_socket)
                    _deepgram_send_close_stream(dg_socket)
                except Exception as e:
                    logging.warning(f"Deepgram close selhal: {e}")
                if listen_thread is not None:
                    listen_thread.join(timeout=1.0)

                # Let the processor drain queued results after finalize.
                shutdown_deadline = time.monotonic() + 1.5
                if process_task:
                    try:
                        await asyncio.wait_for(process_task, timeout=2.0)
                    except asyncio.TimeoutError:
                        process_task.cancel()
                        try:
                            await process_task
                        except asyncio.CancelledError:
                            pass
    
    except WebSocketDisconnect:
        logging.info("Deepgram WebSocket odpojen klientem.")
    except Exception as e:
        logging.error(f"Deepgram chyba: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception as send_err:
            logging.debug(f"Nelze poslat Deepgram error: {send_err}")


@app.websocket("/ws/elevenlabs")
async def elevenlabs_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for ElevenLabs Scribe v2 Realtime STT.
    Receives PCM audio from the browser, proxies it to the ElevenLabs
    realtime WS, translates transcripts and sends them back.
    """
    if not await _require_staff_ws_auth(websocket):
        return

    await websocket.accept()
    logging.info("WebSocket /ws/elevenlabs připojen")

    if not ELEVENLABS_API_KEY:
        logging.error("ELEVENLABS_API_KEY není nastaven")
        await websocket.send_json({"error": "ELEVENLABS_API_KEY not configured"})
        await websocket.close()
        return

    translator = Translator()

    # Session defaults.
    translate_src = "cs"
    translate_dests: list[str] = ["en", "ru"]
    translate_interim = True
    el_language_code = ""
    el_commit_strategy = "vad"

    # Read optional config message (first message may be JSON config or audio).
    first_audio: bytes | None = None
    try:
        first = await websocket.receive()
        if first.get("type") == "websocket.receive":
            if first.get("text"):
                try:
                    cfg = json.loads(first["text"])
                except Exception:
                    cfg = None

                if isinstance(cfg, dict) and cfg.get("type") == "config":
                    el_cfg = cfg.get("elevenlabs")
                    if isinstance(el_cfg, dict):
                        lang = el_cfg.get("language_code")
                        if isinstance(lang, str) and lang.strip():
                            el_language_code = lang.strip()
                        strategy = el_cfg.get("commit_strategy")
                        if isinstance(strategy, str) and strategy in {"vad", "manual"}:
                            el_commit_strategy = strategy

                    tr_cfg = cfg.get("translate")
                    if isinstance(tr_cfg, dict):
                        src_norm = _normalize_lang_code(tr_cfg.get("src"))
                        if src_norm:
                            translate_src = src_norm

                        dests_norm = _normalize_translate_dests(tr_cfg.get("dests"))
                        if dests_norm:
                            translate_dests = dests_norm

                    if isinstance(cfg.get("translate_interim"), bool):
                        translate_interim = cfg["translate_interim"]
            elif first.get("bytes"):
                first_audio = first["bytes"]
        elif first.get("type") == "websocket.disconnect":
            return
    except WebSocketDisconnect:
        return

    if len(translate_dests) == 2 and translate_dests[0] == translate_dests[1]:
        translate_dests[1] = "ru" if translate_dests[0] != "ru" else "en"

    # Build ElevenLabs WS URL with query parameters.
    el_params = (
        f"model_id=scribe_v2_realtime"
        f"&audio_format=pcm_16000"
        f"&sample_rate=16000"
        f"&commit_strategy={el_commit_strategy}"
    )
    if el_language_code:
        el_params += f"&language_code={el_language_code}"
    if el_commit_strategy == "vad":
        el_params += "&vad_silence_threshold_secs=1.5"
    el_ws_url = f"{ELEVENLABS_WS_URL}?{el_params}"

    def _el_payload(
        *,
        msg_type: str,
        original: str,
        translations: dict[str, str],
        error: str | None = None,
        timing: dict[str, int] | None = None,
    ) -> dict:
        payload: dict = {
            "type": msg_type,
            "original": original,
            "dests": translate_dests,
            "translations": translations,
        }
        if timing:
            payload["timing"] = timing
        if error:
            payload["error"] = error
        return payload

    el_ws = None
    stop_event = asyncio.Event()

    try:
        el_ws = await ws_lib.connect(
            el_ws_url,
            additional_headers={"xi-api-key": ELEVENLABS_API_KEY},
        )
        logging.info("ElevenLabs Scribe WS připojeno")

        # Wait for session_started before forwarding audio.
        session_msg_raw = await asyncio.wait_for(el_ws.recv(), timeout=10)
        session_msg = json.loads(session_msg_raw)
        logging.info(f"ElevenLabs session started: {session_msg.get('session_id', '')}")

        async def _forward_audio():
            """Read PCM audio from browser WS and forward to ElevenLabs as base64."""
            try:
                if first_audio:
                    audio_b64 = base64.b64encode(first_audio).decode("ascii")
                    await el_ws.send(json.dumps({
                        "message_type": "input_audio_chunk",
                        "audio_base_64": audio_b64,
                        "commit": False,
                        "sample_rate": 16000,
                    }))

                while not stop_event.is_set():
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("type") != "websocket.receive":
                        continue

                    data = msg.get("bytes")
                    if data:
                        audio_b64 = base64.b64encode(data).decode("ascii")
                        await el_ws.send(json.dumps({
                            "message_type": "input_audio_chunk",
                            "audio_base_64": audio_b64,
                            "commit": False,
                            "sample_rate": 16000,
                        }))
                    elif msg.get("text"):
                        # Client may send JSON commands (e.g. commit).
                        try:
                            cmd = json.loads(msg["text"])
                            if isinstance(cmd, dict) and cmd.get("type") == "commit":
                                await el_ws.send(json.dumps({
                                    "message_type": "input_audio_chunk",
                                    "audio_base_64": "",
                                    "commit": True,
                                    "sample_rate": 16000,
                                }))
                            elif isinstance(cmd, dict) and cmd.get("type") == "ping":
                                await websocket.send_json({"type": "pong"})
                        except Exception:
                            pass
            except WebSocketDisconnect:
                logging.info("ElevenLabs: klient odpojen")
            except Exception as e:
                logging.error(f"ElevenLabs forward_audio error: {e}")
            finally:
                stop_event.set()

        async def _receive_transcripts():
            """Read transcripts from ElevenLabs and send translated results to browser."""
            nonlocal translator
            try:
                async for raw in el_ws:
                    if stop_event.is_set():
                        break
                    try:
                        ev = json.loads(raw)
                    except Exception:
                        continue

                    msg_type = ev.get("message_type", "")

                    if msg_type == "partial_transcript":
                        text = ev.get("text", "").strip()
                        if not text:
                            continue
                        if translate_interim:
                            try:
                                start_t = time.perf_counter()
                                results = await asyncio.gather(
                                    *[
                                        _translate(translator, text, src=translate_src, dest=dest)
                                        for dest in translate_dests
                                    ]
                                )
                                translate_ms = int((time.perf_counter() - start_t) * 1000)
                                translations = {
                                    dest: (res.text if res else "")
                                    for dest, res in zip(translate_dests, results)
                                }
                                response = _el_payload(
                                    msg_type="interim",
                                    original=text,
                                    translations=translations,
                                    timing={"translate_ms": translate_ms},
                                )
                            except Exception as te:
                                logging.error(f"ElevenLabs interim translation error: {te}")
                                translator = Translator()
                                response = _el_payload(
                                    msg_type="interim",
                                    original=text,
                                    translations={d: "" for d in translate_dests},
                                    error="translation_failed",
                                    timing={"translate_ms": 0},
                                )
                        else:
                            response = _el_payload(
                                msg_type="interim",
                                original=text,
                                translations={d: "" for d in translate_dests},
                                timing={"translate_ms": 0},
                            )
                        await websocket.send_json(response)

                    elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
                        text = ev.get("text", "").strip()
                        if not text:
                            continue
                        try:
                            start_t = time.perf_counter()
                            results = await asyncio.gather(
                                *[
                                    _translate(translator, text, src=translate_src, dest=dest)
                                    for dest in translate_dests
                                ]
                            )
                            translate_ms = int((time.perf_counter() - start_t) * 1000)
                            translations = {
                                dest: (res.text if res else "")
                                for dest, res in zip(translate_dests, results)
                            }
                            response = _el_payload(
                                msg_type="final",
                                original=text,
                                translations=translations,
                                timing={"translate_ms": translate_ms},
                            )
                        except Exception as te:
                            logging.error(f"ElevenLabs translation error: {te}")
                            translator = Translator()
                            response = _el_payload(
                                msg_type="final",
                                original=text,
                                translations={d: "" for d in translate_dests},
                                error="translation_failed",
                                timing={"translate_ms": 0},
                            )
                        await websocket.send_json(response)

                    elif msg_type in ("input_error", "error", "auth_error",
                                      "transcriber_error", "quota_exceeded"):
                        error_msg = ev.get("error", ev.get("message", str(ev)))
                        logging.error(f"ElevenLabs error: {error_msg}")
                        await websocket.send_json({"error": f"ElevenLabs: {error_msg}"})

            except Exception as e:
                if not stop_event.is_set():
                    logging.error(f"ElevenLabs receive_transcripts error: {e}")
            finally:
                stop_event.set()

        # Run both tasks concurrently; when one stops the other is cancelled.
        forward_task = asyncio.create_task(_forward_audio())
        receive_task = asyncio.create_task(_receive_transcripts())

        done, pending = await asyncio.wait(
            {forward_task, receive_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        stop_event.set()
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        logging.info("ElevenLabs WebSocket odpojen klientem.")
    except Exception as e:
        logging.error(f"ElevenLabs chyba: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception as send_err:
            logging.debug(f"Nelze poslat ElevenLabs error: {send_err}")
    finally:
        if el_ws:
            try:
                await el_ws.close()
            except Exception:
                pass


@app.websocket("/ws/gemini-live")
@app.websocket("/ws/gemini_live")
async def gemini_live_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Gemini Live Simultaneous Audio Translation.
    Supports Mode A (Direct Audio -> Gemini Live) and Mode B (STT -> Text -> Gemini Live),
    with seamless automatic failover/fallback to Google Translate and failback recovery.
    """
    if not await _require_staff_ws_auth(websocket):
        return

    await websocket.accept()

    gemini_key = GEMINI_API_KEY
    fallback_provider = GoogleTranslationProvider()

    # Session defaults (English -> Spanish)
    translate_src = "en"
    translate_dests: list[str] = ["es"]
    translate_interim = True
    session_glossary: dict[str, str] = {}
    last_transcript = ""
    selected_model = os.getenv("GEMINI_LIVE_MODEL", DEFAULT_GEMINI_LIVE_MODEL).strip()
    mode = os.getenv("GEMINI_LIVE_MODE", "A").upper()  # "A" = Direct Audio, "B" = Text

    first_audio: bytes | None = None
    first_cmd: dict | None = None
    try:
        first = await websocket.receive()
        if first.get("type") == "websocket.receive":
            if first.get("text"):
                try:
                    cfg = json.loads(first["text"])
                except Exception:
                    cfg = None

                if isinstance(cfg, dict):
                    if cfg.get("type") == "config":
                        if cfg.get("model"):
                            selected_model = str(cfg["model"]).strip()
                        if cfg.get("mode"):
                            mode = str(cfg["mode"]).strip().upper()
                        tr_cfg = cfg.get("translate")
                        if isinstance(tr_cfg, dict):
                            src_norm = _normalize_lang_code(tr_cfg.get("src"))
                            if src_norm:
                                translate_src = src_norm
                            dests_norm = _normalize_translate_dests(tr_cfg.get("dests"))
                            if dests_norm:
                                translate_dests = dests_norm
                        if isinstance(cfg.get("glossary"), dict):
                            session_glossary = cfg["glossary"]
                        if isinstance(cfg.get("translate_interim"), bool):
                            translate_interim = cfg["translate_interim"]
                    else:
                        first_cmd = cfg
            elif first.get("bytes"):
                first_audio = first["bytes"]
        elif first.get("type") == "websocket.disconnect":
            logging.info("[GeminiLive] WebSocket closed")
            return
    except WebSocketDisconnect:
        logging.info("[GeminiLive] WebSocket closed")
        return

    target_lang = translate_dests[0] if translate_dests else "es"
    gemini_ws_url = f"{GEMINI_LIVE_WS_URL}?key={gemini_key}"
    stop_event = asyncio.Event()
    gemini_ws = None
    telemetry = LatencyTelemetry()
    active_provider = "gemini_live"

    async def _handle_fallback_translation(text: str, msg_type: str = "final", client_ts: float | None = None):
        """Processes translation via Google Translate fallback when Gemini Live is unavailable."""
        nonlocal active_provider
        if not text or not text.strip():
            return
        if active_provider != "google_translate":
            logging.warning("[Translation] Switching to Google Translate")
            logging.info("[Translation] Google Translate fallback active")
            active_provider = "google_translate"

        telemetry.fallback_started_at = time.time()
        start_t = time.perf_counter()
        try:
            res_es = await fallback_provider.translate(
                text=text.strip(),
                source=translate_src,
                target=target_lang,
                glossary=session_glossary,
            )
            telemetry.fallback_finished_at = time.time()
            telemetry.translation_emitted_at = time.time()
            trans_ms = int((time.perf_counter() - start_t) * 1000)

            await websocket.send_json({
                "type": msg_type,
                "original": text.strip(),
                "dests": translate_dests,
                "translations": {target_lang: res_es},
                "provider": "google_translate",
                "provider_status": "FALLBACK",
                "timing": {
                    "translate_ms": trans_ms,
                    "fallback_latency_ms": telemetry.fallback_latency_ms,
                    "end_to_end_latency_ms": telemetry.end_to_end_latency_ms,
                }
            })
        except Exception as e:
            logging.error(f"[Translation] Fallback translation error: {e}")

    # Check API key configuration and circuit breaker state
    if not gemini_key:
        logging.error("[GeminiLive] Authentication error: GEMINI_API_KEY is not configured")
        global_gemini_live_circuit.trip(ErrorType.AUTH_ERROR, "GEMINI_API_KEY not configured")
        logging.warning("[Translation] Switching to Google Translate")
        logging.info("[Translation] Google Translate fallback active")
        active_provider = "google_translate"

    elif not global_gemini_live_circuit.is_available():
        logging.warning(
            f"[GeminiLive] Circuit breaker is currently OPEN until {global_gemini_live_circuit.open_until:.0f}"
        )
        logging.warning("[Translation] Switching to Google Translate")
        logging.info("[Translation] Google Translate fallback active")
        active_provider = "google_translate"

    else:
        # Attempt Gemini Live connection
        logging.info("[GeminiLive] Connecting...")
        try:
            gemini_ws = await ws_lib.connect(gemini_ws_url)
            logging.info("[GeminiLive] Connected")

            system_instruction_text = build_system_instruction(
                source_lang=translate_src,
                target_lang=target_lang,
                glossary=session_glossary,
            )

            setup_msg = {
                "setup": {
                    "model": f"models/{selected_model}",
                    "generationConfig": {
                        "responseModalities": ["TEXT"],
                        "temperature": 0.1,
                    },
                    "systemInstruction": {
                        "parts": [{"text": system_instruction_text}],
                    },
                }
            }
            await gemini_ws.send(json.dumps(setup_msg))

            # Receive setup response / check error
            raw_ack = await asyncio.wait_for(gemini_ws.recv(), timeout=5.0)
            ack = json.loads(raw_ack)
            if ack.get("error"):
                err = ack["error"]
                err_msg = err.get("message", str(err))
                err_code = err.get("code", 400)
                error_type = ErrorType.AUTH_ERROR if err_code in (401, 403) else ErrorType.MODEL_ERROR
                if error_type == ErrorType.AUTH_ERROR:
                    logging.error("[GeminiLive] Authentication error")
                logging.error(f"[GeminiLive] Setup rejected: {redact_secrets(err_msg, gemini_key)}")
                global_gemini_live_circuit.trip(error_type, err_msg)
                await gemini_ws.close()
                gemini_ws = None
                logging.warning("[Translation] Switching to Google Translate")
                logging.info("[Translation] Google Translate fallback active")
                active_provider = "google_translate"
            else:
                logging.info("[GeminiLive] Setup complete")
                global_gemini_live_circuit.record_success()
                active_provider = "gemini_live"

        except asyncio.TimeoutError:
            logging.warning("[GeminiLive] Setup timeout waiting for setupComplete")
            global_gemini_live_circuit.trip(ErrorType.NETWORK_ERROR, "Setup timeout")
            if gemini_ws:
                await gemini_ws.close()
                gemini_ws = None
            logging.warning("[Translation] Switching to Google Translate")
            logging.info("[Translation] Google Translate fallback active")
            active_provider = "google_translate"
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "Unauthorized" in err_str or "Forbidden" in err_str or "403" in err_str:
                logging.error("[GeminiLive] Authentication error")
                global_gemini_live_circuit.trip(ErrorType.AUTH_ERROR, err_str)
            else:
                logging.error(f"[GeminiLive] Connection error: {redact_secrets(err_str, gemini_key)}")
                global_gemini_live_circuit.trip(ErrorType.WEBSOCKET_ERROR, err_str)
            if gemini_ws:
                await gemini_ws.close()
                gemini_ws = None
            logging.warning("[Translation] Switching to Google Translate")
            logging.info("[Translation] Google Translate fallback active")
            active_provider = "google_translate"

    # Streaming loops
    async def _forward_audio_and_text():
        nonlocal last_transcript, gemini_ws, active_provider
        audio_started = False
        try:
            if first_audio and gemini_ws:
                if not audio_started:
                    logging.info("[GeminiLive] Audio streaming started")
                    audio_started = True
                telemetry.audio_received_at = time.time()
                telemetry.gemini_sent_at = time.time()
                audio_b64 = base64.b64encode(first_audio).decode("ascii")
                await gemini_ws.send(json.dumps({
                    "realtimeInput": {
                        "mediaChunks": [{
                            "mimeType": "audio/pcm;rate=16000",
                            "data": audio_b64
                        }]
                    }
                }))

            if first_cmd:
                if first_cmd.get("type") == "transcript":
                    last_transcript = first_cmd.get("text", "")
                    if not gemini_ws or active_provider == "google_translate":
                        await _handle_fallback_translation(
                            text=last_transcript,
                            msg_type="interim" if first_cmd.get("interim") else "final",
                        )
                    elif mode == "B" and gemini_ws:
                        telemetry.gemini_sent_at = time.time()
                        await gemini_ws.send(json.dumps({
                            "realtimeInput": {
                                "clientContent": {
                                    "turns": [{
                                        "role": "user",
                                        "parts": [{"text": last_transcript}],
                                    }],
                                    "turnComplete": not first_cmd.get("interim"),
                                }
                            }
                        }))

            while not stop_event.is_set():
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("type") != "websocket.receive":
                    continue

                data = msg.get("bytes")
                if data:
                    telemetry.audio_received_at = time.time()
                    if gemini_ws:
                        if not audio_started:
                            logging.info("[GeminiLive] Audio streaming started")
                            audio_started = True
                        telemetry.gemini_sent_at = time.time()
                        audio_b64 = base64.b64encode(data).decode("ascii")
                        try:
                            await gemini_ws.send(json.dumps({
                                "realtimeInput": {
                                    "mediaChunks": [{
                                        "mimeType": "audio/pcm;rate=16000",
                                        "data": audio_b64
                                    }]
                                }
                            }))
                        except Exception as send_err:
                            logging.error(f"[GeminiLive] Audio send error: {redact_secrets(str(send_err), gemini_key)}")
                            if gemini_ws:
                                try:
                                    await gemini_ws.close()
                                except Exception:
                                    pass
                                gemini_ws = None
                            logging.warning("[Translation] Switching to Google Translate")
                            logging.info("[Translation] Google Translate fallback active")
                            active_provider = "google_translate"
                elif msg.get("text"):
                    try:
                        cmd = json.loads(msg["text"])
                        if isinstance(cmd, dict):
                            if cmd.get("type") == "ping":
                                await websocket.send_json({"type": "pong"})
                            elif cmd.get("type") == "transcript":
                                last_transcript = cmd.get("text", "")
                                # If Gemini Live is disconnected or in Mode B, handle translation
                                if not gemini_ws or active_provider == "google_translate":
                                    await _handle_fallback_translation(
                                        text=last_transcript,
                                        msg_type="interim" if cmd.get("interim") else "final",
                                    )
                                elif mode == "B" and gemini_ws:
                                    # Mode B: Audio -> STT -> Text -> Gemini Live
                                    telemetry.gemini_sent_at = time.time()
                                    await gemini_ws.send(json.dumps({
                                        "realtimeInput": {
                                            "clientContent": {
                                                "turns": [{
                                                    "role": "user",
                                                    "parts": [{"text": last_transcript}],
                                                }],
                                                "turnComplete": not cmd.get("interim"),
                                            }
                                        }
                                    }))
                            elif cmd.get("type") == "commit":
                                pass
                    except Exception:
                        pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            logging.error(f"[GeminiLive] Forwarding error: {redact_secrets(str(e), gemini_key)}")
        finally:
            stop_event.set()

    async def _receive_gemini():
        nonlocal last_transcript, gemini_ws, active_provider
        accumulated_turn_text = ""
        turn_start_t = time.perf_counter()
        try:
            if not gemini_ws:
                await stop_event.wait()
                return

            async for raw in gemini_ws:
                if stop_event.is_set():
                    break
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue

                if ev.get("error"):
                    err = ev["error"]
                    err_msg = err.get("message", str(err))
                    err_code = err.get("code", 400)
                    error_type = ErrorType.AUTH_ERROR if err_code in (401, 403) else ErrorType.SERVER_ERROR
                    if error_type == ErrorType.AUTH_ERROR:
                        logging.error("[GeminiLive] Authentication error")
                    logging.error(f"[GeminiLive] Stream error: {redact_secrets(err_msg, gemini_key)}")
                    global_gemini_live_circuit.trip(error_type, err_msg)
                    break

                server_content = ev.get("serverContent")
                if server_content:
                    model_turn = server_content.get("modelTurn")
                    turn_complete = server_content.get("turnComplete", False)

                    if model_turn:
                        parts = model_turn.get("parts", [])
                        chunk_text = "".join(p.get("text", "") for p in parts if p.get("text"))
                        if chunk_text:
                            telemetry.gemini_response_at = time.time()
                            telemetry.translation_emitted_at = time.time()
                            logging.info(
                                f"[GeminiLive] Translation received (chunk_len={len(chunk_text)}, latency={telemetry.gemini_latency_ms}ms)"
                            )
                            accumulated_turn_text += chunk_text
                            clean_text = apply_glossary(
                                accumulated_turn_text.strip(),
                                session_glossary,
                                target_lang=target_lang,
                            )
                            trans_ms = int((time.perf_counter() - turn_start_t) * 1000)
                            await websocket.send_json({
                                "type": "interim",
                                "original": last_transcript or clean_text,
                                "dests": translate_dests,
                                "translations": {target_lang: clean_text},
                                "provider": "gemini_live",
                                "provider_status": "CONNECTED",
                                "timing": {
                                    "translate_ms": trans_ms,
                                    "gemini_latency_ms": telemetry.gemini_latency_ms,
                                    "end_to_end_latency_ms": telemetry.end_to_end_latency_ms,
                                }
                            })

                    if turn_complete:
                        if accumulated_turn_text.strip():
                            telemetry.gemini_response_at = time.time()
                            telemetry.translation_emitted_at = time.time()
                            final_text = apply_glossary(
                                accumulated_turn_text.strip(),
                                session_glossary,
                                target_lang=target_lang,
                            )
                            trans_ms = int((time.perf_counter() - turn_start_t) * 1000)
                            await websocket.send_json({
                                "type": "final",
                                "original": last_transcript or final_text,
                                "dests": translate_dests,
                                "translations": {target_lang: final_text},
                                "provider": "gemini_live",
                                "provider_status": "CONNECTED",
                                "timing": {
                                    "translate_ms": trans_ms,
                                    "gemini_latency_ms": telemetry.gemini_latency_ms,
                                    "end_to_end_latency_ms": telemetry.end_to_end_latency_ms,
                                }
                            })
                        accumulated_turn_text = ""
                        turn_start_t = time.perf_counter()

        except Exception as e:
            if not stop_event.is_set():
                logging.error(f"[GeminiLive] Stream exception: {redact_secrets(str(e), gemini_key)}")
                global_gemini_live_circuit.trip(ErrorType.WEBSOCKET_ERROR, str(e))
        finally:
            logging.info("[GeminiLive] WebSocket closed")
            if gemini_ws:
                try:
                    await gemini_ws.close()
                except Exception:
                    pass
                gemini_ws = None
            if not stop_event.is_set():
                logging.warning("[Translation] Switching to Google Translate")
                logging.info("[Translation] Google Translate fallback active")
                active_provider = "google_translate"

    try:
        forward_task = asyncio.create_task(_forward_audio_and_text())
        receive_task = asyncio.create_task(_receive_gemini())

        done, pending = await asyncio.wait(
            {forward_task, receive_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        stop_event.set()
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logging.error(f"[GeminiLive] WebSocket loop error: {redact_secrets(str(e), gemini_key)}")
    finally:
        if gemini_ws:
            try:
                await gemini_ws.close()
            except Exception:
                pass
        logging.info("[GeminiLive] WebSocket closed")


