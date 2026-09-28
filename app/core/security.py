import os
import base64
import hashlib
import hmac
import json
import time
import secrets
from urllib.parse import urlparse
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

from app.core import config

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



def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _sign(payload_b64: str) -> str:
    if not config.AUTH_SECRET:
        return ""
    mac = hmac.new(
        config.AUTH_SECRET.encode("utf-8"),
        payload_b64.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return _b64url_encode(mac)


def create_auth_token() -> str:
    now = int(time.time())
    payload = {"iat": now, "exp": now + config.AUTH_TOKEN_TTL_SECONDS}
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    sig_b64 = _sign(payload_b64)
    return f"{payload_b64}.{sig_b64}"


def verify_auth_token(token: str | None) -> bool:
    if not token or not config.AUTH_SECRET:
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


