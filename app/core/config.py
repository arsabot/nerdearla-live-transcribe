import os
import logging
from dotenv import load_dotenv

load_dotenv()


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

# --- Latency diagnostics ---
# Set DEBUG_LATENCY=true to emit per-event timing logs and a session summary.
DEBUG_LATENCY: bool = os.getenv("DEBUG_LATENCY", "false").lower() in ("1", "true", "yes")
LATENCY_LOG_FILE: str = os.getenv("LATENCY_LOG_FILE", "logs/latency.log")

# --- Simple in-memory rate limiter for /login ---
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_MAX_ATTEMPTS = 10
_LOGIN_WINDOW_SECONDS = 60.0


