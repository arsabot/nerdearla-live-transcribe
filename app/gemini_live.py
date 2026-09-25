"""
Gemini Live Real-time Translation Layer & Provider.
Implements bidirectional WebSocket streaming via BidiGenerateContent
with automatic fallback to Google Translate, granular circuit breakers,
latency telemetry, and full support for multilingual technical interpretations.
"""

from __future__ import annotations

import asyncio
import base64
import enum
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Tuple

try:
    import websockets.asyncio.client as ws_lib
except ImportError:
    try:
        import websockets as ws_lib
    except ImportError:
        ws_lib = None  # type: ignore

from app.translation_provider import (
    GoogleTranslationProvider,
    TranslationProvider,
    apply_glossary,
)


# Language names for professional system interpretation prompt
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "pt": "Portuguese",
    "it": "Italian",
    "fr": "French",
    "de": "German",
    "ru": "Russian",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "ja": "Japanese",
    "auto": "the spoken language",
}


def get_language_name(code: str) -> str:
    """Resolve ISO language code to human-readable English name."""
    c = (code or "en").strip().lower()
    return LANGUAGE_NAMES.get(c, LANGUAGE_NAMES.get(c.split("-")[0], c.upper()))


class ConnectionState(str, enum.Enum):
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    ERROR = "ERROR"


class ErrorType(str, enum.Enum):
    NONE = "NONE"
    AUTH_ERROR = "AUTH_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    MODEL_ERROR = "MODEL_ERROR"
    RATE_LIMIT = "RATE_LIMIT"
    SERVER_ERROR = "SERVER_ERROR"
    WEBSOCKET_ERROR = "WEBSOCKET_ERROR"


class LatencyTelemetry:
    """Tracks latency metrics for audio/text through Gemini Live and Fallback."""

    def __init__(self) -> None:
        self.audio_received_at: float = 0.0
        self.gemini_sent_at: float = 0.0
        self.gemini_response_at: float = 0.0
        self.translation_emitted_at: float = 0.0
        self.fallback_started_at: float = 0.0
        self.fallback_finished_at: float = 0.0

    @property
    def gemini_latency_ms(self) -> float:
        if self.gemini_sent_at > 0 and self.gemini_response_at >= self.gemini_sent_at:
            return round((self.gemini_response_at - self.gemini_sent_at) * 1000, 2)
        return 0.0

    @property
    def fallback_latency_ms(self) -> float:
        if self.fallback_started_at > 0 and self.fallback_finished_at >= self.fallback_started_at:
            return round((self.fallback_finished_at - self.fallback_started_at) * 1000, 2)
        return 0.0

    @property
    def end_to_end_latency_ms(self) -> float:
        start = self.audio_received_at or self.gemini_sent_at or self.fallback_started_at
        end = self.translation_emitted_at or self.gemini_response_at or self.fallback_finished_at
        if start > 0 and end >= start:
            return round((end - start) * 1000, 2)
        return 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "gemini_latency_ms": self.gemini_latency_ms,
            "fallback_latency_ms": self.fallback_latency_ms,
            "end_to_end_latency_ms": self.end_to_end_latency_ms,
        }


def redact_secrets(text: str, secret: Optional[str] = None) -> str:
    """Redacts API keys and secrets from string representations and logs."""
    if not text:
        return text
    res = text
    if secret and len(secret) > 4:
        res = res.replace(secret, "[REDACTED_API_KEY]")
    # Redact common Gemini key formats AIzaSy... or AQ.Ab8...
    res = re.sub(r"key=([A-Za-z0-9_\-\.]{8,})", "key=[REDACTED_KEY]", res)
    res = re.sub(r"(AIzaSy[A-Za-z0-9_\-]{30,})", "[REDACTED_AIZA_KEY]", res)
    res = re.sub(r"(AQ\.[A-Za-z0-9_\-]{30,})", "[REDACTED_AQ_KEY]", res)
    return res


class GeminiLiveCircuitBreaker:
    """
    Differentiated Circuit Breaker for Gemini Live API:
    - AUTH_ERROR (401 / Invalid Key): Opens immediately with 120s cooldown.
    - RATE_LIMIT (429): 60s cooldown.
    - NETWORK_ERROR / WEBSOCKET_ERROR: Exponential backoff (2s -> 4s -> 8s -> 30s).
    - SERVER_ERROR (5xx): 30s cooldown.
    """

    def __init__(self) -> None:
        self.is_open: bool = False
        self.open_until: float = 0.0
        self.last_error_type: ErrorType = ErrorType.NONE
        self.last_error_reason: str = ""
        self.failure_count: int = 0
        self.backoff_seconds: float = 2.0

    def is_available(self) -> bool:
        if not self.is_open:
            return True
        if time.time() >= self.open_until:
            # Half-open: allow probe
            return True
        return False

    def trip(self, error_type: ErrorType, reason: str, cooldown: Optional[float] = None) -> None:
        self.is_open = True
        self.last_error_type = error_type
        self.last_error_reason = redact_secrets(reason)
        self.failure_count += 1

        if cooldown is not None:
            delay = cooldown
        elif error_type == ErrorType.AUTH_ERROR:
            delay = 120.0
        elif error_type == ErrorType.RATE_LIMIT:
            delay = 60.0
        elif error_type == ErrorType.SERVER_ERROR:
            delay = 30.0
        else:
            # Exponential backoff for network/websocket transient issues
            delay = min(self.backoff_seconds, 30.0)
            self.backoff_seconds = min(self.backoff_seconds * 2.0, 30.0)

        self.open_until = time.time() + delay
        logging.warning(
            f"[GeminiLive] Circuit breaker OPEN (type={error_type.value}, duration={delay:.1f}s, reason={self.last_error_reason})"
        )

    def record_success(self) -> None:
        if self.is_open:
            logging.info("[GeminiLive] Circuit breaker CLOSED (reconnection successful)")
        self.is_open = False
        self.open_until = 0.0
        self.last_error_type = ErrorType.NONE
        self.last_error_reason = ""
        self.failure_count = 0
        self.backoff_seconds = 2.0


# Global circuit breaker instance shared across providers
global_gemini_live_circuit = GeminiLiveCircuitBreaker()


def build_system_instruction(
    source_lang: str = "en",
    target_lang: str = "es",
    glossary: Optional[Dict[str, str]] = None,
) -> str:
    """Builds strict simultaneous translation system prompt for Gemini Live."""
    src_name = get_language_name(source_lang)
    tgt_name = get_language_name(target_lang)

    glossary_lines = ""
    if glossary:
        substitutions = [f"'{k}' -> '{v}'" for k, v in glossary.items() if k and v]
        if substitutions:
            glossary_lines = (
                f"\nPreserve these technical terminology substitutions strictly: {', '.join(substitutions)}."
            )

    return (
        f"You are a professional realtime simultaneous interpreter for technical conferences.\n"
        f"Translate only the speaker's meaning from {src_name} to {tgt_name} in continuous real-time.\n"
        f"Strict Rules:\n"
        f"- Output ONLY the direct translated plain text in {tgt_name}.\n"
        f"- Do not explain.\n"
        f"- Do not summarize.\n"
        f"- Do not answer questions.\n"
        f"- Do not add context or conversational filler.\n"
        f"- Preserve names, numbers, technical terminology and acronyms.{glossary_lines}\n"
        f"- Always detect and preserve conference brand names: 'Nerdearla', 'Vibeathon', 'Kubernetes'. When you hear 'nerdearla' (or phonetic distortions like 'nerd de habla', 'nerd arla', 'merdearla'), always transcribe/preserve it strictly as 'Nerdearla'.\n"
        f"- Return only the translated text."
    )


class GeminiLiveTranslator:
    """
    Direct WebSocket client for Gemini Live BidiGenerateContent.
    Handles session setup, PCM audio streaming, text streaming,
    turnComplete tracking, error categorization, and latency telemetry.
    """

    DEFAULT_WS_URL = (
        "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        ws_url: Optional[str] = None,
        circuit_breaker: Optional[GeminiLiveCircuitBreaker] = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.getenv("GEMINI_LIVE_MODEL", "gemini-3.5-live-translate-preview")).strip()
        self.ws_url = (ws_url or os.getenv("GEMINI_LIVE_WS_URL", self.DEFAULT_WS_URL)).strip()
        self.circuit_breaker = circuit_breaker or global_gemini_live_circuit

        self.state: ConnectionState = ConnectionState.DISCONNECTED
        self.ws: Any = None
        self.source_lang: str = "en"
        self.target_lang: str = "es"
        self.glossary: Dict[str, str] = {}
        self.telemetry = LatencyTelemetry()
        self._audio_started: bool = False
        self._setup_complete: bool = False
        self._close_event = asyncio.Event()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    async def connect(
        self,
        source_lang: str = "en",
        target_lang: str = "es",
        glossary: Optional[Dict[str, str]] = None,
    ) -> bool:
        """Connects to Gemini Live BidiGenerateContent WebSocket and performs setup."""
        if not self.api_key:
            logging.error("[GeminiLive] Authentication error: GEMINI_API_KEY is not configured")
            self.circuit_breaker.trip(ErrorType.AUTH_ERROR, "GEMINI_API_KEY not configured")
            self.state = ConnectionState.ERROR
            return False

        if not self.circuit_breaker.is_available():
            logging.info(
                f"[GeminiLive] Circuit breaker is currently OPEN until {self.circuit_breaker.open_until:.0f}"
            )
            self.state = ConnectionState.ERROR
            return False

        if ws_lib is None:
            logging.error("[GeminiLive] websockets library is not available")
            self.state = ConnectionState.ERROR
            return False

        self.source_lang = source_lang or "en"
        self.target_lang = target_lang or "es"
        self.glossary = glossary or {}
        self._audio_started = False
        self._setup_complete = False
        self._close_event.clear()

        endpoint_url = f"{self.ws_url}?key={self.api_key}"
        logging.info("[GeminiLive] Connecting...")
        self.state = ConnectionState.CONNECTING

        try:
            self.ws = await ws_lib.connect(endpoint_url)
            logging.info("[GeminiLive] Connected")
            self.state = ConnectionState.CONNECTED

            # Construct setup message following official Gemini Live specification
            system_prompt = build_system_instruction(
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                glossary=self.glossary,
            )

            setup_msg = {
                "setup": {
                    "model": f"models/{self.model}",
                    "generationConfig": {
                        "responseModalities": ["TEXT"],
                        "temperature": 0.1,
                    },
                    "systemInstruction": {
                        "parts": [{"text": system_prompt}],
                    },
                }
            }

            await self.ws.send(json.dumps(setup_msg))

            # Wait for setup acknowledgment / first response with timeout
            raw_ack = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            ack = json.loads(raw_ack)

            if ack.get("error"):
                err = ack["error"]
                err_msg = err.get("message", str(err))
                err_code = err.get("code", 400)
                error_type = ErrorType.AUTH_ERROR if err_code in (401, 403) else ErrorType.MODEL_ERROR
                logging.error(f"[GeminiLive] Setup rejected: {redact_secrets(err_msg, self.api_key)}")
                if error_type == ErrorType.AUTH_ERROR:
                    logging.error("[GeminiLive] Authentication error")
                self.circuit_breaker.trip(error_type, err_msg)
                self.state = ConnectionState.ERROR
                await self.close()
                return False

            self._setup_complete = True
            logging.info("[GeminiLive] Setup complete")
            self.circuit_breaker.record_success()
            return True

        except asyncio.TimeoutError:
            logging.warning("[GeminiLive] Setup timeout waiting for setupComplete")
            self.circuit_breaker.trip(ErrorType.NETWORK_ERROR, "Setup timeout")
            self.state = ConnectionState.ERROR
            await self.close()
            return False
        except Exception as e:
            err_str = str(e)
            if "401" in err_str or "Unauthorized" in err_str or "Forbidden" in err_str or "403" in err_str:
                logging.error("[GeminiLive] Authentication error")
                self.circuit_breaker.trip(ErrorType.AUTH_ERROR, err_str)
            else:
                logging.error(f"[GeminiLive] Connection error: {redact_secrets(err_str, self.api_key)}")
                self.circuit_breaker.trip(ErrorType.WEBSOCKET_ERROR, err_str)
            self.state = ConnectionState.ERROR
            await self.close()
            return False

    async def send_audio(self, pcm_bytes: bytes, client_ts: Optional[float] = None) -> bool:
        """Sends PCM 16kHz 16-bit mono audio chunk to Gemini Live."""
        if not self.ws or self.state != ConnectionState.CONNECTED:
            return False

        if not self._audio_started:
            logging.info("[GeminiLive] Audio streaming started")
            self._audio_started = True

        self.telemetry.audio_received_at = client_ts or time.time()
        self.telemetry.gemini_sent_at = time.time()

        try:
            audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
            msg = {
                "realtimeInput": {
                    "mediaChunks": [
                        {
                            "mimeType": "audio/pcm;rate=16000",
                            "data": audio_b64,
                        }
                    ]
                }
            }
            await self.ws.send(json.dumps(msg))
            return True
        except Exception as e:
            logging.error(f"[GeminiLive] Audio send error: {redact_secrets(str(e), self.api_key)}")
            self.state = ConnectionState.ERROR
            return False

    async def send_text(self, text: str, client_ts: Optional[float] = None) -> bool:
        """Sends text turn to Gemini Live for Mode B (STT -> Text -> Gemini Live)."""
        if not self.ws or self.state != ConnectionState.CONNECTED or not text.strip():
            return False

        self.telemetry.gemini_sent_at = client_ts or time.time()

        try:
            msg = {
                "realtimeInput": {
                    "clientContent": {
                        "turns": [
                            {
                                "role": "user",
                                "parts": [{"text": text.strip()}],
                            }
                        ],
                        "turnComplete": True,
                    }
                }
            }
            await self.ws.send(json.dumps(msg))
            return True
        except Exception as e:
            logging.error(f"[GeminiLive] Text send error: {redact_secrets(str(e), self.api_key)}")
            self.state = ConnectionState.ERROR
            return False

    async def stream_responses(self) -> AsyncGenerator[Tuple[str, bool, LatencyTelemetry], None]:
        """
        Yields (translated_text_chunk, is_turn_complete, telemetry) from Gemini Live.
        """
        if not self.ws:
            return

        accumulated_turn_text = ""
        try:
            async for raw in self.ws:
                if self._close_event.is_set():
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
                    logging.error(f"[GeminiLive] Stream error: {redact_secrets(err_msg, self.api_key)}")
                    self.circuit_breaker.trip(error_type, err_msg)
                    break

                server_content = ev.get("serverContent")
                if server_content:
                    model_turn = server_content.get("modelTurn")
                    turn_complete = server_content.get("turnComplete", False)

                    if model_turn:
                        parts = model_turn.get("parts", [])
                        chunk = "".join(p.get("text", "") for p in parts if p.get("text"))
                        if chunk:
                            self.telemetry.gemini_response_at = time.time()
                            self.telemetry.translation_emitted_at = time.time()
                            accumulated_turn_text += chunk
                            logging.info(
                                f"[GeminiLive] Translation received (chunk_len={len(chunk)}, latency={self.telemetry.gemini_latency_ms}ms)"
                            )
                            clean_text = apply_glossary(
                                accumulated_turn_text.strip(),
                                self.glossary,
                                target_lang=self.target_lang,
                            )
                            yield clean_text, False, self.telemetry

                    if turn_complete:
                        if accumulated_turn_text.strip():
                            self.telemetry.gemini_response_at = time.time()
                            self.telemetry.translation_emitted_at = time.time()
                            final_text = apply_glossary(
                                accumulated_turn_text.strip(),
                                self.glossary,
                                target_lang=self.target_lang,
                            )
                            yield final_text, True, self.telemetry
                        accumulated_turn_text = ""

        except Exception as e:
            if not self._close_event.is_set():
                logging.error(f"[GeminiLive] Stream exception: {redact_secrets(str(e), self.api_key)}")
                self.circuit_breaker.trip(ErrorType.WEBSOCKET_ERROR, str(e))
        finally:
            logging.info("[GeminiLive] WebSocket closed")
            self.state = ConnectionState.DISCONNECTED
            await self.close()

    async def close(self) -> None:
        """Gracefully closes the Gemini Live WebSocket session."""
        self._close_event.set()
        self.state = ConnectionState.DISCONNECTED
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


class GeminiLiveProvider(TranslationProvider):
    """
    Robust Primary Gemini Live Translation Provider with automatic Google Translate Fallback.
    Handles seamless failover, health probing, periodic reconnection, and failback.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback: Optional[TranslationProvider] = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model = (model or os.getenv("GEMINI_LIVE_MODEL", "gemini-3.5-live-translate-preview")).strip()
        self.fallback = fallback or GoogleTranslationProvider()
        self.circuit_breaker = global_gemini_live_circuit
        self._last_active_provider = "gemini_live"
        self._reconnect_task: Optional[asyncio.Task] = None

    def is_gemini_available(self) -> bool:
        """Returns True only when API Key is set and Circuit Breaker is healthy."""
        return bool(self.api_key) and self.circuit_breaker.is_available()

    @property
    def status_summary(self) -> Dict[str, Any]:
        """Exposes health status for UI badges and health endpoints."""
        if not self.api_key:
            return {
                "available": False,
                "provider": "google_translate",
                "state": "FALLBACK",
                "reason": "GEMINI_API_KEY not configured",
            }
        if self.circuit_breaker.is_open:
            return {
                "available": False,
                "provider": "google_translate",
                "state": "FALLBACK",
                "circuit_open_until": self.circuit_breaker.open_until,
                "reason": self.circuit_breaker.last_error_reason,
            }
        return {
            "available": True,
            "provider": "gemini_live",
            "state": "CONNECTED",
            "model": self.model,
        }

    async def translate(
        self,
        text: str,
        source: str = "auto",
        target: str = "es",
        glossary: Optional[Dict] = None,
    ) -> str:
        """Translates text via Gemini Live REST/Live API, with seamless Google Translate Fallback."""
        if not text or not text.strip():
            return ""

        if source and target and source.lower() == target.lower() and source.lower() != "auto":
            return apply_glossary(text, glossary, target_lang=target)

        # Check circuit availability
        if not self.is_gemini_available():
            if self._last_active_provider != "google_translate":
                logging.warning("[Translation] Switching to Google Translate")
                logging.info("[Translation] Google Translate fallback active")
                self._last_active_provider = "google_translate"
            return await self.fallback.translate(text, source=source, target=target, glossary=glossary)

        # Attempt translation via Gemini Flash / Live Generate
        import httpx

        prompt = (
            f"You are a professional realtime simultaneous conference interpreter.\n"
            f"Translate the following text from {get_language_name(source)} to {get_language_name(target)}.\n"
            f"Return ONLY the plain translated text without explanation, markdown or quotes.\n\n"
            f"Text: {text}"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 600},
        }

        try:
            t0 = time.perf_counter()
            async with httpx.AsyncClient(timeout=3.5) as client:
                resp = await client.post(url, json=payload)

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        res_text = parts[0].get("text", "").strip()
                        res_text = res_text.replace("**", "").replace("*", "").replace('"', '').strip()
                        if res_text:
                            if self._last_active_provider != "gemini_live":
                                logging.info("[Translation] Switching back to Gemini Live")
                                self._last_active_provider = "gemini_live"
                            self.circuit_breaker.record_success()
                            return apply_glossary(res_text, glossary, target_lang=target)

            elif resp.status_code in (401, 403):
                logging.error("[GeminiLive] Authentication error")
                self.circuit_breaker.trip(ErrorType.AUTH_ERROR, f"HTTP {resp.status_code}")
            elif resp.status_code == 429:
                self.circuit_breaker.trip(ErrorType.RATE_LIMIT, "HTTP 429 Rate Limit")
            else:
                self.circuit_breaker.trip(ErrorType.SERVER_ERROR, f"HTTP {resp.status_code}")

        except Exception as e:
            err_str = str(e)
            logging.warning(f"[GeminiLive] Translation error: {redact_secrets(err_str, self.api_key)}")
            self.circuit_breaker.trip(ErrorType.NETWORK_ERROR, err_str)

        # Fallback to Google Translate
        if self._last_active_provider != "google_translate":
            logging.warning("[Translation] Switching to Google Translate")
            logging.info("[Translation] Google Translate fallback active")
            self._last_active_provider = "google_translate"

        return await self.fallback.translate(text, source=source, target=target, glossary=glossary)

    async def translate_batch(
        self,
        text: str,
        source: str = "auto",
        targets: Optional[List[str]] = None,
        glossary: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """Translates text across multiple target languages simultaneously."""
        if not targets:
            targets = ["es"]

        if not self.is_gemini_available():
            if self._last_active_provider != "google_translate":
                logging.warning("[Translation] Switching to Google Translate")
                logging.info("[Translation] Google Translate fallback active")
                self._last_active_provider = "google_translate"
            return await self.fallback.translate_batch(text, source=source, targets=targets, glossary=glossary)

        import httpx

        needed = [t for t in targets if not (source and source.lower() == t.lower() and source.lower() != "auto")]
        out: Dict[str, str] = {t: apply_glossary(text, glossary, target_lang=t) for t in targets if t not in needed}

        if not needed:
            return out

        prompt = (
            f"You are a professional realtime simultaneous conference interpreter.\n"
            f"Translate the following text from {get_language_name(source)} into these target languages: {', '.join(needed)}.\n"
            f"Return strictly a valid JSON object mapping each language code to its plain translated text (e.g. {json.dumps({t: '...' for t in needed})}).\n\n"
            f"Text: {text}"
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "temperature": 0.1,
                "maxOutputTokens": 800,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=3.8) as client:
                resp = await client.post(url, json=payload)

            if resp.status_code == 200:
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        raw_json = parts[0].get("text", "{}")
                        parsed = json.loads(raw_json)
                        if isinstance(parsed, dict):
                            for t in needed:
                                val = str(parsed.get(t) or parsed.get(t.lower()) or "").strip()
                                if val:
                                    out[t] = apply_glossary(val, glossary, target_lang=t)
                            if all(out.get(t) for t in needed):
                                if self._last_active_provider != "gemini_live":
                                    logging.info("[Translation] Switching back to Gemini Live")
                                    self._last_active_provider = "gemini_live"
                                self.circuit_breaker.record_success()
                                return out

            elif resp.status_code in (401, 403):
                logging.error("[GeminiLive] Authentication error")
                self.circuit_breaker.trip(ErrorType.AUTH_ERROR, f"HTTP {resp.status_code}")
            else:
                self.circuit_breaker.trip(ErrorType.NETWORK_ERROR, f"HTTP {resp.status_code}")

        except Exception as e:
            logging.warning(f"[GeminiLive] Batch translation error: {redact_secrets(str(e), self.api_key)}")
            self.circuit_breaker.trip(ErrorType.NETWORK_ERROR, str(e))

        # Fallback to Google Translate for any missing targets
        if self._last_active_provider != "google_translate":
            logging.warning("[Translation] Switching to Google Translate")
            logging.info("[Translation] Google Translate fallback active")
            self._last_active_provider = "google_translate"

        missing_targets = [t for t in targets if not out.get(t)]
        if missing_targets:
            fallback_res = await self.fallback.translate_batch(
                text, source=source, targets=missing_targets, glossary=glossary
            )
            out.update(fallback_res)

        return out
