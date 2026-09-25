import asyncio
import json
import logging
import time
from typing import Any, List
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.gemini_live import (
    GeminiLiveTranslator,
    GeminiLiveProvider,
    GeminiLiveCircuitBreaker,
    LatencyTelemetry,
    ErrorType,
    ConnectionState,
    redact_secrets,
    build_system_instruction,
    global_gemini_live_circuit,
)
from app.translation_provider import GoogleTranslationProvider, apply_glossary


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech", "whisper", "nemotron", "deepgram", "elevenlabs", "gemini_live"})
    return TestClient(main.app)


class MockGeminiWebSocket:
    """Mock Gemini Live BidiGenerateContent WebSocket for deterministic testing."""

    def __init__(self, should_fail_setup: bool = False, auth_error: bool = False):
        self.sent_messages: List[str] = []
        self.closed: bool = False
        self.should_fail_setup = should_fail_setup
        self.auth_error = auth_error
        self._recv_queue: asyncio.Queue = asyncio.Queue()

    async def send(self, msg: str) -> None:
        self.sent_messages.append(msg)
        parsed = json.loads(msg)
        
        # Handle setup message
        if "setup" in parsed:
            if self.auth_error:
                await self._recv_queue.put(json.dumps({
                    "error": {"code": 401, "message": "API key not valid. Please pass a valid API key."}
                }))
            elif self.should_fail_setup:
                await self._recv_queue.put(json.dumps({
                    "error": {"code": 400, "message": "Invalid model specified"}
                }))
            else:
                # Send setupComplete
                await self._recv_queue.put(json.dumps({"setupComplete": {}}))

        # Handle realtimeInput (audio or text)
        elif "realtimeInput" in parsed:
            media = parsed["realtimeInput"].get("mediaChunks", [])
            client_content = parsed["realtimeInput"].get("clientContent")

            if media:
                # Return streaming translation chunks
                await self._recv_queue.put(json.dumps({
                    "serverContent": {
                        "modelTurn": {
                            "parts": [{"text": "Hola a todos "}],
                        },
                        "turnComplete": False,
                    }
                }))
                await self._recv_queue.put(json.dumps({
                    "serverContent": {
                        "modelTurn": {
                            "parts": [{"text": "bienvenidos a la conferencia."}],
                        },
                        "turnComplete": True,
                    }
                }))
            elif client_content:
                turns = client_content.get("turns", [])
                text_in = turns[0]["parts"][0]["text"] if turns else ""
                await self._recv_queue.put(json.dumps({
                    "serverContent": {
                        "modelTurn": {
                            "parts": [{"text": f"Traducido: {text_in}"}],
                        },
                        "turnComplete": True,
                    }
                }))

    async def recv(self) -> str:
        return await self._recv_queue.get()

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.closed and self._recv_queue.empty():
            raise StopAsyncIteration
        try:
            val = await asyncio.wait_for(self._recv_queue.get(), timeout=0.5)
            return val
        except asyncio.TimeoutError:
            raise StopAsyncIteration

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def reset_global_circuit():
    global_gemini_live_circuit.record_success()
    yield
    global_gemini_live_circuit.record_success()


def test_build_system_instruction_languages_and_glossary():
    prompt = build_system_instruction("en", "es", glossary={"Kubernetes": "Kubernetes", "deploy": "despliegue"})
    assert "English to Spanish" in prompt
    assert "Kubernetes" in prompt
    assert "Do not explain" in prompt
    assert "Do not summarize" in prompt
    assert "Preserve names, numbers, technical terminology and acronyms" in prompt


def test_redact_secrets_never_leaks_api_keys():
    secret_key = "AIzaSySecretTestKey123456789012345"
    log_sample = f"Connecting to wss://generativelanguage.googleapis.com/ws?key={secret_key} with error 401"
    
    redacted = redact_secrets(log_sample, secret_key)
    assert secret_key not in redacted
    assert "[REDACTED_API_KEY]" in redacted or "[REDACTED_KEY]" in redacted or "[REDACTED_AIZA_KEY]" in redacted


def test_gemini_live_connect_and_setup_success(monkeypatch):
    async def _test():
        mock_ws = MockGeminiWebSocket()
        import app.gemini_live as gl_mod
        async def mock_connect(url):
            return mock_ws
        monkeypatch.setattr(gl_mod.ws_lib, "connect", mock_connect)

        cb = GeminiLiveCircuitBreaker()
        translator = GeminiLiveTranslator(api_key="valid-test-key", circuit_breaker=cb)

        connected = await translator.connect(source_lang="en", target_lang="es")
        assert connected is True
        assert translator.state == ConnectionState.CONNECTED
        assert translator._setup_complete is True
        assert cb.is_available() is True
        assert len(mock_ws.sent_messages) == 1
        assert "setup" in mock_ws.sent_messages[0]

    asyncio.run(_test())


def test_gemini_live_send_audio_and_stream_responses(monkeypatch):
    async def _test():
        mock_ws = MockGeminiWebSocket()
        import app.gemini_live as gl_mod
        async def mock_connect(url):
            return mock_ws
        monkeypatch.setattr(gl_mod.ws_lib, "connect", mock_connect)

        translator = GeminiLiveTranslator(api_key="valid-test-key")
        await translator.connect(source_lang="en", target_lang="es")

        # Send audio chunk
        dummy_pcm = b"\x00\x01" * 1600
        sent = await translator.send_audio(dummy_pcm)
        assert sent is True
        assert len(mock_ws.sent_messages) == 2

        # Receive translations
        chunks = []
        async for text, is_complete, telem in translator.stream_responses():
            chunks.append((text, is_complete))

        assert len(chunks) >= 2
        assert "Hola a todos" in chunks[0][0]
        # The last chunk is turnComplete
        assert chunks[-1][1] is True
        assert "bienvenidos a la conferencia" in chunks[-1][0]

    asyncio.run(_test())


def test_circuit_breaker_trips_immediately_on_401(monkeypatch):
    async def _test():
        mock_ws = MockGeminiWebSocket(auth_error=True)
        import app.gemini_live as gl_mod
        async def mock_connect(url):
            return mock_ws
        monkeypatch.setattr(gl_mod.ws_lib, "connect", mock_connect)

        cb = GeminiLiveCircuitBreaker()
        translator = GeminiLiveTranslator(api_key="invalid-key-401", circuit_breaker=cb)

        connected = await translator.connect(source_lang="en", target_lang="es")
        assert connected is False
        assert cb.is_open is True
        assert cb.last_error_type == ErrorType.AUTH_ERROR
        assert cb.is_available() is False

    asyncio.run(_test())


def test_fallback_automatically_to_google_translate(monkeypatch):
    async def _test():
        class MockFallbackGoogle(GoogleTranslationProvider):
            async def translate(self, text, source="auto", target="es", glossary=None):
                return f"GoogleTranslated: {text}"

        fallback = MockFallbackGoogle()
        cb = GeminiLiveCircuitBreaker()
        # Trip circuit to simulate Gemini down
        cb.trip(ErrorType.AUTH_ERROR, "Simulated 401")

        provider = GeminiLiveProvider(api_key="invalid-key", fallback=fallback)
        provider.circuit_breaker = cb

        result = await provider.translate("Welcome to Nerdearla", source="en", target="es")
        assert result == "GoogleTranslated: Welcome to Nerdearla"
        assert provider._last_active_provider == "google_translate"

    asyncio.run(_test())


def test_failback_to_gemini_when_recovered(monkeypatch):
    async def _test():
        class MockFallbackGoogle(GoogleTranslationProvider):
            async def translate(self, text, source="auto", target="es", glossary=None):
                return f"GoogleTranslated: {text}"

        fallback = MockFallbackGoogle()
        cb = GeminiLiveCircuitBreaker()
        cb.trip(ErrorType.NETWORK_ERROR, "Temporary drop")

        provider = GeminiLiveProvider(api_key="valid-test-key", fallback=fallback)
        provider.circuit_breaker = cb

        # 1. First call uses fallback because circuit is open
        res1 = await provider.translate("Hello", source="en", target="es")
        assert "GoogleTranslated" in res1
        assert provider._last_active_provider == "google_translate"

        # 2. Mock REST Gemini generation for recovery
        import httpx
        class MockHttpxResponse:
            status_code = 200
            def json(self):
                return {
                    "candidates": [{
                        "content": {"parts": [{"text": "Hola mundo"}]}
                    }]
                }

        async def mock_post(*args, **kwargs):
            return MockHttpxResponse()

        monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

        # 3. Simulate recovery (circuit reset)
        cb.record_success()
        res2 = await provider.translate("Hello world", source="en", target="es")
        assert res2 == "Hola mundo"
        assert provider._last_active_provider == "gemini_live"

    asyncio.run(_test())


def test_health_check_endpoint_reflects_gemini_live_availability(client, monkeypatch):
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-gemini-key")
    global_gemini_live_circuit.record_success()

    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["gemini_live_available"] is True
    assert data["gemini_live_status"] == "Gemini Live CONNECTED"
    assert data["active_fallback"] == "google_translate"

    # Now trip the circuit
    global_gemini_live_circuit.trip(ErrorType.AUTH_ERROR, "Invalid Key 401")
    resp_down = client.get("/health")
    data_down = resp_down.json()
    assert data_down["gemini_live_available"] is False
    assert data_down["gemini_live_status"] == "Google Translate FALLBACK"


def test_multi_session_isolation():
    """Ensure two concurrent sessions maintain distinct telemetry, state, and buffers."""
    cb1 = GeminiLiveCircuitBreaker()
    cb2 = GeminiLiveCircuitBreaker()

    t1 = GeminiLiveTranslator(api_key="key-session-1", circuit_breaker=cb1)
    t2 = GeminiLiveTranslator(api_key="key-session-2", circuit_breaker=cb2)

    t1.source_lang = "en"
    t1.target_lang = "es"
    t1.telemetry.gemini_sent_at = 100.0
    t1.telemetry.gemini_response_at = 100.25

    t2.source_lang = "pt"
    t2.target_lang = "it"
    t2.telemetry.gemini_sent_at = 200.0
    t2.telemetry.gemini_response_at = 200.10

    assert t1.telemetry.gemini_latency_ms == 250.0
    assert t2.telemetry.gemini_latency_ms == 100.0
    assert t1.source_lang != t2.source_lang
    assert t1.target_lang != t2.target_lang


def test_ws_gemini_live_endpoint_with_google_fallback(client, monkeypatch):
    """Test /ws/gemini-live endpoint with Google Translate fallback when Gemini Live WS is unavailable."""
    monkeypatch.setattr(main, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "AUTH_ENABLED", True)

    # Force circuit open so it uses fallback
    global_gemini_live_circuit.trip(ErrorType.WEBSOCKET_ERROR, "Simulated failure")

    class MockFallback(GoogleTranslationProvider):
        async def translate(self, text, source="auto", target="es", glossary=None):
            return f"FallbackTr: {text}"

    monkeypatch.setattr(main, "GoogleTranslationProvider", lambda: MockFallback())

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws/gemini-live", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({
            "type": "config",
            "translate": {"src": "en", "dests": ["es"]},
            "mode": "B",
        })
        ws.send_json({
            "type": "transcript",
            "text": "Hello world from keynote",
            "interim": False,
        })
        data = ws.receive_json()

    assert data["type"] == "final"
    assert data["original"] == "Hello world from keynote"
    assert data["translations"]["es"] == "FallbackTr: Hello world from keynote"
    assert data["provider"] == "google_translate"
    assert data["provider_status"] == "FALLBACK"
