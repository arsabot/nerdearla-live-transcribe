import re

import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "AUTH_ENABLED", True)
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech", "whisper", "nemotron", "deepgram", "elevenlabs", "gemini_live"})
    return TestClient(main.app)


def _assert_login_h1(html: str) -> None:
    assert re.search(r"<h1[^>]*>Sign in</h1>", html)


def test_get_index_requires_password(client):
    resp = client.get("/")
    assert resp.status_code == 200
    _assert_login_h1(resp.text)
    assert "Incorrect password" not in resp.text


def test_get_index_wrong_password_shows_error(client):
    resp = client.get("/?pwd=wrong")
    assert resp.status_code == 200
    _assert_login_h1(resp.text)
    assert "Incorrect password" in resp.text


def test_get_index_correct_password_serves_index_html(client):
    resp = client.get("/?pwd=test-password", follow_redirects=False)
    assert resp.status_code == 303

    resp = client.get("/")
    assert resp.status_code == 200
    assert "<title>Live Translator</title>" in resp.text


def test_get_deepgram_always_redirects_to_index(client):
    """The /deepgram legacy endpoint now always redirects to /."""
    resp = client.get("/deepgram", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers.get("location") == "/"


class _FakeTranslation:
    def __init__(self, text: str):
        self.text = text


def test_ws_translates_text(client, monkeypatch):
    class FakeAsyncTranslator:
        def __init__(self):
            self.calls = []

        async def translate(self, text, src, dest):
            self.calls.append((text, src, dest))
            return _FakeTranslation(f"{dest}:{text}")

    fake_translator = FakeAsyncTranslator()
    monkeypatch.setattr(main, "Translator", lambda: fake_translator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_text("Ahoj")
        data = ws.receive_json()

    assert data == {"original": "Ahoj", "en": "en:Ahoj", "ru": "ru:Ahoj"}
    assert fake_translator.calls == [("Ahoj", "cs", "en"), ("Ahoj", "cs", "ru")]


def test_ws_typed_translates_single_dest(client, monkeypatch):
    class FakeAsyncTranslator:
        def __init__(self):
            self.calls = []

        async def translate(self, text, src, dest):
            self.calls.append((text, src, dest))
            return _FakeTranslation(f"{dest}:{text}")

    fake_translator = FakeAsyncTranslator()
    monkeypatch.setattr(main, "Translator", lambda: fake_translator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "config", "translate": {"src": "cs", "dests": ["en"]}})
        ws.send_json({"type": "final", "text": "Ahoj", "src": "cs", "dests": ["en"]})
        data = ws.receive_json()

    assert data["type"] == "final"
    assert data["original"] == "Ahoj"
    assert data["dests"] == ["en"]
    assert data["translations"] == {"en": "en:Ahoj"}
    assert isinstance(data.get("timing", {}), dict)
    assert fake_translator.calls == [("Ahoj", "cs", "en")]


def test_ws_empty_text_does_not_call_translator(client, monkeypatch):
    class FakeAsyncTranslator:
        async def translate(self, *_args, **_kwargs):
            raise AssertionError("translate should not be called for empty input")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_text("   ")
        data = ws.receive_json()

    assert data == {"original": "", "en": "", "ru": ""}


def test_ws_ping_pong(client, monkeypatch):
    """Server responds to keepalive ping with pong."""

    class FakeAsyncTranslator:
        async def translate(self, *_args, **_kwargs):
            raise AssertionError("translate should not be called for ping")

    monkeypatch.setattr(main, "Translator", FakeAsyncTranslator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "ping"})
        data = ws.receive_json()

    assert data == {"type": "pong"}


def test_ws_elevenlabs_missing_api_key_returns_error(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "")

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws/elevenlabs", headers={"origin": "http://testserver"}) as ws:
        payload = ws.receive_json()

    assert payload == {"error": "ELEVENLABS_API_KEY not configured"}


def test_ws_deepgram_missing_api_key_returns_error(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "")

    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        payload = ws.receive_json()

    assert payload == {"error": "DEEPGRAM_API_KEY not configured"}


def test_ws_deepgram_init_failure_sends_error(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")

    class BoomDeepgramClient:
        def __init__(self, api_key):
            raise RuntimeError("boom")

    monkeypatch.setattr(main, "DeepgramClient", BoomDeepgramClient)

    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        # The endpoint blocks on the first client message (config or audio) before
        # it constructs DeepgramClient, so send a config to trigger the failing init.
        # Without this the server never reaches the boom and both sides deadlock.
        ws.send_json({"type": "config", "deepgram": {"language": "cs"}})
        payload = ws.receive_json()

    assert payload == {"error": "boom"}


def test_ws_deepgram_missing_sdk_returns_error(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")
    monkeypatch.setattr(main, "DeepgramClient", None)

    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        payload = ws.receive_json()

    assert payload == {"error": "deepgram-sdk not installed"}


def test_ws_deepgram_happy_path_emits_interim_and_final(client, monkeypatch):
    monkeypatch.setattr(main, "DEEPGRAM_API_KEY", "test-key")

    class FakeAlt:
        def __init__(self, transcript: str):
            self.transcript = transcript

    class FakeChannel:
        def __init__(self, transcript: str):
            self.alternatives = [FakeAlt(transcript)]

    class FakeListenV1Results:
        def __init__(self, transcript: str, is_final: bool):
            self.channel = FakeChannel(transcript)
            self.is_final = is_final

    class FakeAsyncTranslator:
        def __init__(self):
            self.calls = []

        async def translate(self, text, src, dest):
            self.calls.append((text, src, dest))
            return _FakeTranslation(f"{dest}:{text}")

    translator = FakeAsyncTranslator()
    monkeypatch.setattr(main, "Translator", lambda: translator)

    class FakeDgSocket:
        def __init__(self):
            self._handlers = {}
            self.sent_media = []
            self.finalized = False
            self.closed = False

        def on(self, event_type, callback):
            self._handlers[event_type] = callback

        def start_listening(self):
            msg_cb = self._handlers.get(main.EventType.MESSAGE)
            if msg_cb:
                msg_cb(FakeListenV1Results("prubezne", False))
                msg_cb(FakeListenV1Results("finalni", True))

        def send_media(self, data):
            self.sent_media.append(data)

        def send_finalize(self, _message=None):
            self.finalized = True

        def send_close_stream(self, _message=None):
            self.closed = True

    class _FakeSocketIterator:
        def __init__(self, socket):
            self._socket = socket
            self._sent = False
            self.closed = False

        def __iter__(self):
            return self

        def __next__(self):
            if self._sent:
                raise StopIteration
            self._sent = True
            return self._socket

        def close(self):
            self.closed = True

    fake_socket = FakeDgSocket()
    fake_iter = _FakeSocketIterator(fake_socket)

    class FakeDeepgramClient:
        def __init__(self, api_key):
            self.api_key = api_key

            class _V1:
                def connect(self, **_kwargs):
                    return fake_iter

            class _Listen:
                v1 = _V1()

            self.listen = _Listen()

    monkeypatch.setattr(main, "DeepgramClient", FakeDeepgramClient)

    client.post("/login", data={"password": "test-password", "next": "/deepgram"}, follow_redirects=False)

    with client.websocket_connect("/ws/deepgram", headers={"origin": "http://testserver"}) as ws:
        ws.send_bytes(b"\x00\x01")
        interim = ws.receive_json()
        final = ws.receive_json()

    assert interim["type"] == "interim"
    assert interim["original"] == "prubezne"
    assert interim["dests"] == ["en", "ru"]
    assert interim["translations"] == {"en": "", "ru": ""}
    assert interim["en"] == ""
    assert interim["ru"] == ""
    assert isinstance(interim.get("timing", {}), dict)

    assert final["type"] == "final"
    assert final["original"] == "finalni"
    assert final["dests"] == ["en", "ru"]
    assert final["translations"] == {"en": "en:finalni", "ru": "ru:finalni"}
    assert final["en"] == "en:finalni"
    assert final["ru"] == "ru:finalni"
    assert isinstance(final.get("timing", {}), dict)
    assert translator.calls == [("finalni", "cs", "en"), ("finalni", "cs", "ru")]
    assert fake_socket.sent_media == [b"\x00\x01"]


# --- /api/elevenlabs/token tests ---


def test_elevenlabs_token_requires_auth(client):
    resp = client.post("/api/elevenlabs/token", json={})
    assert resp.status_code == 401


def test_elevenlabs_token_missing_api_key(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "")

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/elevenlabs/token", json={})
    assert resp.status_code == 400
    assert "No ElevenLabs API key" in resp.json()["detail"]


def test_elevenlabs_token_uses_env_key(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "xi-env-key")

    import httpx

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"token": "tok_abc123"}

        def raise_for_status(self):
            pass

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            assert kwargs.get("headers", {}).get("xi-api-key") == "xi-env-key"
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeAsyncClient())

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/elevenlabs/token", json={})
    assert resp.status_code == 200
    assert resp.json() == {"token": "tok_abc123"}


def test_elevenlabs_token_uses_client_key(client, monkeypatch):
    monkeypatch.setattr(main, "ELEVENLABS_API_KEY", "xi-env-key")

    import httpx

    class FakeResponse:
        status_code = 200

        def json(self):
            return {"token": "tok_client"}

        def raise_for_status(self):
            pass

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            # Should use the client-provided key, not env key.
            assert kwargs.get("headers", {}).get("xi-api-key") == "xi-my-key"
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", lambda **kw: FakeAsyncClient())

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = client.post("/api/elevenlabs/token", json={"api_key": "xi-my-key"})
    assert resp.status_code == 200
    assert resp.json() == {"token": "tok_client"}


# --- AUTH_ENABLED tests ---


def test_auth_disabled_skips_login(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "APP_PASSWORD", "")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech"})

    c = TestClient(main.app)
    resp = c.get("/")
    assert resp.status_code == 200
    assert "<title>Live Translator</title>" in resp.text


def test_auth_disabled_ws_no_cookie_needed(monkeypatch):
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "APP_PASSWORD", "")

    c = TestClient(main.app)
    with c.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        data = ws.receive_json()
        assert data == {"type": "pong"}


# --- ENABLED_ENGINES tests ---


def test_enabled_engines_passed_to_template(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech", "deepgram"})

    c = TestClient(main.app)
    c.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = c.get("/")
    assert resp.status_code == 200
    # webspeech and deepgram should NOT have disabled attribute
    assert 'value="webspeech" ' in resp.text  # not disabled
    assert 'value="deepgram" ' in resp.text    # not disabled
    # elevenlabs should be disabled
    assert 'value="elevenlabs" disabled' in resp.text


def test_enabled_engines_includes_whisper_in_template(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"whisper"})

    c = TestClient(main.app)
    c.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = c.get("/")
    assert resp.status_code == 200
    assert 'value="whisper" ' in resp.text
    # Engine module is loaded on demand via dynamic import(); ensure there is no
    # eager <script src="..."> tag pulling it during initial page load.
    assert '<script src="/static/whisper/whisper-engine.mjs"' not in resp.text


def test_static_whisper_worklet_served(client):
    resp = client.get("/static/whisper/pcm-worklet.js")
    assert resp.status_code == 200
    assert "Int16PCMProcessor" in resp.text
    # /static/whisper/* must always revalidate so browsers can't pin a stale
    # (possibly ABI-incompatible) engine copy across deploys.
    assert resp.headers.get("cache-control") == "no-cache"


def test_static_whisper_engine_served(client):
    resp = client.get("/static/whisper/whisper-engine.mjs")
    assert resp.status_code == 200
    assert "WhisperLocalEngine" in resp.text
    assert resp.headers.get("cache-control") == "no-cache"


def test_csp_allows_whisper_runtime_sources(client):
    # The local Whisper engine needs WASM execution plus Transformers.js/ONNX from
    # jsdelivr and the model weights from Hugging Face. The CSP must permit these.
    resp = client.get("/health")
    csp = resp.headers.get("Content-Security-Policy", "")
    assert "'wasm-unsafe-eval'" in csp
    assert "https://cdn.jsdelivr.net" in csp
    assert "https://huggingface.co" in csp


def test_cross_origin_isolation_headers(client):
    # COOP+COEP make the page cross-origin isolated, which enables SharedArrayBuffer
    # and lets ONNX Runtime Web run the Whisper WASM/CPU path multi-threaded.
    resp = client.get("/health")
    assert resp.headers.get("Cross-Origin-Opener-Policy") == "same-origin"
    assert resp.headers.get("Cross-Origin-Embedder-Policy") == "credentialless"


def test_enabled_engines_includes_nemotron_in_template(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"nemotron"})

    c = TestClient(main.app)
    c.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = c.get("/")
    assert resp.status_code == 200
    assert 'value="nemotron" ' in resp.text


def test_static_nemotron_engine_served(client):
    resp = client.get("/static/nemotron/nemotron-engine.mjs")
    assert resp.status_code == 200
    assert "NemotronLocalEngine" in resp.text
    # Engine code revalidates like Whisper's so a stale copy can't be pinned.
    assert resp.headers.get("cache-control") == "no-cache"


def test_nemotron_model_assets_cached_immutably(client):
    # The ~1.2 GB fp16 weights are content-stable and must be cached hard. The
    # header is path-based, so it applies even before the model files are built.
    resp = client.get("/static/nemotron/models/config.json")
    assert resp.headers.get("cache-control") == "public, max-age=31536000, immutable"


def test_csp_allows_nemotron_onnxruntime(client):
    # The local Nemotron engine loads onnxruntime-web 1.20.1 (encoder on WebGPU,
    # decoder on WASM); the CSP must permit that pinned build.
    resp = client.get("/health")
    csp = resp.headers.get("Content-Security-Policy", "")
    assert "onnxruntime-web@1.20.1" in csp


def test_enabled_engines_default_webspeech_only(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"webspeech"})

    c = TestClient(main.app)
    c.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    resp = c.get("/")
    assert resp.status_code == 200
    assert 'value="deepgram" disabled' in resp.text
    assert 'value="elevenlabs" disabled' in resp.text


# --- /health endpoint ---


def test_health_endpoint_no_auth_needed(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "active_sessions" in data
    assert "active_viewers" in data


# --- Rate limiting on /login ---


def test_login_rate_limiting(client, monkeypatch):
    # Reset rate limiter state.
    monkeypatch.setattr(main, "_LOGIN_ATTEMPTS", {})
    monkeypatch.setattr(main, "_LOGIN_MAX_ATTEMPTS", 3)

    for _ in range(3):
        resp = client.post(
            "/login",
            data={"password": "wrong", "next": "/"},
            follow_redirects=False,
        )
        assert resp.status_code == 200  # renders login form

    resp = client.post(
        "/login",
        data={"password": "wrong", "next": "/"},
        follow_redirects=False,
    )
    assert resp.status_code == 429


# --- Nerdearla Live Multi-Session & Stage Isolation Tests ---


def test_session_rest_crud(client):
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    # 1. List sessions
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert len(data["sessions"]) >= 2

    # 2. Create new stage
    create_resp = client.post(
        "/api/sessions",
        json={
            "id": "stage-test",
            "name": "Testing & QA Stage",
            "speaker": "QA Architect",
            "source_language": "en",
            "target_languages": ["es"],
            "description": "Automated testing discussion",
            "glossary": {"pipeline": "tubería"},
        },
    )
    assert create_resp.status_code == 200
    created = create_resp.json()
    assert created["id"] == "stage-test"
    assert created["name"] == "Testing & QA Stage"

    # 3. Get single stage
    get_resp = client.get("/api/sessions/stage-test")
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Testing & QA Stage"

    # 4. Update stage
    patch_resp = client.patch(
        "/api/sessions/stage-test",
        json={"status": "paused", "speaker": "Updated Speaker"},
    )
    assert patch_resp.status_code == 200
    assert patch_resp.json()["status"] == "paused"
    assert patch_resp.json()["speaker"] == "Updated Speaker"

    # 5. Delete stage
    del_resp = client.delete("/api/sessions/stage-test")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # 6. Verify 404
    get_404 = client.get("/api/sessions/stage-test")
    assert get_404.status_code == 404


def test_session_isolation_and_broadcast(client, monkeypatch):
    """
    Guarantees requirement: Events from Session A must NEVER leak into Session B.
    """
    class FakeAsyncTranslator:
        def __init__(self):
            self.calls = []

        async def translate(self, text, src="auto", dest="es", **_kwargs):
            self.calls.append((text, src, dest))
            return _FakeTranslation(f"ES:{text}")

    fake_translator = FakeAsyncTranslator()
    monkeypatch.setattr("app.translator.RobustTranslator", lambda *a, **k: fake_translator)
    monkeypatch.setattr("app.translation_provider.RobustTranslator", lambda *a, **k: fake_translator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    # Connect Viewer to Stage A
    with client.websocket_connect("/ws/session/stage-a/viewer", headers={"origin": "http://testserver"}) as ws_a:
        handshake_a = ws_a.receive_json()
        assert handshake_a["type"] == "handshake"
        assert handshake_a["session"]["id"] == "stage-a"

        # Connect Viewer to Stage B
        with client.websocket_connect("/ws/session/stage-b/viewer", headers={"origin": "http://testserver"}) as ws_b:
            handshake_b = ws_b.receive_json()
            assert handshake_b["type"] == "handshake"
            assert handshake_b["session"]["id"] == "stage-b"

            # Inject event into Stage A
            inject_resp = client.post(
                "/api/session/stage-a/inject",
                json={"type": "final", "text": "Welcome to Stage A AI keynote!"},
            )
            assert inject_resp.status_code == 200
            assert inject_resp.json()["status"] == "ok"

            # Viewer A MUST receive the event
            event_a = ws_a.receive_json()
            assert event_a["session_id"] == "stage-a"
            assert event_a["type"] == "final"
            assert "Welcome to Stage A" in event_a["original"]
            assert "translations" in event_a
            assert "metrics" in event_a

            # Ping Viewer B to ensure its connection is active and verify NO Stage A message was queued
            ws_b.send_json({"type": "ping"})
            pong_b = ws_b.receive_json()
            assert pong_b == {"type": "pong"}


def test_glossary_substitution():
    from app.translation_provider import apply_glossary

    glossary = {
        "Kubernetes": "Kubernetes",
        "container": "contenedor",
        "pull request": "pull request",
        "deployment": "despliegue",
    }
    text = "We configured the deployment of each container with a new pull request."
    result = apply_glossary(text, glossary)
    assert "despliegue" in result
    assert "contenedor" in result
    assert "pull request" in result


def test_export_vtt_and_srt_and_txt(client, monkeypatch):
    class FakeAsyncTranslator:
        async def translate(self, text, src="auto", dest="es", **_kwargs):
            return _FakeTranslation(f"[{dest}] {text}")

    fake_translator = FakeAsyncTranslator()
    monkeypatch.setattr(main, "Translator", lambda: fake_translator)
    monkeypatch.setattr("app.translator.RobustTranslator", lambda *a, **k: fake_translator)
    monkeypatch.setattr("app.translation_provider.RobustTranslator", lambda *a, **k: fake_translator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    # Inject sample final events
    client.post(
        "/api/session/stage-a/inject",
        json={"type": "final", "text": "Welcome to Nerdearla Live!"},
    )
    client.post(
        "/api/session/stage-a/inject",
        json={"type": "final", "text": "Simultaneous translation is active."},
    )

    # 1. Export VTT
    vtt_resp = client.get("/api/session/stage-a/export/vtt?lang=es")
    assert vtt_resp.status_code == 200
    assert "WEBVTT" in vtt_resp.text
    assert "-->" in vtt_resp.text

    # 2. Export SRT
    srt_resp = client.get("/api/session/stage-a/export/srt?lang=es")
    assert srt_resp.status_code == 200
    assert "00:00:" in srt_resp.text
    assert "-->" in srt_resp.text

    # 3. Export TXT
    txt_resp = client.get("/api/session/stage-a/export/txt?lang=es")
    assert txt_resp.status_code == 200
    assert len(txt_resp.text.strip()) > 0


def test_pages_render_successfully(client):
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    # 1. Audience Hub
    resp_hub = client.get("/")
    assert resp_hub.status_code == 200
    assert "Conference Stages" in resp_hub.text

    # 2. Stage Viewer
    resp_session = client.get("/session/stage-a")
    assert resp_session.status_code == 200
    assert "AI &amp; Open Source" in resp_session.text or "AI & Open Source" in resp_session.text

    # 3. Producer Dashboard
    resp_producer = client.get("/producer")
    assert resp_producer.status_code == 200
    assert "Producer Control Room" in resp_producer.text

    # 4. 2-Stage Demo
    resp_demo = client.get("/demo")
    assert resp_demo.status_code == 200
    assert "Simultaneous 2-Stage" in resp_demo.text

    # 5. Standalone Translator
    resp_standalone = client.get("/standalone")
    assert resp_standalone.status_code == 200

    # 6. Speaker Terminal
    resp_speaker = client.get("/speaker/stage-a")
    assert resp_speaker.status_code == 200
    assert "Speaker Terminal" in resp_speaker.text

    # 7. Smart TV Display
    resp_display = client.get("/display/stage-a")
    assert resp_display.status_code == 200
    assert "Smart TV" in resp_display.text


def test_session_history_persistence_and_handshake(client, monkeypatch):
    class FakeAsyncTranslator:
        async def translate(self, text, src, dest):
            return _FakeTranslation(f"[{dest}] {text}")

    fake_translator = FakeAsyncTranslator()
    monkeypatch.setattr(main, "Translator", lambda: fake_translator)
    monkeypatch.setattr("app.translator.RobustTranslator", lambda *a, **k: fake_translator)
    monkeypatch.setattr("app.translation_provider.RobustTranslator", lambda *a, **k: fake_translator)

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    # 1. Inject an event into Stage A
    inject_resp = client.post(
        "/api/session/stage-a/inject",
        json={"type": "final", "text": "Persistence test sentence for reload"},
    )
    assert inject_resp.status_code == 200

    # 2. Check API session detail contains history
    api_resp = client.get("/api/sessions/stage-a")
    assert api_resp.status_code == 200
    data = api_resp.json()
    assert "history" in data
    assert any("Persistence test sentence" in ev["original"] for ev in data["history"])

    # 3. Connect a new Viewer to Stage A and verify handshake includes history
    with client.websocket_connect("/ws/session/stage-a/viewer", headers={"origin": "http://testserver"}) as ws:
        handshake = ws.receive_json()
        assert handshake["type"] == "handshake"
        assert "history" in handshake
        assert any("Persistence test sentence" in ev["original"] for ev in handshake["history"])


def test_staff_protection_blocks_unauthenticated_users(monkeypatch):
    """
    Even when global audience auth is disabled (AUTH_ENABLED=False),
    Staff / Producer / Speaker routes and mutating APIs MUST still require authentication.
    """
    monkeypatch.setattr(main, "AUTH_ENABLED", False)
    monkeypatch.setattr(main, "APP_PASSWORD", "secret123")
    monkeypatch.setattr(main, "STAFF_PASSWORD", "secret123")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")

    unauthed_client = TestClient(main.app)

    # 1. Audience views are open (Status 200 without password)
    assert unauthed_client.get("/").status_code == 200
    assert unauthed_client.get("/session/stage-a").status_code == 200
    assert unauthed_client.get("/display/stage-a").status_code == 200

    # 2. Staff views MUST prompt for login
    producer_resp = unauthed_client.get("/producer")
    assert producer_resp.status_code == 200
    _assert_login_h1(producer_resp.text)

    speaker_resp = unauthed_client.get("/speaker/stage-a")
    assert speaker_resp.status_code == 200
    _assert_login_h1(speaker_resp.text)

    demo_resp = unauthed_client.get("/demo")
    assert demo_resp.status_code == 200
    _assert_login_h1(demo_resp.text)

    standalone_resp = unauthed_client.get("/standalone")
    assert standalone_resp.status_code == 200
    _assert_login_h1(standalone_resp.text)

    # 3. Mutating APIs MUST return 401 Unauthorized
    post_resp = unauthed_client.post("/api/sessions", json={"id": "test", "name": "Test"})
    assert post_resp.status_code == 401
    assert post_resp.json()["detail"] == "staff_authentication_required"

    delete_resp = unauthed_client.delete("/api/sessions/stage-a")
    assert delete_resp.status_code == 401

    inject_resp = unauthed_client.post("/api/session/stage-a/inject", json={"text": "hacked"})
    assert inject_resp.status_code == 401

    # 4. Once logged in as staff, all staff views and mutating APIs are accessible
    unauthed_client.post("/login", data={"password": "secret123", "next": "/producer"}, follow_redirects=False)
    assert unauthed_client.get("/producer").status_code == 200
    assert "Producer Control Room" in unauthed_client.get("/producer").text

    # 5. Logout clears the staff session
    logout_resp = unauthed_client.get("/logout", follow_redirects=False)
    assert logout_resp.status_code == 303
    assert logout_resp.headers.get("location") == "/"


def test_gemini_live_websocket_missing_api_key_fallback_to_google(client, monkeypatch):
    """
    Test /ws/gemini-live websocket endpoint gracefully falls back to Google Translate
    when GEMINI_API_KEY is not configured without interrupting translation delivery.
    """
    monkeypatch.setattr(main, "GEMINI_API_KEY", "")

    class MockFallback(main.GoogleTranslationProvider):
        async def translate(self, text, source="auto", target="es", glossary=None):
            return f"FallbackGoogle: {text}"

    monkeypatch.setattr(main, "GoogleTranslationProvider", lambda: MockFallback())

    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws/gemini-live", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({"type": "transcript", "text": "Hello world", "interim": False})
        data = ws.receive_json()
        assert data["type"] == "final"
        assert data["translations"]["es"] == "FallbackGoogle: Hello world"
        assert data["provider"] == "google_translate"
        assert data["provider_status"] == "FALLBACK"


def test_enabled_engines_includes_gemini_live_in_template(monkeypatch):
    monkeypatch.setattr(main, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(main, "AUTH_SECRET", "test-secret")
    monkeypatch.setattr(main, "ENABLED_ENGINES", {"gemini_live"})

    c = TestClient(main.app)
    c.post("/login", data={"password": "test-password", "next": "/standalone"}, follow_redirects=False)

    resp = c.get("/standalone")
    assert resp.status_code == 200
    assert 'value="gemini_live" ' in resp.text


def test_clear_session_history_and_broadcast(client, monkeypatch):
    """
    Test that clearing a session history removes transcripts and broadcasts clear_history to connected viewers.
    """
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    # 1. Inject a message
    inject_resp = client.post(
        "/api/session/stage-a/inject",
        json={"text": "Message to be cleared", "type": "final"},
    )
    assert inject_resp.status_code == 200

    # 2. Check history has items
    get_resp = client.get("/api/sessions/stage-a")
    assert get_resp.status_code == 200
    assert len(get_resp.json()["history"]) >= 1

    # 3. Connect viewer and call clear endpoint
    with client.websocket_connect("/ws/session/stage-a/viewer", headers={"origin": "http://testserver"}) as ws:
        handshake = ws.receive_json()
        assert handshake["type"] == "handshake"

        clear_resp = client.post("/api/session/stage-a/clear")
        assert clear_resp.status_code == 200
        assert clear_resp.json()["status"] == "cleared"

        # Verify viewer received clear_history broadcast
        clear_event = ws.receive_json()
        assert clear_event["type"] == "clear_history"
        assert clear_event["session_id"] == "stage-a"

    # 4. Verify stage-a history is now empty
    get_after = client.get("/api/sessions/stage-a")
    assert get_after.status_code == 200
    assert len(get_after.json()["history"]) == 0


def test_clear_all_sessions_history(client):
    """
    Test that clear-history endpoint clears all sessions histories.
    """
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    # Inject to stage-a and stage-b
    client.post("/api/session/stage-a/inject", json={"text": "Hello Stage A", "type": "final"})
    client.post("/api/session/stage-b/inject", json={"text": "Hello Stage B", "type": "final"})

    # Clear all
    clear_all_resp = client.post("/api/sessions/clear-history")
    assert clear_all_resp.status_code == 200
    assert clear_all_resp.json()["status"] == "all_cleared"

    # Verify both stages are empty
    res_a = client.get("/api/sessions/stage-a").json()
    res_b = client.get("/api/sessions/stage-b").json()
    assert len(res_a["history"]) == 0
    assert len(res_b["history"]) == 0

def test_nerdearla_phonetic_normalization():
    """
    Test that various acoustic/phonetic speech recognition distortions of 'Nerdearla'
    are properly normalized to canonical 'Nerdearla'.
    """
    from app.translation_provider import normalize_brand_terms

    distortions = [
        ("Bienvenidos a nerdearla 2026", "Bienvenidos a Nerdearla 2026"),
        ("vamos a nerd de habla", "vamos a Nerdearla"),
        ("estamos en nerd arla", "estamos en Nerdearla"),
        ("hoy en nerdear la", "hoy en Nerdearla"),
        ("transmitiendo desde merdearla", "transmitiendo desde Nerdearla"),
        ("conferencia nerderla", "conferencia Nerdearla"),
        ("evento nerdeada", "evento Nerdearla"),
        ("hola nerd.la comunidad", "hola Nerdearla comunidad"),
    ]

    for distorted, expected in distortions:
        normalized = normalize_brand_terms(distorted)
        assert normalized == expected, f"Failed for input: {distorted} -> got {normalized}, expected {expected}"

