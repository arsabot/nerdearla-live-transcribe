"""
Tests for the latency instrumentation layer.

Verifies LatencyTracker and its integration with TranslationWorker
without breaking the existing pipeline.
"""

import asyncio
import time

import pytest

import app.services.latency_logger as latency_module
from app.services.latency_logger import LatencyTracker
from app.services.translation_worker import TranslationWorker
import app.main as main
import app.core.config as config
import app.api.routes.websockets as websockets_route
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeTr:
    async def translate(self, text, src, dest):
        class _R:
            pass
        r = _R()
        r.text = f"{dest}:{text}"
        return r


@pytest.fixture()
def debug_on(monkeypatch):
    """Enable DEBUG_LATENCY for the duration of the test."""
    monkeypatch.setattr(latency_module, "DEBUG_LATENCY", True)
    yield


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(config, "APP_PASSWORD", "test-password")
    monkeypatch.setattr(config, "STAFF_PASSWORD", "test-password")
    monkeypatch.setattr(config, "AUTH_SECRET", "test-secret")
    monkeypatch.delenv("AUTH_COOKIE_SECURE", raising=False)
    monkeypatch.setattr(config, "AUTH_ENABLED", True)
    monkeypatch.setattr(config, "ENABLED_ENGINES",
                        {"webspeech", "whisper", "nemotron", "deepgram", "elevenlabs", "gemini_live"})
    return TestClient(main.app, base_url="http://testserver.local")


# ---------------------------------------------------------------------------
# LatencyTracker unit tests
# ---------------------------------------------------------------------------

def test_tracker_events_do_not_block(debug_on):
    """Logging 200 events must return in <100ms (non-blocking)."""
    async def _inner():
        tracker = LatencyTracker("perf-session", engine="test")
        tracker.start()
        t0 = time.time()
        for i in range(200):
            tracker.event("STT_PARTIAL", event_id=f"utt-{i}", text="hello world")
        elapsed_ms = (time.time() - t0) * 1000
        await tracker.stop()
        assert elapsed_ms < 100, f"Logging 200 events blocked for {elapsed_ms:.0f}ms"
    asyncio.run(_inner())


def test_tracker_all_event_types(debug_on):
    """All event type strings can be logged without raising."""
    async def _inner():
        tracker = LatencyTracker("all-events", engine="test")
        tracker.start()
        for evt in [
            "AUDIO_RECEIVED", "VAD_SPEECH_START", "VAD_SPEECH_END",
            "STT_REQUEST", "STT_PARTIAL", "STT_STABLE", "STT_FINAL",
            "STABILITY_EMITTED", "TRANSLATION_QUEUED", "TRANSLATION_STARTED",
            "TRANSLATION_COMPLETED", "TRANSLATION_FAILED", "WEBSOCKET_SENT",
            "UI_RENDERED",
        ]:
            tracker.event(evt, event_id="utt-1", text="x", revision_id=1, queue_size=2, latency_ms=10.0)
        await tracker.stop()
    asyncio.run(_inner())


def test_tracker_note_audio_sets_utterance_start(debug_on):
    async def _inner():
        tracker = LatencyTracker("aud-sess", engine="deepgram")
        tracker.start()
        assert tracker._utterance_start_ms is None
        tracker.note_audio()
        assert tracker._utterance_start_ms is not None
        await tracker.stop()
    asyncio.run(_inner())


def test_tracker_note_audio_resets_on_silence(debug_on):
    async def _inner():
        tracker = LatencyTracker("sil-sess", engine="deepgram")
        tracker.start()
        tracker.note_audio()
        first = tracker._utterance_start_ms
        assert first is not None, "first note_audio should set _utterance_start_ms"
        # Simulate 900ms gap by shifting both _last_audio_ms far in the past
        tracker._last_audio_ms = first - 900
        # Manually set _utterance_start_ms to a known old value so we can detect the reset
        old_utt_start = first - 5000
        tracker._utterance_start_ms = old_utt_start
        tracker.note_audio()
        # The gap > 800ms should have triggered a new utterance start
        assert tracker._utterance_start_ms != old_utt_start, (
            f"Expected _utterance_start_ms to be refreshed from {old_utt_start}, "
            f"but got {tracker._utterance_start_ms}"
        )
        assert tracker._utterance_start_ms >= first, (
            f"New utterance start {tracker._utterance_start_ms} should be >= {first}"
        )
        await tracker.stop()
    asyncio.run(_inner())


def test_tracker_discarded_count(debug_on):
    async def _inner():
        tracker = LatencyTracker("disc-sess", engine="test")
        tracker.start()
        tracker.event("UI_DISCARDED", event_id="utt-1")
        tracker.event("UI_DISCARDED", event_id="utt-2")
        await asyncio.sleep(0.15)
        await tracker.stop()
        assert tracker.discarded_results == 2
    asyncio.run(_inner())


def test_tracker_error_count(debug_on):
    async def _inner():
        tracker = LatencyTracker("err-sess", engine="test")
        tracker.start()
        tracker.event("TRANSLATION_FAILED", event_id="utt-1", error="timeout")
        await asyncio.sleep(0.15)
        await tracker.stop()
        assert tracker.errors == 1
    asyncio.run(_inner())


def test_tracker_max_queue_size_tracked(debug_on):
    async def _inner():
        tracker = LatencyTracker("qs-sess", engine="test")
        tracker.start()
        for qs in [3, 7, 2]:
            tracker.event("TRANSLATION_QUEUED", event_id="utt-1", queue_size=qs)
        await asyncio.sleep(0.15)
        await tracker.stop()
        assert tracker.max_queue_size == 7
    asyncio.run(_inner())


def test_tracker_text_truncated(debug_on):
    """Texts longer than 60 chars must be truncated in the log payload."""
    from app.services.latency_logger import _truncate
    # Test the truncation function directly — it's pure and synchronous
    long_text = "a" * 200
    truncated = _truncate(long_text, max_len=60)
    assert len(truncated) <= 63, "_truncate did not shorten text"
    assert truncated.endswith("..."), "truncated text must end with ..."

    # Also verify it goes through the event payload correctly via async
    async def _inner():
        tracker = LatencyTracker("trunc-sess", engine="test")
        tracker.start()
        tracker.event("STT_PARTIAL", event_id="utt-1", text=long_text)
        await asyncio.sleep(0.15)
        # events list is populated in the same asyncio.run() context
        stt_events = [ev for ev in tracker.events if ev.get("event") == "STT_PARTIAL"]
        assert stt_events, "STT_PARTIAL event was not recorded in tracker.events"
        assert len(stt_events[0].get("text", "")) <= 63, "event text was not truncated"
        await tracker.stop()
    asyncio.run(_inner())


def test_tracker_stop_idempotent(debug_on):
    async def _inner():
        tracker = LatencyTracker("stop-sess", engine="test")
        tracker.start()
        await tracker.stop()
        await tracker.stop()  # must not raise
    asyncio.run(_inner())


def test_tracker_disabled_is_noop(monkeypatch):
    """When DEBUG_LATENCY=false, tracker is a full no-op."""
    monkeypatch.setattr(latency_module, "DEBUG_LATENCY", False)
    async def _inner():
        tracker = latency_module.LatencyTracker("off-sess", engine="test")
        tracker.start()
        assert tracker._task is None
        tracker.event("STT_PARTIAL", event_id="utt-1", text="hello")
        assert tracker._queue.empty()
        await tracker.stop()
    asyncio.run(_inner())


# ---------------------------------------------------------------------------
# TranslationWorker + tracker integration
# ---------------------------------------------------------------------------

def test_translation_worker_with_tracker_emits_events(monkeypatch):
    """TranslationWorker must emit TRANSLATION_STARTED and TRANSLATION_COMPLETED when tracker is set."""
    monkeypatch.setattr(latency_module, "DEBUG_LATENCY", True)
    async def _inner():
        tracker = LatencyTracker("tw-sess", engine="test")
        tracker.start()
        worker = TranslationWorker(_FakeTr(), ["en", "pt"], tracker=tracker)
        worker.start()
        worker.enqueue("hello world", "es", "utt-1")
        await worker.queue.join()
        # Give the drain loop time to process
        await asyncio.sleep(0.2)
        event_names = {ev["event"] for ev in tracker.events}
        assert "TRANSLATION_STARTED" in event_names, f"Got events: {event_names}"
        assert "TRANSLATION_COMPLETED" in event_names, f"Got events: {event_names}"
        await worker.stop()
        await tracker.stop()
    asyncio.run(_inner())


def test_translation_worker_without_tracker_still_works():
    async def _inner():
        worker = TranslationWorker(_FakeTr(), ["en"], tracker=None)
        worker.start()
        worker.enqueue("test text", "es")
        await worker.queue.join()
        assert "test text" in worker.stable_original
        await worker.stop()
    asyncio.run(_inner())


# ---------------------------------------------------------------------------
# Pipeline smoke: /ws still works when DEBUG_LATENCY=true
# ---------------------------------------------------------------------------

def test_ws_pipeline_with_debug_latency(monkeypatch, client):
    """The /ws pipeline must return valid translations with DEBUG_LATENCY=true."""
    monkeypatch.setattr(latency_module, "DEBUG_LATENCY", True)

    class _FakeTranslator:
        async def translate(self, text, src, dest):
            class R:
                pass
            r = R()
            r.text = f"{dest}:{text}"
            return r

    monkeypatch.setattr(websockets_route, "Translator", lambda: _FakeTranslator())

    # Use the same login flow as test_main.py
    client.post("/login", data={"password": "test-password", "next": "/"}, follow_redirects=False)

    with client.websocket_connect("/ws", headers={"origin": "http://testserver"}) as ws:
        ws.send_json({
            "type": "interim",
            "text": "hola mundo",
            "src": "es",
            "dests": ["en"],
        })
        data = ws.receive_json()
    assert data is not None
    assert data.get("error") is None or "error" not in data


def test_find_dropped_words():
    from app.services.latency_logger import find_dropped_words

    assert find_dropped_words("", "hello") == []
    assert find_dropped_words("hello", "") == ["hello"]
    assert find_dropped_words("we are testing system", "we are system") == ["testing"]
    assert find_dropped_words("hello world", "hello world extra") == []
    assert find_dropped_words("one two two three", "one two three") == ["two"]


def test_tracker_dropped_words_interim_and_final(debug_on):
    async def _inner():
        tracker = LatencyTracker("drop-test", engine="producer")
        tracker.start()
        
        # 1. Interim expansion
        tracker.event("STT_PARTIAL", event_id="utt-1", text="we are going to build")
        # 2. Interim retraction (engine drops "going to")
        tracker.event("STT_PARTIAL", event_id="utt-1", text="we are build")
        # 3. Final omission
        tracker.event("STT_FINAL", event_id="utt-1", text="we build")
        
        await asyncio.sleep(0.1)
        await tracker.stop()

        assert tracker.total_words_dropped >= 3
        assert tracker.dropped_events_count >= 2
    asyncio.run(_inner())


