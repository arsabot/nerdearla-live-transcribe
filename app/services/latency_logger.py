"""
Latency instrumentation for the real-time transcription/translation pipeline.

Activate with: DEBUG_LATENCY=true (env var)
Log file configured via: LATENCY_LOG_FILE (defaults to logs/latency.log)

Events emitted:
  AUDIO_RECEIVED, VAD_SPEECH_START, VAD_SPEECH_END,
  STT_REQUEST, STT_PARTIAL, STT_STABLE, STT_FINAL,
  STABILITY_EMITTED, TRANSLATION_QUEUED, TRANSLATION_STARTED,
  TRANSLATION_COMPLETED, TRANSLATION_FAILED, WEBSOCKET_SENT, UI_RENDERED, UI_DISCARDED
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Ensure environment variables from .env are loaded
load_dotenv()

# ---- Configuration --------------------------------------------------------

DEBUG_LATENCY: bool = os.getenv("DEBUG_LATENCY", "false").lower() in ("1", "true", "yes")
_LATENCY_LOG_FILE: str = os.getenv("LATENCY_LOG_FILE", "logs/latency.log")

# ---- File logger setup (separate from app logger) -------------------------

_latency_file_logger: Optional[logging.Logger] = None


def _get_file_logger() -> logging.Logger:
    global _latency_file_logger
    if _latency_file_logger is not None:
        return _latency_file_logger

    log_file = os.getenv("LATENCY_LOG_FILE", _LATENCY_LOG_FILE)
    logger = logging.getLogger("latency_trace")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't send to root logger

    if not logger.handlers:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)

        # Also echo to stderr so it appears in uvicorn console
        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(logging.Formatter("[LATENCY] %(message)s"))
        logger.addHandler(sh)

    _latency_file_logger = logger
    return logger


def _truncate(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _now_ms() -> float:
    return time.time() * 1000


def find_dropped_words(prev_text: Optional[str], curr_text: Optional[str]) -> list[str]:
    """
    Identifies words present in prev_text that are missing in curr_text.
    Useful for detecting retracted interim words or words omitted in final transcripts.
    """
    import re
    from collections import Counter

    if not prev_text or not prev_text.strip():
        return []

    def tokenize(t: str) -> list[str]:
        cleaned = re.sub(r"[^\w\s]", " ", t.lower())
        return [w.strip() for w in cleaned.split() if w.strip()]

    prev_words = tokenize(prev_text)
    if not prev_words:
        return []

    curr_words = tokenize(curr_text or "")
    prev_cnt = Counter(prev_words)
    curr_cnt = Counter(curr_words)

    dropped: list[str] = []
    for word, count in prev_cnt.items():
        diff = count - curr_cnt.get(word, 0)
        if diff > 0:
            dropped.extend([word] * diff)
    return dropped


# ---- LatencyTracker -------------------------------------------------------

class LatencyTracker:
    """
    Non-blocking per-connection latency and word-accuracy tracker.

    Usage:
        tracker = LatencyTracker(session_id="dg-abc123", engine="deepgram")
        tracker.start()
        tracker.event("STT_PARTIAL", event_id="utt-1", text="hello", text_len=5)
        ...
        await tracker.stop()   # prints summary
    """

    def __init__(self, session_id: str, engine: str = "unknown"):
        self.session_id = session_id
        self.engine = engine
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._session_start = _now_ms()

        # Metrics accumulators (updated inside the async loop — no locking needed)
        self._metrics: dict[str, list[float]] = defaultdict(list)
        self._event_ts: dict[str, dict[str, float]] = defaultdict(dict)  # event_id -> {event_name: ts}
        self._first_stt_logged: set[str] = set()  # utterance ids where STT_FIRST was captured

        self.max_queue_size: int = 0
        self.discarded_results: int = 0
        self.messages_received: int = 0
        self.messages_rendered: int = 0
        self.messages_discarded: int = 0
        self.max_pending_updates: int = 0
        self.errors: int = 0
        self.events: list[dict] = []  # chronological log of processed events (for testing)

        # Word tracking & dropped word analytics
        self.total_words_received: int = 0
        self.total_words_finalized: int = 0
        self.total_words_dropped: int = 0
        self.dropped_events_count: int = 0
        self._last_interim_text: dict[str, str] = {}
        self._highest_interim_text: dict[str, str] = {}

        # Audio presence tracking (for audio→STT metrics)
        self._last_audio_ms: Optional[float] = None
        self._utterance_start_ms: Optional[float] = None

    @property
    def is_enabled(self) -> bool:
        return bool(DEBUG_LATENCY)

    # ---- Public API -------------------------------------------------------

    def start(self) -> None:
        if not self.is_enabled:
            return
        self._running = True
        self._task = asyncio.create_task(self._drain_loop(), name=f"latency-{self.session_id}")

    def note_audio(self) -> None:
        """Call every time an audio chunk is received (does NOT log a line — too spammy)."""
        if not self.is_enabled:
            return
        now = _now_ms()
        # Treat a gap of >800 ms as the start of a new utterance
        if self._last_audio_ms is None or (now - self._last_audio_ms) > 800:
            self._utterance_start_ms = now
        self._last_audio_ms = now

    def note_audio_verbose(self, event_id: str, buf_ms: Optional[float] = None) -> None:
        """Like note_audio() but also emits an AUDIO_RECEIVED log line."""
        self.note_audio()
        self.event("AUDIO_RECEIVED", event_id=event_id, buf_ms=buf_ms)

    def event(self, name: str, *, event_id: str, revision_id: Optional[int] = None,
              text: Optional[str] = None, text_len: Optional[int] = None,
              buf_ms: Optional[float] = None, queue_size: Optional[int] = None,
              latency_ms: Optional[float] = None, error: Optional[str] = None,
              engine: Optional[str] = None, stt_ts: Optional[float] = None,
              ws_sent_ts: Optional[float] = None, ws_recv_ts: Optional[float] = None,
              render_ts: Optional[float] = None, **extra) -> None:
        """Emit a latency event. Thread-safe (uses call_soon_threadsafe if needed)."""
        if not self.is_enabled:
            return
        now = _now_ms()
        payload: dict = {
            "ts": now,
            "session": self.session_id,
            "event_id": event_id,
            "event": name,
            "engine": engine or self.engine,
        }
        if revision_id is not None:
            payload["revision_id"] = revision_id
        if text is not None:
            payload["text"] = _truncate(text)
            payload["text_len"] = len(text)
        if text_len is not None and "text_len" not in payload:
            payload["text_len"] = text_len
        if buf_ms is not None:
            payload["buf_ms"] = round(buf_ms, 1)
        if queue_size is not None:
            payload["queue_size"] = queue_size
        if latency_ms is not None:
            payload["latency_ms"] = round(latency_ms, 1)
        if error is not None:
            payload["error"] = error
        if stt_ts is not None:
            payload["stt_ts"] = stt_ts
        if ws_sent_ts is not None:
            payload["ws_sent_ts"] = ws_sent_ts
        if ws_recv_ts is not None:
            payload["ws_recv_ts"] = ws_recv_ts
        if render_ts is not None:
            payload["render_ts"] = render_ts
        payload.update(extra)

        try:
            self._queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # never block the pipeline

    async def stop(self) -> None:
        if not self.is_enabled:
            return
        self._running = False
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        self._print_summary()

    # ---- Internal async drain loop ----------------------------------------

    async def _drain_loop(self) -> None:
        logger = _get_file_logger()
        while self._running or not self._queue.empty():
            try:
                payload = await asyncio.wait_for(self._queue.get(), timeout=0.3)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                self._process_payload(payload)
                self.events.append(payload)
                logger.info(json.dumps(payload, ensure_ascii=False))
            except Exception as exc:
                logging.debug("LatencyTracker drain error: %s", exc)
            finally:
                try:
                    self._queue.task_done()
                except ValueError:
                    pass

        # Drain anything that arrived after _running was set False
        while not self._queue.empty():
            try:
                payload = self._queue.get_nowait()
                self._process_payload(payload)
                _get_file_logger().info(json.dumps(payload, ensure_ascii=False))
            except asyncio.QueueEmpty:
                break
            except Exception:
                break

    def _process_payload(self, payload: dict) -> None:
        name = payload["event"]
        eid = payload["event_id"]
        ts = payload["ts"]

        # Store per-event timestamps for delta calculations
        self._event_ts[eid][name] = ts
        if "stt_ts" in payload:
            self._event_ts[eid]["stt_ts"] = payload["stt_ts"]

        # ---- Metric deltas for the 7 stages -------------------------------

        # 1. STT_PARTIAL → STABILITY_EMITTED
        if name == "STABILITY_EMITTED":
            t_stt = self._event_ts[eid].get("STT_PARTIAL") or self._event_ts[eid].get("STT_REQUEST") or payload.get("stt_ts")
            if t_stt and ts >= t_stt:
                self._metrics["1_stt_to_stability_ms"].append(ts - t_stt)

        # 2. STABILITY_EMITTED → TRANSLATION_QUEUED
        if name == "TRANSLATION_QUEUED":
            t_stab = self._event_ts[eid].get("STABILITY_EMITTED") or self._event_ts[eid].get("STT_STABLE")
            if t_stab and ts >= t_stab:
                self._metrics["2_stability_to_queued_ms"].append(ts - t_stab)
            else:
                self._metrics["2_stability_to_queued_ms"].append(0.0)

        # 3. TRANSLATION_QUEUED → TRANSLATION_STARTED
        if name == "TRANSLATION_STARTED":
            t_q = self._event_ts[eid].get("TRANSLATION_QUEUED")
            if t_q and ts >= t_q:
                self._metrics["3_queued_to_started_ms"].append(ts - t_q)

        # 4. TRANSLATION_STARTED → TRANSLATION_COMPLETED
        if name == "TRANSLATION_COMPLETED":
            t_start = self._event_ts[eid].get("TRANSLATION_STARTED")
            if t_start and ts >= t_start:
                self._metrics["4_started_to_completed_ms"].append(ts - t_start)
            elif payload.get("latency_ms") is not None:
                self._metrics["4_started_to_completed_ms"].append(payload["latency_ms"])

        # 5. TRANSLATION_COMPLETED → WEBSOCKET_SENT
        if name == "WEBSOCKET_SENT":
            self.messages_received += 1
            t_comp = self._event_ts[eid].get("TRANSLATION_COMPLETED")
            if t_comp and ts >= t_comp:
                self._metrics["5_completed_to_sent_ms"].append(ts - t_comp)

        # 6. WEBSOCKET_SENT → UI_RENDERED & 7. STT_PARTIAL → UI_RENDERED
        if name == "UI_RENDERED":
            self.messages_rendered += 1
            t_sent = self._event_ts[eid].get("WEBSOCKET_SENT") or payload.get("ws_sent_ts")
            if t_sent and ts >= t_sent:
                self._metrics["6_sent_to_rendered_ms"].append(ts - t_sent)
            elif payload.get("ws_to_render_ms") is not None:
                self._metrics["6_sent_to_rendered_ms"].append(payload["ws_to_render_ms"])

            # 7. Total STT → Rendered
            t_stt = self._event_ts[eid].get("STT_PARTIAL") or self._event_ts[eid].get("stt_ts") or payload.get("stt_ts")
            if payload.get("latency_total_ms") is not None:
                self._metrics["7_stt_to_rendered_ms"].append(payload["latency_total_ms"])
            elif t_stt and ts >= t_stt:
                self._metrics["7_stt_to_rendered_ms"].append(ts - t_stt)

            # Frontend specific metrics if provided by telemetry
            if payload.get("stt_to_ws_ms") is not None:
                self._metrics["fe_stt_to_ws"].append(payload["stt_to_ws_ms"])
            if payload.get("ws_receive_delay_ms") is not None:
                self._metrics["fe_ws_receive_delay"].append(payload["ws_receive_delay_ms"])
            if payload.get("ws_to_render_ms") is not None:
                self._metrics["fe_ws_to_render"].append(payload["ws_to_render_ms"])
            if payload.get("latency_total_ms") is not None:
                self._metrics["fe_stt_to_render"].append(payload["latency_total_ms"])

        # Discards
        if name == "UI_DISCARDED":
            self.discarded_results += 1
            self.messages_discarded += 1

        if name == "UI_SUMMARY":
            if payload.get("max_pending_updates") is not None:
                self.max_pending_updates = max(self.max_pending_updates, int(payload["max_pending_updates"]))
            if payload.get("messages_received") is not None:
                self.messages_received = int(payload["messages_received"])
            if payload.get("messages_rendered") is not None:
                self.messages_rendered = int(payload["messages_rendered"])
            if payload.get("messages_discarded") is not None:
                self.messages_discarded = int(payload["messages_discarded"])

        # Word tracking & dropped words detection
        if name == "WORDS_DROPPED":
            dropped_cnt = int(payload.get("dropped_count", 0))
            self.total_words_dropped += dropped_cnt
            self.dropped_events_count += 1

        raw_text = payload.get("text")
        if raw_text and name == "STT_PARTIAL":
            curr_words = raw_text.split()
            self.total_words_received += len(curr_words)
            prev_interim = self._last_interim_text.get(eid, "")
            
            if prev_interim:
                dropped = find_dropped_words(prev_interim, raw_text)
                if dropped and len(curr_words) < len(prev_interim.split()):
                    self.total_words_dropped += len(dropped)
                    self.dropped_events_count += 1
                    drop_payload = {
                        "ts": ts,
                        "session": self.session_id,
                        "event_id": eid,
                        "event": "WORDS_DROPPED",
                        "engine": payload.get("engine", self.engine),
                        "dropped_count": len(dropped),
                        "dropped_words": dropped,
                        "prev_word_count": len(prev_interim.split()),
                        "curr_word_count": len(curr_words),
                        "prev_text": prev_interim,
                        "curr_text": raw_text,
                        "context": "interim_retraction",
                    }
                    _get_file_logger().info(json.dumps(drop_payload, ensure_ascii=False))

            self._last_interim_text[eid] = raw_text
            if len(curr_words) > len(self._highest_interim_text.get(eid, "").split()):
                self._highest_interim_text[eid] = raw_text

        elif raw_text and name == "STT_FINAL":
            curr_words = raw_text.split()
            self.total_words_finalized += len(curr_words)
            highest_interim = self._highest_interim_text.get(eid, "") or self._last_interim_text.get(eid, "")
            
            if highest_interim:
                dropped = find_dropped_words(highest_interim, raw_text)
                if dropped and len(curr_words) < len(highest_interim.split()):
                    self.total_words_dropped += len(dropped)
                    self.dropped_events_count += 1
                    drop_payload = {
                        "ts": ts,
                        "session": self.session_id,
                        "event_id": eid,
                        "event": "WORDS_DROPPED",
                        "engine": payload.get("engine", self.engine),
                        "dropped_count": len(dropped),
                        "dropped_words": dropped,
                        "prev_word_count": len(highest_interim.split()),
                        "curr_word_count": len(curr_words),
                        "prev_text": highest_interim,
                        "curr_text": raw_text,
                        "context": "final_omission",
                    }
                    _get_file_logger().info(json.dumps(drop_payload, ensure_ascii=False))

            self._last_interim_text.pop(eid, None)
            self._highest_interim_text.pop(eid, None)

        # Audio tracking
        utterance_start = self._utterance_start_ms
        if name in ("STT_PARTIAL", "STT_STABLE", "STT_FINAL") and eid not in self._first_stt_logged:
            self._first_stt_logged.add(eid)
            if utterance_start and ts >= utterance_start:
                self._metrics["stt_first_response_ms"].append(ts - utterance_start)

        if name == "UI_RENDERED" and utterance_start and ts >= utterance_start:
            self._metrics["end_to_end_ms"].append(ts - utterance_start)

        # Queue size high-water mark
        qs = payload.get("queue_size")
        if qs is not None and qs > self.max_queue_size:
            self.max_queue_size = qs

        # Errors
        if payload.get("error"):
            self.errors += 1

    # ---- Summary ----------------------------------------------------------

    def _print_summary(self) -> None:
        def avg(k: str) -> float:
            vals = self._metrics.get(k, [])
            return sum(vals) / len(vals) if vals else 0.0

        def mx(k: str) -> float:
            vals = self._metrics.get(k, [])
            return max(vals) if vals else 0.0

        duration_s = (_now_ms() - self._session_start) / 1000

        # Fallback values for frontend stats if only server-side metrics exist
        fe_stt_to_ws_avg = avg("fe_stt_to_ws") or avg("5_completed_to_sent_ms")
        fe_stt_to_ws_max = mx("fe_stt_to_ws") or mx("5_completed_to_sent_ms")
        fe_ws_recv_delay_avg = avg("fe_ws_receive_delay")
        fe_ws_to_render_avg = avg("fe_ws_to_render") or avg("6_sent_to_rendered_ms")
        fe_stt_to_render_avg = avg("fe_stt_to_render") or avg("7_stt_to_rendered_ms")

        total_analyzed_words = self.total_words_finalized + self.total_words_dropped
        drop_rate = (self.total_words_dropped / max(1, total_analyzed_words)) * 100

        lines = [
            f"\n=== PIPELINE STAGE BREAKDOWN ({self.session_id} / {self.engine}) ===",
            f"session_duration:                   {duration_s:.2f}s",
            f"1. STT_PARTIAL → STABILITY_EMITTED: {avg('1_stt_to_stability_ms'):.1f}ms",
            f"2. STABILITY → TRANSLATION_QUEUED:  {avg('2_stability_to_queued_ms'):.1f}ms",
            f"3. TRANSLATION_QUEUED → STARTED:    {avg('3_queued_to_started_ms'):.1f}ms",
            f"4. TRANSLATION_STARTED → COMPLETED: {avg('4_started_to_completed_ms'):.1f}ms",
            f"5. TRANSLATION_COMPLETED → WS_SENT: {avg('5_completed_to_sent_ms'):.1f}ms",
            f"6. WEBSOCKET_SENT → UI_RENDERED:    {avg('6_sent_to_rendered_ms'):.1f}ms",
            f"7. STT_PARTIAL → UI_RENDERED:       {avg('7_stt_to_rendered_ms'):.1f}ms (max: {mx('7_stt_to_rendered_ms'):.1f}ms)",
            f"max_queue_size:                     {self.max_queue_size}",
            f"errors:                             {self.errors}",
            f"",
            f"=== WORD ACCURACY & DROPPED WORDS ===",
            f"words_received_interim: {self.total_words_received}",
            f"words_finalized:        {self.total_words_finalized}",
            f"words_dropped_total:    {self.total_words_dropped}",
            f"drop_incidents:         {self.dropped_events_count}",
            f"drop_rate:              {drop_rate:.1f}%",
            f"",
            f"=== FRONTEND LATENCY ===",
            f"stt_to_ws_avg:        {fe_stt_to_ws_avg:.1f} ms",
            f"stt_to_ws_max:        {fe_stt_to_ws_max:.1f} ms",
            f"ws_receive_delay_avg: {fe_ws_recv_delay_avg:.1f} ms",
            f"ws_to_render_avg:     {fe_ws_to_render_avg:.1f} ms",
            f"stt_to_render_avg:    {fe_stt_to_render_avg:.1f} ms",
            f"messages_received:    {self.messages_received}",
            f"messages_rendered:    {self.messages_rendered}",
            f"messages_discarded:   {self.messages_discarded}",
            f"max_pending_updates:  {self.max_pending_updates}",
            f"========================",
        ]

        # Diagnostic bottleneck detection
        lines.append("\n--- DIAGNOSTIC (suspected bottlenecks) ---")
        bottlenecks = []
        if avg("1_stt_to_stability_ms") > 500:
            bottlenecks.append("STABILITY_DELAY — StabilityEngine is waiting too long before confirming stable words.")
        if avg("3_queued_to_started_ms") > 300:
            bottlenecks.append("QUEUE_BACKLOG — TranslationWorker queue is backing up.")
        if avg("4_started_to_completed_ms") > 1200:
            bottlenecks.append("TRANSLATION_SLOW — Translation provider takes >1.2s per chunk.")
        if fe_ws_to_render_avg > 150:
            bottlenecks.append("FRONTEND_RENDER_DELAY — Browser takes >150ms between WS message arrival and DOM render.")
        if self.messages_discarded > 5:
            bottlenecks.append(f"OUTDATED_REVISIONS — {self.messages_discarded} outdated revisions were discarded by the frontend.")
        if self.total_words_dropped > 5:
            bottlenecks.append(f"STT_WORD_DROPS — {self.total_words_dropped} words were dropped or retracted ({self.dropped_events_count} incidents).")
        if not bottlenecks:
            bottlenecks.append("No obvious bottleneck detected from available data.")
        lines += [f"  - {b}" for b in bottlenecks]

        lines.append("=" * 52)

        summary = "\n".join(lines)
        _get_file_logger().info(summary)
        logging.info(summary)
