import contextlib

from app.core import config
import asyncio
import os
import json
import logging
import time
import threading
import base64
import websockets as ws_lib
from fastapi import APIRouter, WebSocket
from starlette.websockets import WebSocketDisconnect

from app.core.dependencies import _require_ws_auth, _require_staff_ws_auth
from app.core.config import _normalize_lang_code, _normalize_translate_dests
from app.session_manager import SessionManager, TranscriptEvent, LatencyMetrics
from app.translator import Translator
from app.translation_provider import get_translation_provider, normalize_brand_terms, GoogleTranslationProvider
from app.services.utils import _looks_like_deepgram_results, _deepgram_send_finalize, _deepgram_send_close_stream, _translate
from app.services.stability import StabilityEngine
from app.services.semantic_chunker import SemanticChunker
from app.services.translation_worker import TranslationWorker
from app.services.latency_logger import LatencyTracker

from app.gemini_live import global_gemini_live_circuit, redact_secrets, ErrorType, LatencyTelemetry, build_system_instruction

try:
    from deepgram import DeepgramClient
except Exception:
    DeepgramClient = None

try:
    from deepgram.core.events import EventType
except Exception:
    class EventType:
        MESSAGE = "message"
        ERROR = "error"
        CLOSE = "close"

router = APIRouter()
session_manager = SessionManager.get_instance()

@router.websocket("/ws/session/{session_id}/viewer")
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


@router.websocket("/ws/session/{session_id}/producer")
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

    tracker = LatencyTracker(session_id=session_id, engine="producer")
    tracker.start()

    active_translation_task: Optional[asyncio.Task] = None
    last_translations_by_event: dict[str, dict[str, str]] = {}

    async def _run_translation_and_broadcast(
        msg_type: str,
        current_text: str,
        source_lang: str,
        target_langs: list[str],
        glossary: dict | None,
        event_id: str,
        audio_ts: float | None,
        client_sent_ms: float | None,
        current_session_speaker: str | None,
        current_session_provider: str | None,
    ):
        tracker.event("TRANSLATION_STARTED", event_id=event_id, text=current_text)
        start_t = time.perf_counter()
        if msg_type == "interim" and (current_session_provider or "").startswith("gemini"):
            prov = get_translation_provider("googletrans")
        else:
            prov = get_translation_provider(current_session_provider)

        try:
            translations = await prov.translate_batch(
                text=current_text,
                source=source_lang,
                targets=target_langs,
                glossary=glossary,
            )
            translations = {k: normalize_brand_terms(v) for k, v in translations.items()}
        except asyncio.CancelledError:
            return  # newer interim superseded this translation
        except Exception as tr_err:
            logging.error("Translation batch error for session %s: %s", session_id, tr_err)
            tracker.event("TRANSLATION_FAILED", event_id=event_id, error=str(tr_err))
            translations = {t: (current_text if t == source_lang else "") for t in target_langs}

        trans_ms = (time.perf_counter() - start_t) * 1000
        tracker.event(
            "TRANSLATION_COMPLETED",
            event_id=event_id,
            latency_ms=trans_ms,
            text=str(translations.get("es", "") or next(iter(translations.values()), "")),
        )

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
            original=current_text,
            translations=translations,
            speaker=current_session_speaker,
            metrics=metrics,
        )

        await session_manager.broadcast_event(session_id, event)
        tracker.event("WEBSOCKET_SENT", event_id=event_id)
        
        if msg_type == "interim":
            last_translations_by_event[event_id] = translations
        else:
            last_translations_by_event.pop(event_id, None)
            last_translations_by_event.pop(f"{event_id}-split", None)

    chunker = SemanticChunker(max_words=18, soft_limit_words=12, min_prefix_words=6)

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
            event_id = (data.get("event_id") if isinstance(data, dict) else None) or f"prod-{int(time.time()*1000)}"

            tracker.event(
                "STT_PARTIAL" if msg_type == "interim" else "STT_FINAL",
                event_id=event_id,
                text=text,
                text_len=len(text),
                engine="producer",
            )

            try:
                current_session = await session_manager.get_session(session_id) or session
                source_lang = (data.get("source_language") if isinstance(data, dict) else None) or current_session.source_language or "auto"
                target_langs = list(dict.fromkeys((current_session.target_languages or ["es"]) + ["es", "en", "pt"]))

                if msg_type == "interim":
                    split_chunk, remaining_text = chunker.process_interim(text, lang=source_lang)

                    # If an auto-split occurred at a semantic boundary, commit the prefix as final
                    if split_chunk:
                        if active_translation_task and not active_translation_task.done():
                            active_translation_task.cancel()
                        await _run_translation_and_broadcast(
                            msg_type="final",
                            current_text=split_chunk,
                            source_lang=source_lang,
                            target_langs=target_langs,
                            glossary=current_session.glossary,
                            event_id=f"{event_id}-split",
                            audio_ts=audio_ts,
                            client_sent_ms=client_sent_ms,
                            current_session_speaker=current_session.speaker,
                            current_session_provider=current_session.translation_provider,
                        )

                    # Now broadcast and translate the remaining in-flight interim
                    if remaining_text.strip():
                        immediate_metrics = LatencyMetrics(
                            audio_ms=audio_ts if audio_ts is not None else 0.0,
                            stt_ms=50.0,
                            translation_ms=0.0,
                            delivery_ms=10.0,
                            total_ms=60.0,
                        )
                        # Prevent UI fallback to source language by using a truthy space (" ")
                        # if there is no previous translation for this interim event.
                        last_trans = last_translations_by_event.get(event_id, {})
                        imm_translations = {
                            t: (remaining_text if t == source_lang else last_trans.get(t, " "))
                            for t in target_langs
                        }
                        
                        immediate_event = TranscriptEvent(
                            session_id=session_id,
                            type="interim",
                            timestamp=time.time(),
                            source_language=source_lang,
                            original=remaining_text,
                            translations=imm_translations,
                            speaker=current_session.speaker,
                            metrics=immediate_metrics,
                        )
                        await session_manager.broadcast_event(session_id, immediate_event)

                        if active_translation_task and not active_translation_task.done():
                            active_translation_task.cancel()
                        active_translation_task = asyncio.create_task(
                            _run_translation_and_broadcast(
                                msg_type="interim",
                                current_text=remaining_text,
                                source_lang=source_lang,
                                target_langs=target_langs,
                                glossary=current_session.glossary,
                                event_id=event_id,
                                audio_ts=audio_ts,
                                client_sent_ms=client_sent_ms,
                                current_session_speaker=current_session.speaker,
                                current_session_provider=current_session.translation_provider,
                            )
                        )
                else:
                    # Final statement: commit any remaining uncommitted text and reset chunker
                    final_uncommitted = chunker.process_final(text)
                    if final_uncommitted.strip():
                        if active_translation_task and not active_translation_task.done():
                            active_translation_task.cancel()
                        await _run_translation_and_broadcast(
                            msg_type="final",
                            current_text=final_uncommitted,
                            source_lang=source_lang,
                            target_langs=target_langs,
                            glossary=current_session.glossary,
                            event_id=event_id,
                            audio_ts=audio_ts,
                            client_sent_ms=client_sent_ms,
                            current_session_speaker=current_session.speaker,
                            current_session_provider=current_session.translation_provider,
                        )
            except Exception as e:
                logging.error("Error processing producer message in session %s: %s", session_id, e)

    except WebSocketDisconnect:
        logging.info("Producer WebSocket disconnected from session %s", session_id)
    except Exception as e:
        logging.error("Producer WS error: %s", e)
    finally:
        if active_translation_task and not active_translation_task.done():
            active_translation_task.cancel()
        await session_manager.remove_producer(session_id, websocket)
        await tracker.stop()



@router.websocket("/ws")
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

    revision_id = 0
    stability = StabilityEngine()

    _ws_session_id = f"ws-{int(time.time()*1000)}"
    _tracker = LatencyTracker(session_id=_ws_session_id, engine="webspeech")
    _tracker.start()

    worker = TranslationWorker(translator, session_dest_langs, tracker=_tracker)
    worker.start()

    try:
        while True:
            # Čekáme na text z frontendu
            raw = await websocket.receive_text()
            if not raw:
                continue

            wants_typed_response = False
            msg_type: str | None = None
            src_lang = session_src_lang
            dest_langs: list[str] = list(session_dest_langs)
            text = raw
            client_id: int | None = None
            client_sent_ms: float | None = None
            _client_event_id: str | None = None
            _client_rev_id: int | None = None
            _client_stt_ts: float | None = None

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
                                worker.set_dest_langs(dests_norm)
                        continue

                    if parsed.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue

                    # Latency telemetry from browser (UI_RENDERED / UI_DISCARDED / UI_SUMMARY)
                    if parsed.get("type") in ("ui_rendered", "ui_discarded", "ui_summary"):
                        _evt_type = parsed["type"].upper()
                        _r_id = parsed.get("revision_id")
                        _eid = parsed.get("event_id") or (f"utt-{_r_id}" if _r_id is not None else "unknown")
                        _tracker.event(
                            _evt_type,
                            event_id=_eid,
                            revision_id=_r_id,
                            text=parsed.get("text"),
                            text_len=parsed.get("text_len"),
                            stt_ts=parsed.get("stt_ts"),
                            ws_sent_ts=parsed.get("ws_sent_ts"),
                            ws_recv_ts=parsed.get("ws_recv_ts"),
                            render_ts=parsed.get("render_ts"),
                            latency_ms=parsed.get("latency_total_ms") or parsed.get("latency_ms"),
                            stt_to_ws_ms=parsed.get("stt_to_ws_ms"),
                            ws_receive_delay_ms=parsed.get("ws_receive_delay_ms"),
                            ws_to_render_ms=parsed.get("ws_to_render_ms"),
                            max_pending_updates=parsed.get("max_pending_updates"),
                            messages_received=parsed.get("messages_received"),
                            messages_rendered=parsed.get("messages_rendered"),
                            messages_discarded=parsed.get("messages_discarded"),
                        )
                        continue

                    if isinstance(parsed.get("text"), str):
                        wants_typed_response = True
                        msg_type = parsed.get("type")
                        text = parsed["text"]

                        if parsed.get("event_id"):
                            _client_event_id = str(parsed["event_id"])
                        if isinstance(parsed.get("revision_id"), int):
                            _client_rev_id = parsed["revision_id"]
                        if isinstance(parsed.get("stt_ts"), (int, float)):
                            _client_stt_ts = float(parsed["stt_ts"])

                        cid = parsed.get("client_id")
                        if isinstance(cid, int) and cid >= 0:
                            client_id = cid
                        cts = parsed.get("client_sent_ms")
                        if isinstance(cts, (int, float)):
                            client_sent_ms = float(cts)
                            if _client_stt_ts is None:
                                _client_stt_ts = float(cts)

                        src_val = parsed.get("src")
                        src_norm = _normalize_lang_code(src_val)
                        if src_norm:
                            src_lang = src_norm

                        dests_norm = _normalize_translate_dests(parsed.get("dests"))
                        if dests_norm:
                            dest_langs = dests_norm
                            worker.set_dest_langs(dests_norm)
            except Exception:
                # Legacy klient posílá prostý text.
                dest_langs = ["en", "ru"]
                src_lang = "cs"
                worker.set_dest_langs(dest_langs)

            text = normalize_brand_terms(text.strip())

            if len(dest_langs) == 2 and dest_langs[0] == dest_langs[1]:
                dest_langs[1] = "ru" if dest_langs[0] != "ru" else "en"
                worker.set_dest_langs(dest_langs)

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
                revision_id: int | None = None,
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
                if revision_id is not None:
                    payload["revision_id"] = revision_id
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
                
            if len(text) > config.MAX_TEXT_LENGTH:
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

            is_interim = msg_type == "interim"
            _evt_id = _client_event_id or f"utt-{revision_id+1}"
            _stt_ts = _client_stt_ts or client_sent_ms or (time.time() * 1000)
            _tracker.event("STT_REQUEST", event_id=_evt_id, text=text, revision_id=revision_id+1, stt_ts=_stt_ts)

            try:
                if not is_interim:
                    # Final text
                    _tracker.event("STT_FINAL", event_id=_evt_id, text=text, revision_id=revision_id+1, stt_ts=_stt_ts)
                    unemitted = stability.get_unemitted(text)
                    if unemitted.strip():
                        _tracker.event("STABILITY_EMITTED", event_id=_evt_id, revision_id=revision_id+1, text=unemitted, stt_ts=_stt_ts, queue_size=worker.queue.qsize())
                        _tracker.event("TRANSLATION_QUEUED", event_id=_evt_id, revision_id=revision_id+1, text=unemitted, stt_ts=_stt_ts, queue_size=worker.queue.qsize())
                        worker.enqueue(unemitted, src_lang, event_id=_evt_id, revision_id=revision_id+1, stt_ts=_stt_ts)
                    
                    # Wait for translation to complete
                    await worker.queue.join()
                    
                    revision_id += 1
                    now_ws_sent = time.time() * 1000
                    if wants_typed_response:
                        response = _typed_payload(
                            original=worker.stable_original,
                            translations={
                                dest: normalize_brand_terms(worker.stable_translations.get(dest, ""))
                                for dest in dest_langs
                            },
                            timing={"translate_ms": 0},
                            revision_id=revision_id,
                        )
                        response["event_id"] = _evt_id
                        response["stt_ts"] = _stt_ts
                        response["ws_sent_ts"] = now_ws_sent
                    else:
                        response = _legacy_payload(
                            original=worker.stable_original,
                            en=normalize_brand_terms(worker.stable_translations.get("en", "")),
                            ru=normalize_brand_terms(worker.stable_translations.get("ru", "")),
                        )
                    _tracker.event("WEBSOCKET_SENT", event_id=_evt_id, revision_id=revision_id, stt_ts=_stt_ts, ws_sent_ts=now_ws_sent)
                    await websocket.send_json(response)
                    stability.reset()
                    worker.reset()
                else:
                    # Interim
                    _tracker.event("STT_PARTIAL", event_id=_evt_id, text=text, revision_id=revision_id+1, stt_ts=_stt_ts)
                    new_stable = stability.update(text)
                    if new_stable:
                        _tracker.event("STABILITY_EMITTED", event_id=_evt_id, revision_id=revision_id+1, text=new_stable, stt_ts=_stt_ts, queue_size=worker.queue.qsize())
                        _tracker.event("TRANSLATION_QUEUED", event_id=_evt_id, revision_id=revision_id+1, text=new_stable, stt_ts=_stt_ts, queue_size=worker.queue.qsize())
                        worker.enqueue(new_stable, src_lang, event_id=_evt_id, revision_id=revision_id+1, stt_ts=_stt_ts)
                        
                    unemitted = stability.get_unemitted(text)
                    unemitted_tr = {d: "" for d in dest_langs}
                    if unemitted.strip():
                        try:
                            # Translate unemitted quickly
                            results = await asyncio.gather(*[
                                _translate(translator, unemitted, src=src_lang, dest=d)
                                for d in dest_langs
                            ])
                            unemitted_tr = {d: (r.text if r else "") for d, r in zip(dest_langs, results)}
                        except Exception:
                            pass
                            
                    current_translations = {}
                    for d in dest_langs:
                        curr = worker.stable_translations.get(d, "")
                        if curr and unemitted_tr[d]:
                            curr += " "
                        curr += unemitted_tr[d]
                        current_translations[d] = curr
                        
                    original_full = worker.stable_original
                    if original_full and unemitted.strip():
                        original_full += " "
                    original_full += unemitted
                    
                    revision_id += 1
                    now_ws_sent = time.time() * 1000
                    if wants_typed_response:
                        response = _typed_payload(
                            original=original_full,
                            translations={
                                dest: normalize_brand_terms(current_translations.get(dest, ""))
                                for dest in dest_langs
                            },
                            timing={"translate_ms": 0},
                            revision_id=revision_id,
                        )
                        response["event_id"] = _evt_id
                        response["stt_ts"] = _stt_ts
                        response["ws_sent_ts"] = now_ws_sent
                    else:
                        response = _legacy_payload(
                            original=original_full,
                            en=normalize_brand_terms(current_translations.get("en", "")),
                            ru=normalize_brand_terms(current_translations.get("ru", "")),
                        )
                    _tracker.event("WEBSOCKET_SENT", event_id=_evt_id, revision_id=revision_id, stt_ts=_stt_ts, ws_sent_ts=now_ws_sent)
                    await websocket.send_json(response)
            except Exception as e:
                logging.error(f"Překlad selhal: {str(e)}")
                translator = Translator()
                if wants_typed_response:
                    await websocket.send_json(_typed_payload(
                        original=text,
                        translations={d: "" for d in dest_langs},
                        error="translation_failed",
                        timing={"translate_ms": 0},
                    ))
                else:
                    await websocket.send_json(_legacy_payload(
                        original=text,
                        en="",
                        ru="",
                        error="translation_failed",
                    ))

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
    finally:
        await worker.stop()
        await _tracker.stop()


@router.websocket("/ws/deepgram")
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
    
    if not config.DEEPGRAM_API_KEY:
        logging.error("config.DEEPGRAM_API_KEY není nastaven")
        await websocket.send_json({"error": "config.DEEPGRAM_API_KEY not configured"})
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

    _dg_session_id = f"dg-{int(time.time()*1000)}"
    _tracker = LatencyTracker(session_id=_dg_session_id, engine="deepgram")
    _tracker.start()

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
        revision_id: int | None = None,
    ) -> dict:
        payload: dict = {
            "type": msg_type,
            "original": original,
            "dests": translate_dests,
            "translations": translations,
        }
        if revision_id is not None:
            payload["revision_id"] = revision_id
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
        deepgram = DeepgramClient(api_key=config.DEEPGRAM_API_KEY)

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
                "endpointing": "200",
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
                maxsize=config.DEEPGRAM_RESULT_QUEUE_SIZE
            )

            # Grace window to drain final results after shutdown.
            shutdown_deadline: float | None = None
            
            def on_message(*args, **kwargs):
                if len(args) == 1:
                    result = args[0]
                elif len(args) >= 2:
                    result = args[1]
                else:
                    return
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
            
            def on_error(*args, **kwargs):
                if len(args) == 1:
                    error = args[0]
                elif len(args) >= 2:
                    error = args[1]
                else:
                    return
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
            
            def on_close(*args, **kwargs):
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
            
            revision_id = 0
            worker = None
            
            # Coroutine pro zpracování výsledků
            async def process_results():
                nonlocal revision_id, worker
                stability = StabilityEngine()
                worker = TranslationWorker(translator, translate_dests, tracker=_tracker)
                worker.start()
                _utt_counter = 0
                
                try:
                    while True:
                        if stop_event.is_set() and result_queue.empty():
                            if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                                break
                            if listen_thread is None or not listen_thread.is_alive():
                                break
                        try:
                            result = await asyncio.wait_for(result_queue.get(), timeout=0.1)
                            transcript = result["transcript"]
                            is_final = result["is_final"]
                            _utt_counter += 1
                            _evt_id = f"dg-utt-{_utt_counter}"
                            
                            if is_final:
                                _tracker.event("STT_FINAL", event_id=_evt_id, text=transcript)
                                unemitted = stability.get_unemitted(transcript)
                                if unemitted.strip():
                                    _tracker.event("STABILITY_EMITTED", event_id=_evt_id, text=unemitted, queue_size=worker.queue.qsize())
                                    _tracker.event("TRANSLATION_QUEUED", event_id=_evt_id, text=unemitted, queue_size=worker.queue.qsize())
                                    worker.enqueue(unemitted, translate_src, _evt_id)
                                
                                # Process remaining items quickly
                                await worker.queue.join()
                                
                                revision_id += 1
                                response = _dg_payload(
                                    msg_type="final",
                                    original=worker.stable_original,
                                    translations=worker.stable_translations,
                                    timing={"translate_ms": 0},
                                    revision_id=revision_id,
                                )
                                try:
                                    _tracker.event("WEBSOCKET_SENT", event_id=_evt_id, revision_id=revision_id)
                                    await websocket.send_json(response)
                                except Exception as send_err:
                                    logging.error(f"Cannot send final deepgram ws msg: {send_err}")
                                
                                stability.reset()
                                worker.reset()
                            else:
                                _tracker.event("STT_PARTIAL", event_id=_evt_id, text=transcript)
                                new_stable = stability.update(transcript)
                                if new_stable:
                                    _tracker.event("STT_STABLE", event_id=_evt_id, text=new_stable)
                                    _tracker.event("STABILITY_EMITTED", event_id=_evt_id, text=new_stable, queue_size=worker.queue.qsize())
                                    _tracker.event("TRANSLATION_QUEUED", event_id=_evt_id, text=new_stable, queue_size=worker.queue.qsize())
                                    worker.enqueue(new_stable, translate_src, _evt_id)
                                    
                                if translate_interim:
                                    unemitted = stability.get_unemitted(transcript)
                                    unemitted_tr = {d: "" for d in translate_dests}
                                    if unemitted.strip():
                                        try:
                                            # Quick inline translation for just the unemitted tip
                                            results = await asyncio.gather(*[
                                                _translate(translator, unemitted, src=translate_src, dest=d)
                                                for d in translate_dests
                                            ])
                                            unemitted_tr = {d: (r.text if r else "") for d, r in zip(translate_dests, results)}
                                        except Exception:
                                            pass
                                            
                                    current_translations = {}
                                    for d in translate_dests:
                                        curr = worker.stable_translations.get(d, "")
                                        if curr and unemitted_tr[d]:
                                            curr += " "
                                        curr += unemitted_tr[d]
                                        current_translations[d] = curr
                                        
                                    original_full = worker.stable_original
                                    if original_full and unemitted.strip():
                                        original_full += " "
                                    original_full += unemitted
                                    
                                    revision_id += 1
                                    response = _dg_payload(
                                        msg_type="interim",
                                        original=original_full,
                                        translations=current_translations,
                                        timing={"translate_ms": 0},
                                        revision_id=revision_id,
                                    )
                                    try:
                                        _tracker.event("WEBSOCKET_SENT", event_id=_evt_id, revision_id=revision_id)
                                        await websocket.send_json(response)
                                    except Exception as send_err:
                                        logging.error(f"Cannot send interim deepgram ws msg: {send_err}")
                                else:
                                    # Just send the text if translation is disabled
                                    revision_id += 1
                                    response = _dg_payload(
                                        msg_type="interim",
                                        original=transcript,
                                        translations={d: "" for d in translate_dests},
                                        timing={"translate_ms": 0},
                                        revision_id=revision_id,
                                    )
                                    try:
                                        _tracker.event("WEBSOCKET_SENT", event_id=_evt_id, revision_id=revision_id)
                                        await websocket.send_json(response)
                                    except Exception as send_err:
                                        logging.error(f"Cannot send interim deepgram ws msg: {send_err}")
                        except asyncio.TimeoutError:
                            continue
                        except asyncio.CancelledError:
                            break
                        except Exception as e:
                            logging.error(f"Chyba při zpracování: {e}")
                finally:
                    await worker.stop()
            
            # Spustit task pro zpracování výsledků
            process_task = asyncio.create_task(process_results())
            
            # Přijímání audio dat z prohlížeče
            try:
                if first_audio:
                    _tracker.note_audio()
                    dg_socket.send_media(first_audio)
                while not stop_event.is_set():
                    msg = await websocket.receive()
                    if msg.get("type") == "websocket.disconnect":
                        break
                    if msg.get("type") != "websocket.receive":
                        continue
                    data = msg.get("bytes")
                    if data:
                        _tracker.note_audio()
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
                if worker:
                    await worker.stop()

    
    except WebSocketDisconnect:
        logging.info("Deepgram WebSocket odpojen klientem.")
    except Exception as e:
        print(f"Deepgram chyba: {str(e)}")
        logging.error(f"Deepgram chyba: {str(e)}")
        try:
            await websocket.send_json({"error": str(e)})
        except Exception as send_err:
            logging.debug(f"Nelze poslat Deepgram error: {send_err}")
    finally:
        await _tracker.stop()


@router.websocket("/ws/elevenlabs")
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

    if not config.ELEVENLABS_API_KEY:
        logging.error("config.ELEVENLABS_API_KEY není nastaven")
        await websocket.send_json({"error": "config.ELEVENLABS_API_KEY not configured"})
        await websocket.close()
        return

    translator = Translator()

    # Session defaults.
    translate_src = "cs"
    translate_dests: list[str] = ["en", "ru"]
    translate_interim = True
    el_language_code = ""
    el_commit_strategy = "vad"

    _el_session_id = f"el-{int(time.time()*1000)}"
    _tracker = LatencyTracker(session_id=_el_session_id, engine="elevenlabs")
    _tracker.start()
    _el_utt_counter = 0

    # Read optional config message (first message may be JSON config or audio).
    first_audio: bytes | None = None
    try:
        print("WAITING FOR WEBSOCKET RECEIVE"); first = await websocket.receive(); print("RECEIVED FIRST MESSAGE:", first)
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
        el_params += "&vad_silence_threshold_secs=0.4"
    el_ws_url = f"{config.ELEVENLABS_WS_URL}?{el_params}"

    def _el_payload(
        *,
        msg_type: str,
        original: str,
        translations: dict[str, str],
        error: str | None = None,
        timing: dict[str, int] | None = None,
        revision_id: int | None = None,
    ) -> dict:
        payload: dict = {
            "type": msg_type,
            "original": original,
            "dests": translate_dests,
            "translations": translations,
        }
        if revision_id is not None:
            payload["revision_id"] = revision_id
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
            additional_headers={"xi-api-key": config.ELEVENLABS_API_KEY},
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
                        _tracker.note_audio()
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

        revision_id = 0
        worker = None
        async def _receive_transcripts():
            """Read transcripts from ElevenLabs and send translated results to browser."""
            nonlocal translator, worker, _el_utt_counter
            stability = StabilityEngine()
            worker = TranslationWorker(translator, translate_dests, tracker=_tracker)
            worker.start()
            
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
                        
                        _el_utt_counter += 1
                        _evt_id = f"el-utt-{_el_utt_counter}"
                        _tracker.event("STT_PARTIAL", event_id=_evt_id, text=text)
                            
                        new_stable = stability.update(text)
                        if new_stable:
                            _tracker.event("STT_STABLE", event_id=_evt_id, text=new_stable)
                            _tracker.event("STABILITY_EMITTED", event_id=_evt_id, text=new_stable, queue_size=worker.queue.qsize())
                            _tracker.event("TRANSLATION_QUEUED", event_id=_evt_id, text=new_stable, queue_size=worker.queue.qsize())
                            worker.enqueue(new_stable, translate_src, _evt_id)
                            
                        if translate_interim:
                            unemitted = stability.get_unemitted(text)
                            unemitted_tr = {d: "" for d in translate_dests}
                            if unemitted.strip():
                                try:
                                    # Translate unemitted quickly
                                    results = await asyncio.gather(*[
                                        _translate(translator, unemitted, src=translate_src, dest=d)
                                        for d in translate_dests
                                    ])
                                    unemitted_tr = {d: (r.text if r else "") for d, r in zip(translate_dests, results)}
                                except Exception:
                                    pass
                                    
                            current_translations = {}
                            for d in translate_dests:
                                curr = worker.stable_translations.get(d, "")
                                if curr and unemitted_tr[d]:
                                    curr += " "
                                curr += unemitted_tr[d]
                                current_translations[d] = curr
                                
                            original_full = worker.stable_original
                            if original_full and unemitted.strip():
                                original_full += " "
                            original_full += unemitted
                            
                            revision_id += 1
                            response = _el_payload(
                                msg_type="interim",
                                original=original_full,
                                translations=current_translations,
                                timing={"translate_ms": 0},
                                revision_id=revision_id,
                            )
                        else:
                            revision_id += 1
                            response = _el_payload(
                                msg_type="interim",
                                original=text,
                                translations={d: "" for d in translate_dests},
                                timing={"translate_ms": 0},
                                revision_id=revision_id,
                            )
                        _tracker.event("WEBSOCKET_SENT", event_id=_evt_id, revision_id=revision_id)
                        await websocket.send_json(response)

                    elif msg_type in ("committed_transcript", "committed_transcript_with_timestamps"):
                        text = ev.get("text", "").strip()
                        if not text:
                            continue
                        
                        _el_utt_counter += 1
                        _evt_id = f"el-utt-{_el_utt_counter}"
                        _tracker.event("STT_FINAL", event_id=_evt_id, text=text)
                            
                        unemitted = stability.get_unemitted(text)
                        if unemitted.strip():
                            _tracker.event("STABILITY_EMITTED", event_id=_evt_id, text=unemitted, queue_size=worker.queue.qsize())
                            _tracker.event("TRANSLATION_QUEUED", event_id=_evt_id, text=unemitted, queue_size=worker.queue.qsize())
                            worker.enqueue(unemitted, translate_src, _evt_id)
                            
                        # Wait for translation to complete
                        await worker.queue.join()
                        
                        revision_id += 1
                        response = _el_payload(
                            msg_type="final",
                            original=worker.stable_original,
                            translations=worker.stable_translations,
                            timing={"translate_ms": 0},
                            revision_id=revision_id,
                        )
                        _tracker.event("WEBSOCKET_SENT", event_id=_evt_id, revision_id=revision_id)
                        await websocket.send_json(response)
                        
                        stability.reset()
                        worker.reset()

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
        if worker:
            await worker.stop()
        await _tracker.stop()


@router.websocket("/ws/gemini-live")
@router.websocket("/ws/gemini_live")
async def gemini_live_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for Gemini Live Simultaneous Audio Translation.
    Supports Mode A (Direct Audio -> Gemini Live) and Mode B (STT -> Text -> Gemini Live),
    with seamless automatic failover/fallback to Google Translate and failback recovery.
    """
    if not await _require_staff_ws_auth(websocket):
        return

    await websocket.accept()

    gemini_key = config.GEMINI_API_KEY
    fallback_provider = GoogleTranslationProvider()

    # Session defaults (English -> Spanish)
    translate_src = "en"
    translate_dests: list[str] = ["es"]
    translate_interim = True
    session_glossary: dict[str, str] = {}
    last_transcript = ""
    selected_model = os.getenv("GEMINI_LIVE_MODEL", config.DEFAULT_GEMINI_LIVE_MODEL).strip()
    mode = os.getenv("GEMINI_LIVE_MODE", "A").upper()  # "A" = Direct Audio, "B" = Text

    first_audio: bytes | None = None
    first_cmd: dict | None = None
    try:
        print("WAITING FOR WEBSOCKET RECEIVE"); first = await websocket.receive(); print("RECEIVED FIRST MESSAGE:", first)
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
    gemini_ws_url = f"{config.GEMINI_LIVE_WS_URL}?key={gemini_key}"
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
        logging.error("[GeminiLive] Authentication error: config.GEMINI_API_KEY is not configured")
        global_gemini_live_circuit.trip(ErrorType.AUTH_ERROR, "config.GEMINI_API_KEY not configured")
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



