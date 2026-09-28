import time
from fastapi import APIRouter, Request, HTTPException, Response
from app.core.dependencies import _require_http_auth, _require_staff_http_auth
from app.session_manager import SessionManager, TranscriptEvent, LatencyMetrics
from app.translation_provider import get_translation_provider, normalize_brand_terms
from app.exporter import export_vtt, export_srt, export_txt

router = APIRouter()
session_manager = SessionManager.get_instance()

# --- Multi-Session REST API ---

@router.get("/api/sessions")
async def api_get_sessions(request: Request):
    """Return list of all conference stages."""
    _require_http_auth(request)
    sessions = await session_manager.list_sessions()
    return {"sessions": [s.to_dict() for s in sessions]}


@router.get("/api/sessions/{session_id}")
async def api_get_session(request: Request, session_id: str):
    """Return single session details including recent history."""
    _require_http_auth(request)
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    return session.to_dict(include_history=True)


@router.post("/api/sessions")
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


@router.patch("/api/sessions/{session_id}")
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


@router.delete("/api/sessions/{session_id}")
async def api_delete_session(request: Request, session_id: str):
    """Delete a stage session (Staff Only)."""
    _require_staff_http_auth(request)
    deleted = await session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="session_not_found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/api/session/{session_id}/clear")
@router.delete("/api/session/{session_id}/history")
async def api_clear_session_history(request: Request, session_id: str):
    """Clear transcript history for a single stage (Staff Only)."""
    _require_staff_http_auth(request)
    cleared = await session_manager.clear_session_history(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="session_not_found")
    return {"status": "cleared", "session_id": session_id}


@router.post("/api/sessions/clear-history")
@router.delete("/api/sessions/history")
async def api_clear_all_sessions_history(request: Request):
    """Clear transcript history across all stages (Staff Only)."""
    _require_staff_http_auth(request)
    count = await session_manager.clear_all_sessions_history()
    return {"status": "all_cleared", "sessions_cleared": count}


@router.post("/api/session/{session_id}/inject")
async def api_inject_transcript(request: Request, session_id: str):
    """Inject a transcript event into a session (Staff Only for production; Sandbox open for demo stages)."""
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    if not session.is_demo:
        _require_staff_http_auth(request)

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
    translations = {k: normalize_brand_terms(v) for k, v in translations.items()}
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


@router.get("/api/session/{session_id}/export/vtt")
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


@router.get("/api/session/{session_id}/export/srt")
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


@router.get("/api/session/{session_id}/export/txt")
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


