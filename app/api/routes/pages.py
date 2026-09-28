from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from app.core.dependencies import _check_html_auth, _require_staff_html_auth, _index_context, templates
from app.session_manager import SessionManager

router = APIRouter()
session_manager = SessionManager.get_instance()

@router.get("/", response_class=HTMLResponse)
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


@router.get("/session/{session_id}", response_class=HTMLResponse)
@router.get("/audience/{session_id}", response_class=HTMLResponse)
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


@router.get("/display/{session_id}", response_class=HTMLResponse)
@router.get("/stage/{session_id}/display", response_class=HTMLResponse)
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


@router.get("/speaker/{session_id}", response_class=HTMLResponse)
@router.get("/session/{session_id}/speaker", response_class=HTMLResponse)
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


@router.get("/producer", response_class=HTMLResponse)
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


@router.get("/demo", response_class=HTMLResponse)
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


@router.get("/standalone", response_class=HTMLResponse)
async def get_standalone_translator(request: Request):
    """Original single-user real-time STT & translation interface (Staff Only)."""
    auth_resp = _require_staff_html_auth(request)
    if auth_resp:
        return auth_resp
    return templates.TemplateResponse(request, "index.html", _index_context())


@router.get("/deepgram", response_class=HTMLResponse)
async def get_deepgram_index(request: Request):
    """Legacy endpoint — redirects to main UI."""
    return RedirectResponse(url="/", status_code=303)



