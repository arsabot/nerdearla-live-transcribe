from app.core import config
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.core.security import _CSPMiddleware
from app.core.dependencies import templates
from app.gemini_live import global_gemini_live_circuit
from app.session_manager import SessionManager

from app.api.routes import pages, sessions, auth, websockets

logging.basicConfig(level=logging.INFO)

app = FastAPI()

app.add_middleware(_CSPMiddleware)

app.mount("/static", StaticFiles(directory="app/static"), name="static")

app.include_router(pages.router)
app.include_router(sessions.router)
app.include_router(auth.router)
app.include_router(websockets.router)

session_manager = SessionManager.get_instance()

@app.get("/health")
async def health():
    """Health check endpoint for Docker HEALTHCHECK, load balancers, and metrics."""
    sessions_list = await session_manager.list_sessions()
    active_sessions = sum(1 for s in sessions_list if s.status == "live")
    active_viewers = sum(len(s.viewers) for s in sessions_list)
    
    gemini_key_present = bool(config.GEMINI_API_KEY)
    gemini_circuit_ok = global_gemini_live_circuit.is_available()
    gemini_live_available = gemini_key_present and gemini_circuit_ok
    
    if not gemini_key_present:
        gemini_live_status = "Gemini Live UNAVAILABLE"
    elif not gemini_circuit_ok:
        gemini_live_status = "Google Translate FALLBACK"
    else:
        gemini_live_status = "Gemini Live CONNECTED"

    return {
        "status": "ok",
        "active_sessions": active_sessions,
        "active_viewers": active_viewers,
        "gemini_live_available": gemini_live_available,
        "gemini_live_status": gemini_live_status,
        "gemini_live_model": config.DEFAULT_GEMINI_LIVE_MODEL,
        "active_fallback": "google_translate",
    }
