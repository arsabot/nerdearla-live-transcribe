import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from starlette.websockets import WebSocket


@dataclass
class LatencyMetrics:
    audio_ms: Optional[float] = None
    stt_ms: Optional[float] = None
    translation_ms: Optional[float] = None
    delivery_ms: Optional[float] = None
    total_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_ms": round(self.audio_ms, 1) if self.audio_ms is not None else None,
            "stt_ms": round(self.stt_ms, 1) if self.stt_ms is not None else None,
            "translation_ms": round(self.translation_ms, 1) if self.translation_ms is not None else None,
            "delivery_ms": round(self.delivery_ms, 1) if self.delivery_ms is not None else None,
            "total_ms": round(self.total_ms, 1) if self.total_ms is not None else None,
        }


@dataclass
class TranscriptEvent:
    session_id: str
    type: str  # "interim" | "final"
    timestamp: float = field(default_factory=time.time)
    source_language: str = "en"
    original: str = ""
    translations: Dict[str, str] = field(default_factory=dict)
    speaker: Optional[str] = None
    metrics: Optional[LatencyMetrics] = None
    client_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "session_id": self.session_id,
            "type": self.type,
            "timestamp": self.timestamp,
            "source_language": self.source_language,
            "original": self.original,
            "translations": self.translations,
        }
        if self.speaker:
            data["speaker"] = self.speaker
        if self.metrics:
            data["metrics"] = self.metrics.to_dict()
        if self.client_id is not None:
            data["client_id"] = self.client_id
        return data


@dataclass
class Session:
    id: str
    name: str
    speaker: str = "Keynote Speaker"
    source_language: str = "en"
    target_languages: List[str] = field(default_factory=lambda: ["es"])
    status: str = "live"  # scheduled, live, paused, finished, error
    engine: str = "webspeech"  # webspeech, whisper, nemotron, deepgram, elevenlabs
    translation_provider: str = "gemini"  # gemini, gemini-flash, googletrans, none
    description: str = ""
    glossary: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    history: List[TranscriptEvent] = field(default_factory=list)
    last_metrics: Optional[LatencyMetrics] = None
    is_demo: bool = False

    # Active viewer websockets for this session
    viewers: Set[WebSocket] = field(default_factory=set, repr=False)
    # Active producer websockets for this session
    producers: Set[WebSocket] = field(default_factory=set, repr=False)

    def to_dict(self, include_history_count: bool = True, include_history: bool = False) -> Dict[str, Any]:
        data = {
            "id": self.id,
            "name": self.name,
            "speaker": self.speaker,
            "source_language": self.source_language,
            "target_languages": self.target_languages,
            "status": self.status,
            "engine": self.engine,
            "translation_provider": self.translation_provider,
            "description": self.description,
            "viewers_count": len(self.viewers),
            "producers_count": len(self.producers),
            "created_at": self.created_at,
            "glossary": self.glossary,
            "last_metrics": self.last_metrics.to_dict() if self.last_metrics else None,
            "is_demo": self.is_demo,
        }
        if include_history_count:
            data["events_count"] = len(self.history)
        if include_history:
            data["history"] = [ev.to_dict() for ev in self.history[-100:]]
        return data

    def to_storage_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "speaker": self.speaker,
            "source_language": self.source_language,
            "target_languages": self.target_languages,
            "status": self.status,
            "engine": self.engine,
            "translation_provider": self.translation_provider,
            "description": self.description,
            "glossary": self.glossary,
            "created_at": self.created_at,
            "is_demo": self.is_demo,
            "history": [ev.to_dict() for ev in self.history[-500:]],
        }

    @classmethod
    def from_storage_dict(cls, data: Dict[str, Any]) -> "Session":
        history_raw = data.get("history", [])
        history = [
            TranscriptEvent(
                session_id=ev.get("session_id", data.get("id", "")),
                type=ev.get("type", "final"),
                timestamp=ev.get("timestamp", time.time()),
                source_language=ev.get("source_language", "en"),
                original=ev.get("original", ""),
                translations=ev.get("translations", {}),
                speaker=ev.get("speaker"),
                client_id=ev.get("client_id"),
            )
            for ev in history_raw
        ]
        return cls(
            id=data["id"],
            name=data["name"],
            speaker=data.get("speaker", "Keynote Speaker"),
            source_language=data.get("source_language", "en"),
            target_languages=data.get("target_languages", ["es"]),
            status=data.get("status", "live"),
            engine=data.get("engine", "webspeech"),
            translation_provider=data.get("translation_provider", "gemini"),
            description=data.get("description", ""),
            glossary=data.get("glossary", {}),
            created_at=data.get("created_at", time.time()),
            history=history,
            is_demo=bool(data.get("is_demo", False) or data.get("id", "").startswith("demo-")),
        )


class SessionManager:
    """
    Persistent, thread-safe and async-friendly Session Manager.
    Saves stages and transcript histories to disk for reload / restart survival.
    """
    _instance: Optional["SessionManager"] = None

    def __init__(self, storage_path: Optional[str] = None):
        if storage_path:
            self._storage_path = Path(storage_path)
        else:
            base_dir = Path(__file__).resolve().parent.parent
            self._storage_path = Path(os.getenv("SESSIONS_DATA_PATH", str(base_dir / "data" / "sessions_state.json")))

        self._sessions: Dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._save_task: Optional[asyncio.Task] = None
        self._load_from_storage()

    @classmethod
    def get_instance(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _init_default_stages(self) -> None:
        default_stages = [
            Session(
                id="stage-a",
                name="AI & Open Source",
                speaker="Linus Torvalds & Community",
                source_language="en",
                target_languages=["es", "pt"],
                status="live",
                engine="webspeech",
                translation_provider="gemini",
                description="Keynotes, LLMs on device, ONNX and open AI architectures.",
                glossary={
                    "Kubernetes": "Kubernetes",
                    "container": "contenedor",
                    "pull request": "pull request",
                    "deployment": "despliegue",
                    "repository": "repositorio",
                    "open source": "código abierto",
                    "real-time": "tiempo real",
                    "latency": "latencia",
                    "vibeathon": "Vibeathon",
                    "nerdearla": "Nerdearla",
                },
            ),
            Session(
                id="stage-b",
                name="Cloud Infrastructure & DevOps",
                speaker="Kelsey Hightower & Cloud Architects",
                source_language="en",
                target_languages=["es", "pt"],
                status="live",
                engine="webspeech",
                translation_provider="gemini",
                description="Microservices, multi-region scaling, observability & resilience.",
                glossary={
                    "Kubernetes": "Kubernetes",
                    "cluster": "clúster",
                    "load balancer": "balanceador de carga",
                    "pipeline": "pipeline",
                    "observability": "observabilidad",
                    "throughput": "rendimiento",
                    "nerdearla": "Nerdearla",
                },
            ),
            Session(
                id="stage-c",
                name="Web & Frontend Architecture",
                speaker="Frontend Core Team",
                source_language="en",
                target_languages=["es", "pt"],
                status="scheduled",
                engine="whisper",
                translation_provider="gemini",
                description="WebAssembly, WebGPU, streaming WebSockets and modern UX.",
                glossary={
                    "WebAssembly": "WebAssembly",
                    "WebSockets": "WebSockets",
                    "rendering": "renderizado",
                    "nerdearla": "Nerdearla",
                },
            ),
            Session(
                id="stage-d",
                name="Cybersecurity & SecOps",
                speaker="Security Experts",
                source_language="en",
                target_languages=["es", "pt"],
                status="scheduled",
                engine="nemotron",
                translation_provider="gemini",
                description="Zero trust architectures, supply chain security and privacy.",
                glossary={
                    "zero trust": "confianza cero",
                    "encryption": "cifrado",
                    "vulnerability": "vulnerabilidad",
                    "nerdearla": "Nerdearla",
                },
                is_demo=False,
            ),
            # Dedicated Isolated Sandbox Test Stages for live demos and synthetic injections
            Session(
                id="demo-stage-a",
                name="🧪 Demo Stage A — Keynote EN (Inglés → ES / PT)",
                speaker="Alex Rivera (AI Research)",
                source_language="en",
                target_languages=["es", "pt"],
                status="live",
                engine="webspeech",
                translation_provider="gemini",
                description="Sala demo con orador en inglés: traduce en simultáneo a español y portugués.",
                glossary={
                    "Kubernetes": "Kubernetes",
                    "container": "contenedor",
                    "pull request": "pull request",
                    "deployment": "despliegue",
                    "open source": "código abierto",
                    "real-time": "tiempo real",
                    "nerdearla": "Nerdearla",
                },
                is_demo=True,
            ),
            Session(
                id="demo-stage-b",
                name="🧪 Demo Stage B — DevOps PT (Português → EN / ES)",
                speaker="Thiago Silva (Cloud Specialist)",
                source_language="pt",
                target_languages=["en", "es"],
                status="live",
                engine="webspeech",
                translation_provider="gemini",
                description="Sala demo com orador em português: traduz em tempo real para inglês e espanhol.",
                glossary={
                    "Kubernetes": "Kubernetes",
                    "nuvem": "cloud",
                    "balanceador de carga": "load balancer",
                    "desenvolvedores": "developers",
                    "observabilidade": "observability",
                    "nerdearla": "Nerdearla",
                },
                is_demo=True,
            ),
            Session(
                id="demo-stage-c",
                name="🧪 Demo Stage C — Arquitectura ES (Español → EN / PT)",
                speaker="Camila Gómez (Core Architect)",
                source_language="es",
                target_languages=["en", "pt"],
                status="live",
                engine="webspeech",
                translation_provider="gemini",
                description="Sala demo con oradora en español: traduce en simultáneo a inglés y portugués.",
                glossary={
                    "código abierto": "open source",
                    "despliegue": "deployment",
                    "rendimiento": "performance",
                    "tiempo real": "real-time",
                    "nerdearla": "Nerdearla",
                },
                is_demo=True,
            ),
        ]
        for stage in default_stages:
            if stage.id not in self._sessions:
                self._sessions[stage.id] = stage
            else:
                self._sessions[stage.id].is_demo = stage.is_demo
                if stage.is_demo:
                    self._sessions[stage.id].source_language = stage.source_language
                    self._sessions[stage.id].target_languages = stage.target_languages
                    self._sessions[stage.id].name = stage.name
                    self._sessions[stage.id].speaker = stage.speaker
                    self._sessions[stage.id].description = stage.description
                    self._sessions[stage.id].glossary = stage.glossary

    def _load_from_storage(self) -> None:
        """Load sessions and histories from JSON file on disk."""
        if self._storage_path.exists():
            try:
                with open(self._storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict) and "sessions" in data:
                        for s_data in data["sessions"]:
                            session = Session.from_storage_dict(s_data)
                            self._sessions[session.id] = session
                        logging.info("Loaded %d sessions from persistent storage: %s", len(self._sessions), self._storage_path)
            except Exception as e:
                logging.error("Failed to load sessions from storage (%s): %s", self._storage_path, e)

        # Ensure default stages are available
        self._init_default_stages()
        self._save_to_storage_sync()

    def _save_to_storage_sync(self) -> None:
        """Synchronously write state to disk with atomic replacement."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._storage_path.with_suffix(".tmp")
            payload = {
                "version": 1,
                "saved_at": time.time(),
                "sessions": [s.to_storage_dict() for s in self._sessions.values()],
            }
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self._storage_path)
        except Exception as e:
            logging.error("Failed to save sessions to storage (%s): %s", self._storage_path, e)

    def trigger_save(self) -> None:
        """Non-blocking background save trigger."""
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(asyncio.to_thread(self._save_to_storage_sync))
        except RuntimeError:
            self._save_to_storage_sync()

    async def list_sessions(self) -> List[Session]:
        return list(self._sessions.values())

    async def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def create_session(
        self,
        session_id: str,
        name: str,
        speaker: str = "Speaker",
        source_language: str = "en",
        target_languages: Optional[List[str]] = None,
        engine: str = "webspeech",
        translation_provider: str = "gemini",
        description: str = "",
        glossary: Optional[Dict[str, str]] = None,
    ) -> Session:
        async with self._lock:
            sid = session_id.strip().lower().replace(" ", "-")
            session = Session(
                id=sid,
                name=name,
                speaker=speaker,
                source_language=source_language.strip().lower(),
                target_languages=target_languages or ["es"],
                status="live",
                engine=engine,
                translation_provider=translation_provider,
                description=description,
                glossary=glossary or {},
            )
            self._sessions[sid] = session
            self.trigger_save()
            return session

    async def update_session(
        self,
        session_id: str,
        *,
        name: Optional[str] = None,
        speaker: Optional[str] = None,
        source_language: Optional[str] = None,
        target_languages: Optional[List[str]] = None,
        status: Optional[str] = None,
        engine: Optional[str] = None,
        translation_provider: Optional[str] = None,
        description: Optional[str] = None,
        glossary: Optional[Dict[str, str]] = None,
    ) -> Optional[Session]:
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            if name is not None:
                session.name = name
            if speaker is not None:
                session.speaker = speaker
            if source_language is not None:
                session.source_language = source_language
            if target_languages is not None:
                session.target_languages = target_languages
            if status is not None:
                session.status = status
            if engine is not None:
                session.engine = engine
            if translation_provider is not None:
                session.translation_provider = translation_provider
            if description is not None:
                session.description = description
            if glossary is not None:
                session.glossary = glossary
            self.trigger_save()
            return session

    async def delete_session(self, session_id: str) -> bool:
        async with self._lock:
            if session_id in self._sessions:
                session = self._sessions.pop(session_id)
                self.trigger_save()
                for ws in list(session.viewers) + list(session.producers):
                    try:
                        await ws.close(code=1000, reason="Session deleted")
                    except Exception:
                        pass
                return True
            return False

    async def add_viewer(self, session_id: str, websocket: WebSocket) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.viewers.add(websocket)
        return True

    async def remove_viewer(self, session_id: str, websocket: WebSocket) -> None:
        session = self._sessions.get(session_id)
        if session and websocket in session.viewers:
            session.viewers.remove(websocket)

    async def add_producer(self, session_id: str, websocket: WebSocket) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.producers.add(websocket)
        return True

    async def remove_producer(self, session_id: str, websocket: WebSocket) -> None:
        session = self._sessions.get(session_id)
        if session and websocket in session.producers:
            session.producers.remove(websocket)

    async def record_event(self, session_id: str, event: TranscriptEvent) -> None:
        session = self._sessions.get(session_id)
        if not session:
            return
        if event.metrics:
            session.last_metrics = event.metrics
        if event.type == "final" and event.original.strip():
            session.history.append(event)
            # Persist committed events
            self.trigger_save()

    async def broadcast_event(self, session_id: str, event: TranscriptEvent) -> int:
        session = self._sessions.get(session_id)
        if not session:
            return 0

        await self.record_event(session_id, event)
        payload = event.to_dict()

        recipients = list(session.viewers) + list(session.producers)
        if not recipients:
            return 0

        async def _send(ws: WebSocket):
            try:
                await ws.send_json(payload)
                return True
            except Exception:
                return False

        results = await asyncio.gather(*[_send(ws) for ws in recipients], return_exceptions=True)
        return sum(1 for r in results if r is True)

    async def clear_session_history(self, session_id: str) -> bool:
        """Clear all transcript history and metrics for a single session and broadcast to clients."""
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return False
            session.history = []
            session.last_metrics = None
            self.trigger_save()

            payload = {"type": "clear_history", "session_id": session_id}
            recipients = list(session.viewers) + list(session.producers)
            for ws in recipients:
                try:
                    await ws.send_json(payload)
                except Exception:
                    pass
            return True

    async def clear_all_sessions_history(self) -> int:
        """Clear all transcript history and metrics for all sessions and broadcast to clients."""
        async with self._lock:
            count = 0
            for session_id, session in self._sessions.items():
                session.history = []
                session.last_metrics = None
                count += 1
                payload = {"type": "clear_history", "session_id": session_id}
                recipients = list(session.viewers) + list(session.producers)
                for ws in recipients:
                    try:
                        await ws.send_json(payload)
                    except Exception:
                        pass
            self.trigger_save()
            return count
