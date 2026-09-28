import asyncio
import inspect
from app.translator import Translator
from app.core import config

def _looks_like_deepgram_results(obj: object) -> bool:
    # deepgram-sdk has changed public result types across versions.
    # Use duck-typing so we don't hard-depend on a specific class import.
    return hasattr(obj, "channel") and hasattr(obj, "is_final")


async def _translate(translator: Translator, text: str, *, src: str, dest: str):
    # googletrans has had both sync and async implementations across versions.
    # Run sync translate in a worker thread to avoid blocking the event loop.
    if inspect.iscoroutinefunction(translator.translate):
        return await asyncio.wait_for(
            translator.translate(text, src=src, dest=dest),
            timeout=config.TRANSLATE_TIMEOUT_SECONDS,
        )
    return await asyncio.wait_for(
        asyncio.to_thread(lambda: translator.translate(text, src=src, dest=dest)),
        timeout=config.TRANSLATE_TIMEOUT_SECONDS,
    )



try:
    from deepgram.extensions.types.sockets.listen_v1_control_message import ListenV1ControlMessage
except Exception:
    ListenV1ControlMessage = None

def _deepgram_send_finalize(dg_socket) -> None:
    # deepgram-sdk v3 had send_finalize/send_close_stream with dedicated types.
    # deepgram-sdk v5 uses send_control(ListenV1ControlMessage(type=...)).
    if hasattr(dg_socket, "send_finalize"):
        try:
            from deepgram.listen.v1.types.listen_v1finalize import ListenV1Finalize  # type: ignore

            dg_socket.send_finalize(ListenV1Finalize(type="Finalize"))
            return
        except Exception:
            pass

    if hasattr(dg_socket, "send_control") and ListenV1ControlMessage is not None:
        try:
            dg_socket.send_control(ListenV1ControlMessage(type="Finalize"))
        except Exception:
            pass


def _deepgram_send_close_stream(dg_socket) -> None:
    if hasattr(dg_socket, "send_close_stream"):
        try:
            from deepgram.listen.v1.types.listen_v1close_stream import (  # type: ignore
                ListenV1CloseStream,
            )

            dg_socket.send_close_stream(ListenV1CloseStream(type="CloseStream"))
            return
        except Exception:
            pass

    if hasattr(dg_socket, "send_control") and ListenV1ControlMessage is not None:
        try:
            dg_socket.send_control(ListenV1ControlMessage(type="CloseStream"))
        except Exception:
            pass



