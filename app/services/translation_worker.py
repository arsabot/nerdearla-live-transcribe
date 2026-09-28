import asyncio
import logging
import time
from typing import Dict, List, Optional
from app.translator import Translator
from app.services.utils import _translate

class TranslationWorker:
    def __init__(self, translator: Translator, dest_langs: List[str], tracker=None):
        self.translator = translator
        self.dest_langs = dest_langs
        self.queue = asyncio.Queue()
        self.stable_original = ""
        self.stable_translations: Dict[str, str] = {d: "" for d in dest_langs}
        self.worker_task: Optional[asyncio.Task] = None
        self._tracker = tracker  # optional LatencyTracker
        
    def set_dest_langs(self, dest_langs: List[str]):
        self.dest_langs = dest_langs
        for d in dest_langs:
            if d not in self.stable_translations:
                self.stable_translations[d] = ""
        
    def enqueue(self, stable_chunk: str, src_lang: str, event_id: str = "", revision_id: Optional[int] = None, stt_ts: Optional[float] = None):
        self.queue.put_nowait((stable_chunk, src_lang, event_id, revision_id, stt_ts, time.time() * 1000))
        
    async def process_queue(self):
        while True:
            item = await self.queue.get()
            # Backwards-compat unpacking:
            if len(item) == 6:
                chunk, src_lang, event_id, revision_id, stt_ts, enqueued_ts = item
            elif len(item) == 3:
                chunk, src_lang, event_id = item
                revision_id, stt_ts, enqueued_ts = None, None, time.time() * 1000
            else:
                chunk, src_lang = item
                event_id, revision_id, stt_ts, enqueued_ts = "", None, None, time.time() * 1000

            t0 = time.time() * 1000
            try:
                if self._tracker is not None and event_id:
                    self._tracker.event("TRANSLATION_STARTED", event_id=event_id, revision_id=revision_id,
                                        stt_ts=stt_ts, text=chunk, queue_size=self.queue.qsize())
                results = await asyncio.gather(*[
                    _translate(self.translator, chunk, src=src_lang, dest=d)
                    for d in self.dest_langs
                ])
                
                self.stable_original += (" " if self.stable_original else "") + chunk
                for d, res in zip(self.dest_langs, results):
                    if res and res.text:
                        self.stable_translations[d] += (" " if self.stable_translations[d] else "") + res.text

                if self._tracker is not None and event_id:
                    latency_ms = time.time() * 1000 - t0
                    self._tracker.event("TRANSLATION_COMPLETED", event_id=event_id, revision_id=revision_id,
                                        stt_ts=stt_ts, latency_ms=latency_ms, queue_size=self.queue.qsize())
            except Exception as e:
                logging.error(f"Error chunk translation: {e}")
                if self._tracker is not None and event_id:
                    self._tracker.event("TRANSLATION_FAILED", event_id=event_id, revision_id=revision_id,
                                        stt_ts=stt_ts, error=str(e)[:120], queue_size=self.queue.qsize())
            finally:
                self.queue.task_done()
                
    def start(self):
        self.worker_task = asyncio.create_task(self.process_queue())
        
    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                # We do not await directly to avoid CancelledError bubbling up
                # if the caller is already cancelled. Anyio will wait for it.
                pass
            except Exception:
                pass
                
    def reset(self):
        self.stable_original = ""
        self.stable_translations = {d: "" for d in self.dest_langs}
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
