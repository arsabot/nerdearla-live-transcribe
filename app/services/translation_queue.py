import asyncio
import time
from typing import Dict, List, Optional
from app.services.translator import Translator, _translate

class TranslationWorker:
    def __init__(self, translator: Translator, dest_langs: List[str]):
        self.translator = translator
        self.dest_langs = dest_langs
        self.queue = asyncio.Queue()
        self.stable_original = ""
        self.stable_translations: Dict[str, str] = {d: "" for d in dest_langs}
        self.worker_task: Optional[asyncio.Task] = None
        
    def enqueue(self, stable_chunk: str, src_lang: str):
        self.queue.put_nowait((stable_chunk, src_lang))
        
    async def process_queue(self):
        while True:
            chunk, src_lang = await self.queue.get()
            try:
                results = await asyncio.gather(*[
                    _translate(self.translator, chunk, src=src_lang, dest=d)
                    for d in self.dest_langs
                ])
                
                self.stable_original += (" " if self.stable_original else "") + chunk
                for d, res in zip(self.dest_langs, results):
                    if res and res.text:
                        self.stable_translations[d] += (" " if self.stable_translations[d] else "") + res.text
            except Exception as e:
                import logging
                logging.error(f"Error chunk translation: {e}")
            finally:
                self.queue.task_done()
                
    def start(self):
        self.worker_task = asyncio.create_task(self.process_queue())
        
    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
                
    def reset(self):
        self.stable_original = ""
        self.stable_translations = {d: "" for d in self.dest_langs}
        # Clear queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break
