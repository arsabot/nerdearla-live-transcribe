import asyncio
import re
import urllib.parse
from typing import Dict, List, Optional
import httpx
from googletrans.constants import LANGUAGES

class TranslationResult:
    def __init__(self, text: str, src: str, dest: str, extra_data: Optional[dict] = None):
        self.text = text
        self.src = src
        self.dest = dest
        self.extra_data = extra_data or {}

    def __repr__(self):
        return f"<TranslationResult text={self.text!r} src={self.src!r} dest={self.dest!r}>"

def normalize_lang_for_google(code: str) -> str:
    if not code:
        return "auto"
    c = code.strip().lower()
    if c in ("auto", ""):
        return "auto"
    # Special cases for Chinese
    if c in ("zh-cn", "zh_cn", "zh-hans", "zh"):
        return "zh-CN"
    if c in ("zh-tw", "zh_tw", "zh-hant"):
        return "zh-TW"
    # For standard codes like es-ES, en-US, pt-BR:
    if "-" in c:
        parts = c.split("-")
        if parts[0] in LANGUAGES:
            return parts[0]
    if "_" in c:
        parts = c.split("_")
        if parts[0] in LANGUAGES:
            return parts[0]
    return c

class RobustTranslator:
    _shared_client: Optional[httpx.AsyncClient] = None
    _cache: Dict[str, str] = {}
    _MAX_CACHE = 2000

    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout
        self._headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }

    @classmethod
    async def get_client(cls, headers: dict, timeout: float) -> httpx.AsyncClient:
        if cls._shared_client is None or cls._shared_client.is_closed:
            cls._shared_client = httpx.AsyncClient(
                headers=headers,
                timeout=timeout,
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                follow_redirects=True,
            )
        return cls._shared_client

    async def translate(self, text: str, *, src: str = "auto", dest: str = "en") -> TranslationResult:
        if not text or not text.strip():
            return TranslationResult(text="", src=src, dest=dest)

        cleaned_text = text.strip()
        sl = normalize_lang_for_google(src)
        tl = normalize_lang_for_google(dest)

        cache_key = f"{sl}:{tl}:{cleaned_text}"
        if cache_key in self._cache:
            return TranslationResult(text=self._cache[cache_key], src=sl, dest=tl)

        client = await self.get_client(self._headers, self.timeout)

        # Strategy 1: Google Translate dict-chrome-ex endpoint
        try:
            r = await client.get(
                "https://translate.googleapis.com/translate_a/single",
                params={"client": "dict-chrome-ex", "sl": sl, "tl": tl, "dt": "t", "q": cleaned_text},
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data and isinstance(data[0], list):
                    segs = [s[0] for s in data[0] if isinstance(s, list) and len(s) > 0 and isinstance(s[0], str)]
                    if segs:
                        res_text = "".join(segs)
                        self._set_cache(cache_key, res_text)
                        return TranslationResult(text=res_text, src=sl, dest=tl)
        except Exception:
            pass

        # Strategy 2: Google Translate at (Android client)
        try:
            r = await client.get(
                "https://translate.googleapis.com/translate_a/single",
                params={"client": "at", "sl": sl, "tl": tl, "dt": "t", "q": cleaned_text},
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data and isinstance(data[0], list):
                    segs = [s[0] for s in data[0] if isinstance(s, list) and len(s) > 0 and isinstance(s[0], str)]
                    if segs:
                        res_text = "".join(segs)
                        self._set_cache(cache_key, res_text)
                        return TranslationResult(text=res_text, src=sl, dest=tl)
        except Exception:
            pass

        # Strategy 3: clients5.google.com endpoint
        try:
            r = await client.get(
                "https://clients5.google.com/translate_a/t",
                params={"client": "dict-chrome-ex", "sl": sl, "tl": tl, "q": cleaned_text},
            )
            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list) and data:
                    res_text = "".join([s for s in data if isinstance(s, str)])
                    if res_text:
                        self._set_cache(cache_key, res_text)
                        return TranslationResult(text=res_text, src=sl, dest=tl)
        except Exception:
            pass

        # Strategy 4: MyMemory Translation API fallback
        try:
            pair = f"{sl}|{tl}" if sl != "auto" else f"en|{tl}"
            r = await client.get(
                "https://api.mymemory.translated.net/get",
                params={"q": cleaned_text, "langpair": pair},
            )
            if r.status_code == 200:
                data = r.json()
                res_text = data.get("responseData", {}).get("translatedText")
                if res_text:
                    self._set_cache(cache_key, res_text)
                    return TranslationResult(text=res_text, src=sl, dest=tl)
        except Exception:
            pass

        # Fallback to returning original text if all fail
        return TranslationResult(text=cleaned_text, src=sl, dest=tl)

    def _set_cache(self, key: str, value: str) -> None:
        if len(self._cache) >= self._MAX_CACHE:
            # clear oldest entries
            self._cache.clear()
        self._cache[key] = value

Translator = RobustTranslator
