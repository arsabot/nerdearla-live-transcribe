import abc
import json
import os
import re
import time
import logging
from typing import Dict, List, Optional
from app.translator import RobustTranslator, TranslationResult


NERDEARLA_REGEX = re.compile(
    r"\b(?:"
    r"nerd(?:[-_.\s]*(?:e?ar\s*la|earl[ay]?|ear|er\s*la|eala|e\s*a\s*la|e\s*arla|a\s*la|dar\s*la|diar\s*la|dia\s*la|iar\s*la|ia\s*la|ierla|eada|eado|arlo|alas|alos|ela|ala|alla|alta|era|la)|[\.]la)"
    r"|nerd\s+(?:de\s+)?(?:habla|aula|charla|arla|alma|armas)"
    r"|nerd\s+y\s+(?:habla|arla)"
    r"|nerdy\s*(?:arla|earla)"
    r"|merd(?:[-_.\s]*(?:e?ar\s*la|er\s*la|eada|ear))"
    r"|verd(?:[-_.\s]*(?:ear\s*la|e\s+habla))"
    r")\b",
    re.IGNORECASE,
)


def normalize_brand_terms(text: str) -> str:
    """
    Normalizes common STT phonetic distortions of conference brand names:
    'nerd de habla', 'nerd arla', 'nerdear la', 'nerdeala', 'merdearla', 'nerderla', 'nerd ear' -> 'Nerdearla'.
    """
    if not text:
        return text
    return NERDEARLA_REGEX.sub("Nerdearla", text)


def apply_glossary(text: str, glossary: Optional[Dict], target_lang: str = "es") -> str:
    """
    Substitutes terms from technical glossary into translated text and ensures
    conference terms like Nerdearla are correctly detected and capitalized.
    """
    if not text:
        return ""
    
    result = normalize_brand_terms(text)
    if not glossary:
        return result
    
    target_clean = (target_lang or "es").lower()
    active_glossary = glossary

    # Check for target-scoped sub-glossary
    if target_clean in glossary and isinstance(glossary[target_clean], dict):
        active_glossary = glossary[target_clean]

    for term, replacement in active_glossary.items():
        if not term or not replacement or not isinstance(replacement, str):
            continue
        
        # If term and replacement are different words (e.g. 'deployment' -> 'despliegue'),
        # only apply if target is 'es' (the primary conference translation target)
        # to avoid inserting Spanish words into English/French/German translations.
        if term.lower() != replacement.lower() and target_clean not in ("es", "auto"):
            continue

        pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
        result = pattern.sub(replacement, result)

    return normalize_brand_terms(result)


class TranslationProvider(abc.ABC):
    """
    Abstract translation provider interface for conference translation pipelines.
    """

    @abc.abstractmethod
    async def translate(
        self,
        text: str,
        source: str = "auto",
        target: str = "es",
        glossary: Optional[Dict] = None,
    ) -> str:
        pass

    async def translate_batch(
        self,
        text: str,
        source: str = "auto",
        targets: Optional[List[str]] = None,
        glossary: Optional[Dict] = None,
    ) -> Dict[str, str]:
        if not targets:
            targets = ["es"]
        import asyncio
        results = await asyncio.gather(
            *[self.translate(text, source=source, target=t, glossary=glossary) for t in targets],
            return_exceptions=True,
        )
        out = {}
        for target_lang, res in zip(targets, results):
            if isinstance(res, Exception):
                logging.error(f"Translation failed for {target_lang}: {res}")
                out[target_lang] = ""
            else:
                out[target_lang] = str(res)
        return out


class GoogleTranslationProvider(TranslationProvider):
    """
    Google Translate provider based on existing RobustTranslator.
    """
    def __init__(self):
        self._translator = RobustTranslator()

    async def translate(
        self,
        text: str,
        source: str = "auto",
        target: str = "es",
        glossary: Optional[Dict] = None,
    ) -> str:
        if not text or not text.strip():
            return ""
        
        # If source and target are the same, return as is (with glossary applied if any)
        if source and target and source.lower() == target.lower() and source.lower() != "auto":
            return apply_glossary(text, glossary, target_lang=target)

        try:
            res: TranslationResult = await self._translator.translate(text, src=source, dest=target)
            translated_text = res.text if res else text
            return apply_glossary(translated_text, glossary, target_lang=target)
        except Exception as e:
            logging.error(f"GoogleTranslationProvider error: {e}")
            return apply_glossary(text, glossary, target_lang=target)


class GeminiTranslationProvider(TranslationProvider):
    """
    Google Gemini translation provider via REST API when GEMINI_API_KEY is configured.
    Includes a fast circuit-breaker to instantly use GoogleTranslationProvider if Gemini
    encounters invalid keys (400), rate limits (429) or high latency.
    """
    _circuit_open_until: float = 0.0

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
        self._fallback = GoogleTranslationProvider()

    def _is_circuit_open(self) -> bool:
        return time.time() < GeminiTranslationProvider._circuit_open_until

    def _trip_circuit(self, cooldown_seconds: float = 45.0) -> None:
        GeminiTranslationProvider._circuit_open_until = time.time() + cooldown_seconds

    async def translate(
        self,
        text: str,
        source: str = "auto",
        target: str = "es",
        glossary: Optional[Dict[str, str]] = None,
    ) -> str:
        if not text or not text.strip():
            return ""
        if not self.api_key or self._is_circuit_open():
            return await self._fallback.translate(text, source, target, glossary)

        import httpx
        glossary_note = ""
        if glossary:
            terms = ", ".join([f"'{k}' -> '{v}'" for k, v in glossary.items()])
            glossary_note = f"Respect these technical glossary substitutions: {terms}."

        prompt = (
            f"You are a real-time simultaneous conference translator for tech events.\n"
            f"Translate the following text from {source} to {target}.\n"
            f"{glossary_note}\n"
            f"Return ONLY the plain translated text without any explanation, markdown, asterisks, quotes or extra formatting.\n\n"
            f"Text: {text}"
        )

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
            }
            async with httpx.AsyncClient(timeout=3.5) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        content = candidates[0].get("content", {})
                        parts = content.get("parts", [])
                        if parts:
                            res_text = parts[0].get("text", "").strip()
                            res_text = res_text.replace("**", "").replace("*", "").replace('"', '').strip()
                            if res_text:
                                return apply_glossary(res_text, glossary, target_lang=target)
                elif resp.status_code in (400, 401, 403, 404, 429, 503):
                    logging.warning(f"Gemini API returned status {resp.status_code}, tripping circuit breaker for 45s.")
                    self._trip_circuit()
        except Exception as e:
            logging.warning(f"Gemini translation fast-timeout/error, falling back to Google: {e}")
            self._trip_circuit(cooldown_seconds=30.0)
        
        return await self._fallback.translate(text, source, target, glossary)

    async def translate_batch(
        self,
        text: str,
        source: str = "auto",
        targets: Optional[List[str]] = None,
        glossary: Optional[Dict] = None,
    ) -> Dict[str, str]:
        if not text or not text.strip():
            return {t: "" for t in (targets or ["es"])}
        if not targets:
            targets = ["es"]
        if not self.api_key or self._is_circuit_open():
            return await self._fallback.translate_batch(text, source=source, targets=targets, glossary=glossary)

        out = {}
        needed_targets = []
        for t in targets:
            if source and source.lower() == t.lower() and source.lower() != "auto":
                out[t] = apply_glossary(text, glossary, target_lang=t)
            else:
                needed_targets.append(t)

        if not needed_targets:
            return out

        import httpx
        glossary_note = ""
        if glossary:
            terms = ", ".join([f"'{k}' -> '{v}'" for k, v in glossary.items()])
            glossary_note = f"Respect these technical glossary substitutions: {terms}."

        prompt = (
            f"You are a real-time simultaneous conference translator for tech events.\n"
            f"Translate the following text from {source} into these languages: {', '.join(needed_targets)}.\n"
            f"{glossary_note}\n"
            f"Return strictly a valid JSON object mapping each language code to its translated text (e.g. {json.dumps({t: '...' for t in needed_targets})}).\n\n"
            f"Text: {text}"
        )

        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "temperature": 0.1,
                    "maxOutputTokens": 800,
                },
            }
            async with httpx.AsyncClient(timeout=3.5) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            raw_json = parts[0].get("text", "{}")
                            parsed = json.loads(raw_json)
                            if isinstance(parsed, dict):
                                for t in needed_targets:
                                    val = str(parsed.get(t) or parsed.get(t.lower()) or "").strip()
                                    if val:
                                        out[t] = apply_glossary(val, glossary, target_lang=t)
                elif resp.status_code in (400, 401, 403, 404, 429, 503):
                    logging.warning(f"Gemini API returned status {resp.status_code}, tripping circuit breaker.")
                    self._trip_circuit()
        except Exception as e:
            logging.warning(f"Gemini translate_batch fast-timeout/error, falling back: {e}")
            self._trip_circuit(cooldown_seconds=30.0)

        missing = [t for t in targets if not out.get(t)]
        if missing:
            fallback_res = await self._fallback.translate_batch(text, source=source, targets=missing, glossary=glossary)
            out.update(fallback_res)

        return out


class DisabledTranslationProvider(TranslationProvider):
    """
    Disabled / Passthrough translation provider (0.0ms latency).
    Returns the original recognized text directly across all target subtitle streams.
    """
    async def translate(
        self,
        text: str,
        source: str = "auto",
        target: str = "es",
        glossary: Optional[Dict[str, str]] = None,
    ) -> str:
        return apply_glossary(text, glossary, target_lang=target)

    async def translate_batch(
        self,
        text: str,
        source: str = "auto",
        targets: Optional[List[str]] = None,
        glossary: Optional[Dict] = None,
    ) -> Dict[str, str]:
        if not targets:
            targets = ["es"]
        return {t: apply_glossary(text, glossary, target_lang=t) for t in targets}


class GoogleCloudTranslationProvider(TranslationProvider):
    """
    Google Cloud Translation Enterprise v3 provider using Service Account credentials.
    Directly consumes Google Cloud billing / developer credits with ultra-low ~200ms latency.
    """
    def __init__(self, key_path: Optional[str] = None):
        self.key_path = key_path or os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or "credentials/service_account.json"
        self._creds = None
        self.project_id = "85658241516"
        self._fallback = GoogleTranslationProvider()
        if os.path.exists(self.key_path):
            try:
                with open(self.key_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.project_id = data.get("project_id", self.project_id)
                    if "@" in data.get("client_email", ""):
                        proj_from_email = data["client_email"].split("@")[1].split(".")[0]
                        if proj_from_email.isdigit():
                            self.project_id = proj_from_email
                from google.oauth2 import service_account
                self._creds = service_account.Credentials.from_service_account_file(
                    self.key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            except Exception as e:
                logging.warning(f"Failed to initialize GoogleCloudTranslationProvider: {e}")

    def _get_token(self) -> Optional[str]:
        if not self._creds:
            return None
        try:
            if not self._creds.valid:
                from google.auth.transport.requests import Request
                self._creds.refresh(Request())
            return self._creds.token
        except Exception as e:
            logging.warning(f"Error refreshing Google Cloud token: {e}")
            return None

    async def translate(
        self,
        text: str,
        source: str = "auto",
        target: str = "es",
        glossary: Optional[Dict[str, str]] = None,
    ) -> str:
        if not text or not text.strip():
            return ""
        if source and target and source.lower() == target.lower() and source.lower() != "auto":
            return apply_glossary(text, glossary, target_lang=target)

        token = self._get_token()
        if not token:
            return await self._fallback.translate(text, source, target, glossary)

        try:
            import httpx
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            payload = {
                "contents": [text],
                "targetLanguageCode": target,
            }
            if source and source.lower() != "auto":
                payload["sourceLanguageCode"] = source

            url = f"https://translation.googleapis.com/v3/projects/{self.project_id}:translateText"
            async with httpx.AsyncClient(timeout=4.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    translated = data["translations"][0]["translatedText"]
                    return apply_glossary(translated, glossary, target_lang=target)
        except Exception as e:
            logging.warning(f"GoogleCloudTranslationProvider error, falling back: {e}")

        return await self._fallback.translate(text, source, target, glossary)

    async def translate_batch(
        self,
        text: str,
        source: str = "auto",
        targets: Optional[List[str]] = None,
        glossary: Optional[Dict] = None,
    ) -> Dict[str, str]:
        if not text or not text.strip():
            return {t: "" for t in (targets or ["es"])}
        if not targets:
            targets = ["es"]

        out = {}
        needed_targets = []
        for t in targets:
            if source and source.lower() == t.lower() and source.lower() != "auto":
                out[t] = apply_glossary(text, glossary, target_lang=t)
            else:
                needed_targets.append(t)

        if not needed_targets:
            return out

        token = self._get_token()
        if not token:
            fallback_res = await self._fallback.translate_batch(text, source=source, targets=needed_targets, glossary=glossary)
            out.update(fallback_res)
            return out

        import asyncio
        async def _trans_one(tgt: str) -> tuple[str, str]:
            res = await self.translate(text, source=source, target=tgt, glossary=glossary)
            return tgt, res

        results = await asyncio.gather(*[_trans_one(t) for t in needed_targets], return_exceptions=True)
        for item in results:
            if isinstance(item, tuple):
                out[item[0]] = item[1]

        missing = [t for t in targets if not out.get(t)]
        if missing:
            fb = await self._fallback.translate_batch(text, source=source, targets=missing, glossary=glossary)
            out.update(fb)

        return out


def get_translation_provider(provider_name: Optional[str] = None) -> TranslationProvider:
    """
    Factory returning configured translation provider based on argument or TRANSLATION_PROVIDER env var.
    Supports:
      - 'gemini_live' / 'gemini-live' / 'gemini': Gemini Live Provider with automatic Google Translate fallback
      - 'googlecloud': Google Cloud Translation v3
      - 'googletrans': Google Translate fallback
      - 'none' or 'disabled': Disabled translation (passthrough)
    """
    prov = (provider_name or os.getenv("TRANSLATION_PROVIDER", "gemini_live")).strip().lower()
    if prov in ("none", "disabled", "off", "desactivada", "passthrough"):
        return DisabledTranslationProvider()

    if prov in ("googlecloud", "cloud"):
        sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials/service_account.json")
        if os.path.exists(sa_path):
            return GoogleCloudTranslationProvider(key_path=sa_path)

    gemini_live_enabled = os.getenv("GEMINI_LIVE_ENABLED", "true").strip().lower() in ("true", "1", "yes")

    if prov in ("gemini", "gemini_live", "gemini-live", "gemini-3.5-live", "gemini-3.6-flash") or gemini_live_enabled:
        from app.gemini_live import GeminiLiveProvider
        return GeminiLiveProvider()

    sa_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "credentials/service_account.json")
    if os.path.exists(sa_path):
        return GoogleCloudTranslationProvider(key_path=sa_path)

    return GoogleTranslationProvider()
