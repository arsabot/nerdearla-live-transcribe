"""
Semantic Chunker for real-time speech and continuous live audio.
Splits continuous transcripts (e.g. YouTube videos, keynotes, fast speakers)
into natural, readable subtitle chunks (<= 3 lines / ~12-18 words)
without cutting mid-word or breaking phrases awkwardly.
"""

from typing import Optional

SPLIT_CONNECTORS = {
    "es": [
        ". ", "! ", "? ", "; ", ": ",
        ", ", " — ", " - ",
        " pero ", " porque ", " aunque ", " entonces ", " y ", " o ", " por lo tanto ",
        " además ", " cuando ", " donde ", " ya que ", " mientras ", " de hecho ", " así que "
    ],
    "en": [
        ". ", "! ", "? ", "; ", ": ",
        ", ", " — ", " - ",
        " but ", " because ", " although ", " so ", " and ", " or ", " however ",
        " therefore ", " which ", " when ", " where ", " while ", " since ", " in fact ", " as well as "
    ],
    "pt": [
        ". ", "! ", "? ", "; ", ": ",
        ", ", " — ", " - ",
        " mas ", " porque ", " embora ", " então ", " e ", " ou ", " portanto ",
        " além disso ", " quando ", " onde ", " já que ", " enquanto ", " assim como "
    ],
}


class SemanticChunker:
    """
    Monitors streaming interim transcripts and cuts long continuous speech
    at natural semantic boundaries.
    """

    def __init__(self, max_words: int = 18, soft_limit_words: int = 12, min_prefix_words: int = 6):
        self.max_words = max_words
        self.soft_limit_words = soft_limit_words
        self.min_prefix_words = min_prefix_words
        self.committed_text = ""

    def process_interim(self, full_text: str, lang: str = "es") -> tuple[Optional[str], str]:
        """
        Processes an interim transcript from continuous speech.

        Returns:
            (chunk_to_commit, remaining_interim_text)
            chunk_to_commit is None if no chunk boundary was reached.
        """
        clean_full = full_text.strip()
        if not clean_full:
            return None, ""

        # Extract remaining text not yet committed
        remaining = clean_full
        if self.committed_text:
            if clean_full.startswith(self.committed_text):
                remaining = clean_full[len(self.committed_text):].lstrip()
            elif clean_full.lower().startswith(self.committed_text.lower()):
                remaining = clean_full[len(self.committed_text):].lstrip()

        words = remaining.split()
        if len(words) < self.soft_limit_words:
            return None, remaining

        # Look for natural linguistic connectors / punctuation
        lang_key = lang.lower().split("-")[0] if lang else "en"
        connectors = SPLIT_CONNECTORS.get(lang_key, SPLIT_CONNECTORS["en"])

        best_cut_idx = -1
        best_conn_len = 0

        for conn in connectors:
            idx = remaining.rfind(conn)
            if idx != -1:
                candidate_prefix = remaining[:idx + len(conn)].strip()
                candidate_suffix = remaining[idx + len(conn):].strip()
                prefix_words = candidate_prefix.split()
                suffix_words = candidate_suffix.split()
                if len(prefix_words) >= self.min_prefix_words and len(suffix_words) >= 2:
                    if idx > best_cut_idx:
                        best_cut_idx = idx
                        best_conn_len = len(conn)

        if best_cut_idx != -1:
            prefix = remaining[:best_cut_idx + best_conn_len].strip()
            suffix = remaining[best_cut_idx + best_conn_len:].strip()
            self.committed_text = (self.committed_text + " " + prefix).strip() if self.committed_text else prefix
            return prefix, suffix

        # Hard limit reached without finding a connector: cut at safe word boundary leaving 3 words cushion
        if len(words) >= self.max_words:
            cut_point = max(self.min_prefix_words, len(words) - 3)
            prefix = " ".join(words[:cut_point])
            suffix = " ".join(words[cut_point:])
            self.committed_text = (self.committed_text + " " + prefix).strip() if self.committed_text else prefix
            return prefix, suffix

        return None, remaining

    def process_final(self, full_text: str) -> str:
        """
        Processes the final transcript, returns uncommitted remainder, and resets state.
        """
        clean_full = full_text.strip()
        remaining = clean_full
        if self.committed_text:
            if clean_full.startswith(self.committed_text):
                remaining = clean_full[len(self.committed_text):].lstrip()
            elif clean_full.lower().startswith(self.committed_text.lower()):
                remaining = clean_full[len(self.committed_text):].lstrip()

        self.reset()
        return remaining

    def reset(self) -> None:
        self.committed_text = ""
