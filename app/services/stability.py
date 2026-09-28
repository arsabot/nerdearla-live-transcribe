import re

class StabilityEngine:
    """
    Analyzes a stream of interim transcripts and extracts stable prefixes.
    When a prefix of the words no longer changes over consecutive updates,
    it is considered 'stable' and can be safely sent for translation.
    """
    def __init__(self, min_stable_words=1, min_stable_chars=4):
        self.min_stable_words = min_stable_words
        self.min_stable_chars = min_stable_chars
        self.last_text = ""
        self.emitted_text = ""
        
    def _get_words(self, text):
        return [w for w in text.split(" ") if w.strip()]
        
    def update(self, interim_text: str) -> str:
        """
        Updates the engine with a new interim transcript.
        Returns the new 'stable' chunk of text that was identified, or an empty string.
        """
        remaining_interim = interim_text
        if self.emitted_text and interim_text.startswith(self.emitted_text):
            remaining_interim = interim_text[len(self.emitted_text):].lstrip()
            
        remaining_last = self.last_text
        if self.emitted_text and self.last_text.startswith(self.emitted_text):
            remaining_last = self.last_text[len(self.emitted_text):].lstrip()
            
        words_last = self._get_words(remaining_last)
        words_new = self._get_words(remaining_interim)
        
        common_words = []
        for w_l, w_n in zip(words_last, words_new):
            if w_l.lower() == w_n.lower():
                common_words.append(w_n)
            else:
                break
                
        new_stable = ""
        if len(common_words) >= self.min_stable_words:
            # Drop the very last common word to be safe against partials
            safe_words = common_words[:-1] if len(common_words) > self.min_stable_words else common_words
            new_stable = " ".join(safe_words)
            
            if len(new_stable) >= self.min_stable_chars:
                if self.emitted_text:
                    self.emitted_text += " " + new_stable
                else:
                    self.emitted_text = new_stable
                self.last_text = interim_text
                return new_stable
                
        self.last_text = interim_text
        return ""
        
    def get_unemitted(self, final_text: str) -> str:
        if self.emitted_text and final_text.startswith(self.emitted_text):
            return final_text[len(self.emitted_text):].lstrip()
        return final_text
        
    def reset(self):
        self.last_text = ""
        self.emitted_text = ""
