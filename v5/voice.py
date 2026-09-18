"""Voice engine contracts. Keeps speech separate from reasoning so STT/TTS can be swapped later."""
import os
from typing import Any, Dict

VOICE_LOCALES = {"en": "English", "ml": "Malayalam", "hi": "Hindi", "ta": "Tamil", "kn": "Kannada"}

class VoiceEngine:
    def __init__(self):
        self.stt_provider = os.getenv("V5_STT_PROVIDER", "browser")
        self.tts_provider = os.getenv("V5_TTS_PROVIDER", "browser")
        self.default_locale = os.getenv("V5_VOICE_LOCALE", "en-IN")

    def capabilities(self) -> Dict[str, Any]:
        return {
            "stt": self.stt_provider,
            "tts": self.tts_provider,
            "locales": VOICE_LOCALES,
            "interruptible": True,
            "bargeIn": True,
            "streaming": True,
        }

    def prepare(self, text: str, locale: str = "") -> Dict[str, Any]:
        return {
            "text": text,
            "locale": locale or self.default_locale,
            "provider": self.tts_provider,
            "interruptible": True,
            "ssml": False,
        }
