"""tts/tts_client.py — Multi-lingual TTS client.

Wraps ElevenLabs (>=1.0 SDK) with:
  • Language-aware voice selection (including Arabic RTL)
  • Disk-based audio cache (SHA-256 keyed)
  • Exponential back-off on rate-limit / server errors (tenacity)
  • Async + sync interfaces
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from pathlib import Path
from typing import (
    Dict,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

# ElevenLabs voice IDs per language code.
# Override via ELEVENLABS_VOICE_<LANG_CODE> env vars.
DEFAULT_VOICE_MAP: Dict[str, str] = {
    "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel — English
    "ar": "ErXwobaYiN019PkySvjV",  # Antoni — supports Arabic
    "fr": "MF3mGyEYCl7XYWbV9V6O",  # Elli — French
    "de": "AZnzlk1XvdvUeBnXmlld",  # Domi — German
    "es": "EXAVITQu4vr4xnSDxMaL",  # Bella — Spanish
    "ja": "jBpfuIE2acCO8z3wKNLl",  # Gigi — Japanese
    "zh": "jBpfuIE2acCO8z3wKNLl",  # fallback
}

CACHE_DIR = Path(os.getenv("TTS_CACHE_DIR", "tts_cache"))


class TTSConfig(BaseModel):
    """Configuration settings for the generation pipeline."""

    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""),
        description="ElevenLabs API key (falls back to env var).",
    )
    model_id: str = Field(
        default="eleven_multilingual_v2",
        description="ElevenLabs model to use.",
    )
    cache_dir: str = Field(default=str(CACHE_DIR))
    max_retries: int = Field(default=5)
    output_format: str = Field(default="mp3_44100_128")

    class Config:
        """Configuration settings for the generation pipeline."""

        populate_by_name = True


class TTSClient:
    """Configuration settings for the generation pipeline."""

    def __init__(self, config: Optional[TTSConfig] = None) -> None:
        """Async/sync ElevenLabs TTS client with caching and retry."""
        self.config = config or TTSConfig()
        self._cache_dir = Path(self.config.cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._el_client = None

    def synthesize(
        self,
        text: str,
        lang_code: str = "en",
        voice_id: Optional[str] = None,
    ) -> Path:
        """Synthesize *text* → MP3 file path.

        Returns cached file if already generated.
        """
        voice = voice_id or self._resolve_voice(lang_code)
        cache_key = self._cache_key(text, voice)
        cached = self._cache_dir / f"{cache_key}.mp3"
        if cached.exists():
            logger.debug("TTS cache hit: %s", cache_key)
            return cached

        logger.info("Synthesizing TTS [%s] voice=%s len=%d chars", lang_code, voice, len(text))
        audio_bytes = self._call_api_with_retry(text, voice)
        cached.write_bytes(audio_bytes)
        logger.info("Saved TTS audio → %s (%d bytes)", cached, len(audio_bytes))
        return cached

    async def synthesize_async(
        self,
        text: str,
        lang_code: str = "en",
        voice_id: Optional[str] = None,
    ) -> Path:
        """Async wrapper — runs synthesis in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.synthesize, text, lang_code, voice_id)

    def _resolve_voice(self, lang_code: str) -> str:
        env_key = f"ELEVENLABS_VOICE_{lang_code.upper()}"
        if env_key in os.environ:
            return os.environ[env_key]
        return DEFAULT_VOICE_MAP.get(lang_code, DEFAULT_VOICE_MAP["en"])

    @staticmethod
    def _cache_key(text: str, voice: str) -> str:
        digest = hashlib.sha256(f"{voice}::{text}".encode("utf-8")).hexdigest()
        return digest[:32]

    def _get_el_client(self):
        if self._el_client is None:
            try:
                from elevenlabs import ElevenLabs  # type: ignore

                self._el_client = ElevenLabs(api_key=self.config.api_key or None)
            except ImportError:
                raise RuntimeError("elevenlabs package not installed. Run: pip install elevenlabs>=1.0")
        return self._el_client

    def _call_api_with_retry(self, text: str, voice_id: str) -> bytes:
        """Call ElevenLabs with exponential back-off on retriable errors."""
        return self._retry_call(text, voice_id)

    @retry(
        retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _retry_call(self, text: str, voice_id: str) -> bytes:
        client = self._get_el_client()
        try:
            audio_generator = client.text_to_speech.convert(
                voice_id=voice_id,
                text=text,
                model_id=self.config.model_id,
                output_format=self.config.output_format,
            )
            # SDK may return a generator or bytes
            if hasattr(audio_generator, "__iter__") and not isinstance(audio_generator, bytes):
                chunks = list(audio_generator)
                return b"".join(chunks)
            return audio_generator  # type: ignore[return-value]
        except Exception as exc:
            err_str = str(exc).lower()
            # Rate limit → retriable
            if "rate" in err_str or "429" in err_str or "too many" in err_str:
                logger.warning("ElevenLabs rate limit hit, will retry: %s", exc)
                raise ConnectionError(f"Rate limited: {exc}") from exc
            # Server errors → retriable
            if "500" in err_str or "502" in err_str or "503" in err_str:
                logger.warning("ElevenLabs server error, will retry: %s", exc)
                raise ConnectionError(f"Server error: {exc}") from exc
            # Auth / client errors → not retriable
            raise


class StubTTSClient(TTSClient):
    """Returns a minimal silent MP3 without calling ElevenLabs.

    Used in unit tests and CI environments with no API key.
    """

    _SILENT_MP3 = bytes.fromhex(
        "494433030000000000"  # ID3v2.3 header (9 bytes)
        "fffb9000" + "00" * 413  # MPEG frame header  # silent frame payload
    )

    def _call_api_with_retry(self, text: str, voice_id: str) -> bytes:
        logger.info("[StubTTS] returning silent MP3 for voice=%s", voice_id)
        # Vary length slightly so different texts get different hashes
        pad = bytes(len(text) % 64)
        return self._SILENT_MP3 + pad
