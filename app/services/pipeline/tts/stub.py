"""app/services/pipeline/tts/stub.py — Stub TTS client for tests and CI.

Returns a minimal silent MP3 without calling ElevenLabs. Used wherever
ELEVENLABS_API_KEY is absent (tests, CI, local dev without a key).

#17 integration note:
  - Extracted from island's tts/tts_client.py into its own module so tests
    can import it without pulling in the full ElevenLabs SDK.
  - zh→ja voice bug fixed: the island's DEFAULT_VOICE_MAP mapped "zh" to
    the Japanese voice ID (jBpfuIE2acCO8z3wKNLl) silently labeled "fallback".
    Main's elevenlabs.py (voice_id_for_language) handles language→voice via
    settings and does not have a zh entry — the stub now matches that: no zh
    mapping, unknown languages fall back to English.
  - Cache and synthesize interface match main's elevenlabs.synthesize() API
    so tests can swap one for the other transparently.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

# ── Voice map (stub only — production uses settings.ELEVENLABS_VOICE_ID_*) ──

_VOICE_MAP: dict[str, str] = {
    "en": "21m00Tcm4TlvDq8ikWAM",   # Rachel — English
    "ar": "ErXwobaYiN019PkySvjV",   # Antoni — Arabic
    "fr": "MF3mGyEYCl7XYWbV9V6O",   # Elli — French
    "de": "AZnzlk1XvdvUeBnXmlld",   # Domi — German
    "es": "EXAVITQu4vr4xnSDxMaL",   # Bella — Spanish
    "ja": "jBpfuIE2acCO8z3wKNLl",   # Gigi — Japanese
    # "zh" removed: was silently using the Japanese voice (bug #17).
    # Unknown languages fall back to English below.
}

# Minimal silent MPEG-1 Layer-3 frame (ID3 header + one silent frame).
_SILENT_MP3_BASE = bytes.fromhex(
    "494433030000000000"   # ID3v2.3 header (9 bytes)
    "fffb9000" + "00" * 413  # MPEG frame header + silent payload
)


def voice_id_for_lang(lang_code: str) -> str:
    """Resolve a language code to an ElevenLabs voice ID.

    Falls back to English for any unmapped language code — including 'zh',
    which previously silently used the Japanese voice (fixed in #17).
    """
    return _VOICE_MAP.get(lang_code.lower(), _VOICE_MAP["en"])


def _cache_dir() -> Path:
    """Return (and create) the stub TTS cache directory."""
    path = Path(settings.VIDEO_DATA_DIR) / "tts_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(text: str, voice_id: str) -> Path:
    """Deterministic cache path keyed by (text, voice_id)."""
    digest = hashlib.sha256(f"{voice_id}::{text}".encode("utf-8")).hexdigest()
    return _cache_dir() / f"stub_{digest[:32]}.mp3"


def synthesize_stub(
    text: str,
    voice_id: str | None = None,
    lang_code: str = "en",
) -> Path:
    """Return a cached silent MP3 for the given text + voice (no API call).

    Args:
        text: Narration text (used as cache key — content is ignored).
        voice_id: ElevenLabs voice ID override.
        lang_code: Used to resolve voice_id when voice_id is None.

    Returns:
        Path to an MP3 file containing a minimal silent audio stream.
    """
    resolved_voice = voice_id or voice_id_for_lang(lang_code)
    path = _cache_path(text, resolved_voice)

    if path.is_file() and path.stat().st_size > 0:
        logger.info("stub_tts_cache_hit", path=str(path))
        return path

    # Vary length slightly so different texts get different hashes (important
    # for tests that assert different texts produce different files).
    pad = bytes(len(text) % 64)
    audio = _SILENT_MP3_BASE + pad

    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(audio)
    tmp.rename(path)
    logger.info(
        "stub_tts_synthesized",
        path=str(path),
        voice_id=resolved_voice,
        lang_code=lang_code,
    )
    return path