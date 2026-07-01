"""ElevenLabs text-to-speech client with a disk cache and retry/back-off.

Synthesizes narration to MP3. Results are cached on disk keyed by
SHA-256(text + voice + model) and written atomically (temp file + rename), so
re-rendering the same script skips the API call. Transient failures (429/5xx)
are retried with exponential back-off; auth/validation errors fail fast.
"""

import hashlib
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger

_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}


class TTSError(RuntimeError):
    """Non-retryable TTS failure."""


class TTSQuotaExceededError(TTSError):
    """TTS provider quota or subscription limit was reached."""


class TransientTTSError(Exception):
    """Retryable TTS failure (rate limit / 5xx)."""


def _json_detail(response: httpx.Response) -> dict[str, Any]:
    """Return an ElevenLabs error detail object when available."""
    try:
        payload = response.json()
    except ValueError:
        return {}

    detail = payload.get("detail")
    if isinstance(detail, dict):
        return detail

    return {}


def _raise_api_error(response: httpx.Response, voice_id: str) -> None:
    """Raise a clear provider error for an ElevenLabs failure response."""
    detail = _json_detail(response)
    code = str(detail.get("code") or "")
    message = str(detail.get("message") or response.text[:500])
    status = str(detail.get("status") or "")

    if code in {"quota_exceeded", "paid_plan_required"} or status in {"quota_exceeded", "payment_required"}:
        raise TTSQuotaExceededError(f"ElevenLabs quota/plan limit for voice_id={voice_id}: {message}")

    raise TTSError(
        f"ElevenLabs TTS failed with HTTP {response.status_code} for voice_id={voice_id}. "
        f"Provider code={code or 'unknown'}. Response: {response.text[:500]}"
    )


def _cache_dir() -> Path:
    """Return (and create) the on-disk TTS cache directory."""
    path = Path(settings.VIDEO_DATA_DIR) / "tts_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cache_path(text: str, voice_id: str, model_id: str) -> Path:
    """Deterministic cache path for a (text, voice, model) triple."""
    digest = hashlib.sha256(f"{text}|{voice_id}|{model_id}".encode("utf-8")).hexdigest()
    return _cache_dir() / f"{digest}.mp3"


@retry(
    retry=retry_if_exception_type(TransientTTSError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _request_audio(text: str, voice_id: str, model_id: str) -> bytes:
    """POST to ElevenLabs and return MP3 bytes; raise on failure."""
    headers = {
        "xi-api-key": settings.ELEVENLABS_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.0},
    }
    with httpx.Client(timeout=180.0) as client:
        response = client.post(_API_URL.format(voice_id=voice_id), headers=headers, json=body)

    if response.status_code != 200:
        detail = _json_detail(response)
        code = str(detail.get("code") or "")

        if code not in {"quota_exceeded", "paid_plan_required"} and response.status_code in _TRANSIENT_STATUS:
            logger.warning("tts_transient_error", status=response.status_code)
            raise TransientTTSError(f"elevenlabs returned {response.status_code}")

        _raise_api_error(response, voice_id)

    return response.content


def voice_id_for_language(language: str) -> str:
    """Return the configured ElevenLabs voice id for a narration language."""
    if language == "egyptian_arabic":
        return settings.ELEVENLABS_VOICE_ID_EGYPTIAN_ARABIC
    return settings.ELEVENLABS_VOICE_ID_ENGLISH


def synthesize(text: str, voice_id: str | None = None, model_id: str | None = None) -> Path:
    """Synthesize ``text`` to an MP3 file and return its path (cached).

    Args:
        text: Narration text.
        voice_id: ElevenLabs voice id (defaults to the configured narrator).
        model_id: ElevenLabs model (defaults to ``eleven_multilingual_v2``).

    Returns:
        Path to the MP3 file (existing cache hit or freshly written).
    """
    voice_id = voice_id or settings.ELEVENLABS_VOICE_ID
    model_id = model_id or settings.ELEVENLABS_MODEL
    path = _cache_path(text, voice_id, model_id)

    if path.is_file() and path.stat().st_size > 0:
        logger.info("tts_cache_hit", path=str(path))
        return path

    audio = _request_audio(text, voice_id, model_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(audio)
    tmp.rename(path)
    logger.info("tts_synthesized", path=str(path), bytes=len(audio), voice_id=voice_id, model_id=model_id)
    return path
