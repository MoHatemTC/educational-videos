"""ElevenLabs text-to-speech client with cache, retry, and voice controls.

Synthesizes narration to MP3. Results are cached on disk keyed by text, voice,
model, and voice settings. Segment synthesis lets callers vary stability, style,
voice, and provider-neutral emotion metadata across narration segments.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, cast

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.logging import logger

_API_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
_TRANSIENT_STATUS = {429, 500, 502, 503, 504}
_SEGMENT_CONCAT_TIMEOUT_S = 300


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


def default_voice_settings() -> dict[str, Any]:
    """Return configured default voice settings, including provider-neutral emotion."""
    voice_settings: dict[str, Any] = {
        "stability": settings.ELEVENLABS_VOICE_STABILITY,
        "similarity_boost": settings.ELEVENLABS_VOICE_SIMILARITY_BOOST,
        "style": settings.ELEVENLABS_VOICE_STYLE,
        "use_speaker_boost": settings.ELEVENLABS_USE_SPEAKER_BOOST,
    }
    if settings.ELEVENLABS_VOICE_EMOTION:
        voice_settings["emotion"] = settings.ELEVENLABS_VOICE_EMOTION
    return voice_settings


def _merged_voice_settings(voice_settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge user voice settings over configured defaults."""
    merged = default_voice_settings()
    if voice_settings:
        for key, value in voice_settings.items():
            if value is not None:
                merged[key] = value
    return merged


def _provider_voice_settings(voice_settings: dict[str, Any]) -> dict[str, Any]:
    """Return only ElevenLabs-supported voice_settings fields."""
    allowed = {"stability", "similarity_boost", "style", "use_speaker_boost"}
    return {key: value for key, value in voice_settings.items() if key in allowed}


def _cache_key(payload: dict[str, Any]) -> str:
    """Return a deterministic hash for cacheable TTS inputs."""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_path(text: str, voice_id: str, model_id: str, voice_settings: dict[str, Any] | None = None) -> Path:
    """Deterministic cache path for a single TTS request."""
    digest = _cache_key(
        {"kind": "single", "text": text, "voice_id": voice_id, "model_id": model_id, "settings": voice_settings or {}}
    )
    return _cache_dir() / f"{digest}.mp3"


def _segments_cache_path(segments: list[dict[str, Any]], model_id: str) -> Path:
    """Deterministic cache path for a concatenated segmented narration."""
    digest = _cache_key({"kind": "segments", "segments": segments, "model_id": model_id})
    return _cache_dir() / f"{digest}.mp3"


def _ffmpeg_executable() -> str:
    """Return a usable FFmpeg executable path."""
    try:
        module = cast(Any, __import__("imageio_ffmpeg", fromlist=["get_ffmpeg_exe"]))
        return str(module.get_ffmpeg_exe())
    except Exception as exc:  # noqa: BLE001 - fallback to PATH ffmpeg
        logger.warning("bundled_ffmpeg_unavailable", error=str(exc))
        return "ffmpeg"


@retry(
    retry=retry_if_exception_type(TransientTTSError),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _request_audio(text: str, voice_id: str, model_id: str, voice_settings: dict[str, Any]) -> bytes:
    """POST to ElevenLabs and return MP3 bytes; raise on failure."""
    headers = {
        "xi-api-key": settings.ELEVENLABS_API_KEY,
        "accept": "audio/mpeg",
        "content-type": "application/json",
    }
    body = {
        "text": text,
        "model_id": model_id,
        "voice_settings": _provider_voice_settings(voice_settings),
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


def synthesize(
    text: str,
    voice_id: str | None = None,
    model_id: str | None = None,
    voice_settings: dict[str, Any] | None = None,
) -> Path:
    """Synthesize ``text`` to an MP3 file and return its path (cached).

    Args:
        text: Narration text.
        voice_id: ElevenLabs voice id (defaults to the configured narrator).
        model_id: ElevenLabs model (defaults to ``eleven_multilingual_v2``).
        voice_settings: Optional stability/style/speaker/emotion controls. The
            provider-neutral ``emotion`` field is kept in cache metadata but is
            not sent as an unsupported ElevenLabs parameter.

    Returns:
        Path to the MP3 file (existing cache hit or freshly written).
    """
    voice_id = voice_id or settings.ELEVENLABS_VOICE_ID
    model_id = model_id or settings.ELEVENLABS_MODEL
    merged_settings = _merged_voice_settings(voice_settings)
    path = _cache_path(text, voice_id, model_id, merged_settings)

    if path.is_file() and path.stat().st_size > 0:
        logger.info("tts_cache_hit", path=str(path))
        return path

    audio = _request_audio(text, voice_id, model_id, merged_settings)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(audio)
    tmp.rename(path)
    logger.info(
        "tts_synthesized",
        path=str(path),
        bytes=len(audio),
        voice_id=voice_id,
        model_id=model_id,
        emotion=merged_settings.get("emotion"),
    )
    return path


def synthesize_segments(
    segments: list[dict[str, Any]],
    *,
    default_voice_id: str,
    default_language: str,
    model_id: str | None = None,
) -> Path:
    """Synthesize and concatenate narration segments with per-segment controls."""
    model_id = model_id or settings.ELEVENLABS_MODEL
    normalized_segments: list[dict[str, Any]] = []
    part_paths: list[Path] = []

    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        language = str(segment.get("language") or default_language)
        voice_id = str(segment.get("voice_id") or default_voice_id or voice_id_for_language(language))
        voice_settings = segment.get("voice_settings") if isinstance(segment.get("voice_settings"), dict) else None
        normalized = {
            "text": text,
            "voice_id": voice_id,
            "language": language,
            "voice_settings": _merged_voice_settings(voice_settings),
        }
        normalized_segments.append(normalized)
        part_paths.append(synthesize(text, voice_id=voice_id, model_id=model_id, voice_settings=voice_settings))

    if not part_paths:
        raise TTSError("tts_segments did not include any non-empty text")
    if len(part_paths) == 1:
        return part_paths[0]

    output_path = _segments_cache_path(normalized_segments, model_id)
    if output_path.is_file() and output_path.stat().st_size > 0:
        logger.info("tts_segment_cache_hit", path=str(output_path), segments=len(part_paths))
        return output_path

    concat_file = output_path.with_suffix(".concat.txt")
    concat_file.write_text("".join(f"file '{path.as_posix()}'\n" for path in part_paths), encoding="utf-8")
    cmd = [
        _ffmpeg_executable(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=_SEGMENT_CONCAT_TIMEOUT_S)
    concat_file.unlink(missing_ok=True)
    if result.returncode != 0:
        raise TTSError(f"ffmpeg failed to concatenate TTS segments: {result.stderr[-400:]}")

    logger.info("tts_segments_synthesized", path=str(output_path), segments=len(part_paths))
    return output_path
