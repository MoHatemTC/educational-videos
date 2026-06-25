"""TTS package — multi-lingual synthesis, audio measurement, timeline sync."""

from .tts_client import TTSClient, TTSConfig, StubTTSClient
from .audio_utils import get_audio_duration, compute_stretch_factor, adjust_timestamps
from .timeline_sync import (
    TimelineSyncer,
    MasterTimeline,
    NarrationSegment,
    TimelineEvent,
    make_demo_segments,
)

__all__ = [
    "TTSClient",
    "TTSConfig",
    "StubTTSClient",
    "get_audio_duration",
    "compute_stretch_factor",
    "adjust_timestamps",
    "TimelineSyncer",
    "MasterTimeline",
    "NarrationSegment",
    "TimelineEvent",
    "make_demo_segments",
]
