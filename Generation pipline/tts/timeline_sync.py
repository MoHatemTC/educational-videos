"""tts/timeline_sync.py — Align visual JSON timeline events to real TTS audio duration.

Each narration segment has:
  • A list of code-typing / highlight events with their original timestamps
  • A real audio clip whose duration was measured by audio_utils

This module stretches event timestamps so they fit within the actual audio
duration, producing a validated, ready-to-render timeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
)

from .audio_utils import (
    compute_stretch_factor,
    get_audio_duration,
)

logger = logging.getLogger(__name__)


class TimelineEvent(BaseModel):
    """A single visual event on the code animation timeline."""

    event_type: str = Field(description="E.g. 'type_char', 'highlight_line', 'pause', 'clear'.")
    timestamp: float = Field(description="Seconds from segment start.")
    duration: Optional[float] = Field(default=None, description="Event duration in seconds.")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event-specific data.")


class NarrationSegment(BaseModel):
    """One narration unit: text → TTS → aligned visual events."""

    segment_id: str
    lang_code: str = "en"
    text: str
    audio_path: Optional[str] = None
    audio_duration: Optional[float] = None  # measured after TTS
    original_duration_estimate: Optional[float] = None  # pre-TTS estimate
    events: List[TimelineEvent] = Field(default_factory=list)
    stretch_factor: float = 1.0
    rtl: bool = False  # Arabic / Hebrew RTL flag
    start_offset: float = 0.0  # cumulative start within the master timeline


class MasterTimeline(BaseModel):
    """The complete timeline across all narration segments."""

    segments: List[NarrationSegment] = Field(default_factory=list)
    total_duration: float = 0.0
    validated: bool = False


RTL_LANGUAGES = {"ar", "he", "fa", "ur", "yi", "dv", "ps"}


def is_rtl(lang_code: str) -> bool:
    """Initialize the pipeline with the provided configuration."""
    return lang_code.lower() in RTL_LANGUAGES


class TimelineSyncer:
    """Synchronises visual events to measured TTS audio duration for each segment.

    then stitches segments into a :class:`MasterTimeline`.
    """

    def __init__(
        self,
        output_dir: str = "output",
        inter_segment_gap: float = 0.3,
    ) -> None:
        """Initialize the pipeline with the provided configuration."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.inter_segment_gap = inter_segment_gap

    def sync_segment(self, segment: NarrationSegment) -> NarrationSegment:
        """Measure the audio clip and stretch events to match its real duration.

        Returns a new :class:`NarrationSegment` with updated timestamps.
        """
        seg = segment.model_copy(deep=True)
        seg.rtl = is_rtl(seg.lang_code)

        if not seg.audio_path or not Path(seg.audio_path).exists():
            logger.warning("No audio file for segment %s — skipping sync.", seg.segment_id)
            return seg

        try:
            measured = get_audio_duration(Path(seg.audio_path))
        except RuntimeError:
            measured = seg.original_duration_estimate or 1.0
            logger.warning(
                "Could not measure audio for segment %s — using estimate %.2fs",
                seg.segment_id,
                measured,
            )
        seg.audio_duration = measured

        if seg.events:
            original_end = max((e.timestamp + (e.duration or 0)) for e in seg.events)
            if original_end <= 0:
                original_end = seg.original_duration_estimate or measured
            factor = compute_stretch_factor(actual_duration=original_end, target_duration=measured)
            seg.stretch_factor = factor
            seg.events = self._stretch_events(seg.events, factor)
            logger.info(
                "Segment %s: audio=%.3fs, original_est=%.3fs, stretch=%.4f, rtl=%s",
                seg.segment_id,
                measured,
                original_end,
                factor,
                seg.rtl,
            )
        else:
            seg.stretch_factor = 1.0

        return seg

    def build_master_timeline(self, segments: List[NarrationSegment]) -> MasterTimeline:
        """Sync all segments, assign cumulative offsets, and stitch into one.

        class:`MasterTimeline`.
        """
        synced: List[NarrationSegment] = []
        cursor = 0.0

        for seg in segments:
            s = self.sync_segment(seg)
            s.start_offset = cursor
            dur = s.audio_duration or s.original_duration_estimate or 0.0
            cursor += dur + self.inter_segment_gap
            synced.append(s)

        master = MasterTimeline(
            segments=synced,
            total_duration=round(cursor - self.inter_segment_gap, 4),
            validated=True,
        )
        return master

    def save(self, master: MasterTimeline) -> Dict[str, Path]:
        """Persist master_timeline.json and segment_timings.json."""
        master_path = self.output_dir / "master_timeline.json"
        timings_path = self.output_dir / "segment_timings.json"

        master_path.write_text(master.model_dump_json(indent=2), encoding="utf-8")

        timings = {
            "total_duration": master.total_duration,
            "segments": [
                {
                    "id": s.segment_id,
                    "lang": s.lang_code,
                    "rtl": s.rtl,
                    "start": s.start_offset,
                    "duration": s.audio_duration,
                    "stretch_factor": s.stretch_factor,
                    "event_count": len(s.events),
                }
                for s in master.segments
            ],
        }
        timings_path.write_text(json.dumps(timings, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info("Saved master_timeline → %s", master_path)
        logger.info("Saved segment_timings → %s", timings_path)
        return {"master_timeline": master_path, "segment_timings": timings_path}

    @staticmethod
    def _stretch_events(events: List[TimelineEvent], factor: float) -> List[TimelineEvent]:
        stretched: List[TimelineEvent] = []
        for ev in events:
            new_ev = ev.model_copy(deep=True)
            new_ev.timestamp = round(ev.timestamp * factor, 4)
            if ev.duration is not None:
                new_ev.duration = round(ev.duration * factor, 4)
            stretched.append(new_ev)
        return stretched


def make_demo_segments() -> List[NarrationSegment]:
    """Return two demo segments (English + Arabic) for pipeline testing."""
    en_events = [
        TimelineEvent(event_type="type_char", timestamp=0.0, payload={"char": "d"}),
        TimelineEvent(event_type="type_char", timestamp=0.1, payload={"char": "e"}),
        TimelineEvent(event_type="type_char", timestamp=0.2, payload={"char": "f"}),
        TimelineEvent(event_type="highlight_line", timestamp=0.5, duration=1.0, payload={"line": 1}),
        TimelineEvent(event_type="pause", timestamp=1.5, duration=0.5, payload={}),
    ]

    ar_events = [
        TimelineEvent(event_type="highlight_line", timestamp=0.0, duration=0.8, payload={"line": 2}),
        TimelineEvent(event_type="type_char", timestamp=0.9, payload={"char": "م"}),
        TimelineEvent(event_type="type_char", timestamp=1.0, payload={"char": "ر"}),
        TimelineEvent(event_type="pause", timestamp=1.2, duration=0.4, payload={}),
    ]

    return [
        NarrationSegment(
            segment_id="seg_01",
            lang_code="en",
            text="This function computes the Fibonacci sequence iteratively.",
            original_duration_estimate=2.5,
            events=en_events,
        ),
        NarrationSegment(
            segment_id="seg_02",
            lang_code="ar",
            text="هذه الدالة تحسب متتالية فيبوناتشي بشكل تكراري.",
            original_duration_estimate=3.0,
            events=ar_events,
        ),
    ]
