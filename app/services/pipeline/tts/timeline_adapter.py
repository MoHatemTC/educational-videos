"""app/services/pipeline/tts/timeline_adapter.py — Convert synced output into the shared Timeline.

#15 resolution: TimelineSyncer's duration-driven event stretching produces
NarrationSegment / MasterTimeline objects (seconds, free-form event_type strings).
David's renderer (#22) expects the shared app/core/schemas.py Timeline
(ms-based, discriminated union: TypeEvent | RunEvent | HighlightEvent | ScrollEvent).

This module is the single conversion seam. Call master_timeline_to_shared()
after TimelineSyncer.build_master_timeline() — never pass MasterTimeline
or NarrationSegment directly to the renderer.

CONFIRMED GAPS (discussed with David — see #15 PR comment):
  - "pause" events: dropped by design — timing gaps are implicit in the ms
    ranges between consecutive events. Schema change needed if renderer requires
    explicit pause markers.
  - "type_char": consecutive per-char events are merged into one TypeEvent
    spanning first-char-timestamp to next-event-timestamp (or +0.1s buffer).
  - "clear": no shared-schema equivalent — raises loudly so it fails fast.
  - RunEvent / ScrollEvent: never emitted by NarrationSegment.events today.
"""

from __future__ import annotations

from typing import List

from app.core.schemas import (
    HighlightEvent,
    Timeline,
)
from app.core.schemas import TimelineEvent as SharedTimelineEvent
from app.core.schemas import (
    TypeEvent,
)
from app.services.pipeline.tts.timeline_sync import (
    MasterTimeline,
    NarrationSegment,
)

_TRAILING_TYPE_BUFFER_S = 0.1


def _to_ms(seconds: float) -> int:
    return max(0, round(seconds * 1000))


def _segment_to_shared_events(
    segment: NarrationSegment,
) -> List[SharedTimelineEvent]:
    """Convert one synced NarrationSegment's events into shared-schema events.

    Timestamps are offset by segment.start_offset so results are on the
    master timeline's absolute clock.
    """
    out: List[SharedTimelineEvent] = []
    offset = segment.start_offset
    events = segment.events
    i = 0
    n = len(events)

    while i < n:
        ev = events[i]

        if ev.event_type == "pause":
            i += 1
            continue

        if ev.event_type == "clear":
            raise ValueError(
                f"segment {segment.segment_id}: 'clear' event has no shared-schema "
                "mapping — confirm with David before adapting."
            )

        if ev.event_type == "type_char":
            run_start = ev.timestamp
            code_chars = [ev.payload.get("char", "")]
            j = i + 1
            while j < n and events[j].event_type == "type_char":
                code_chars.append(events[j].payload.get("char", ""))
                j += 1

            run_end_source = events[j].timestamp if j < n else run_start + _TRAILING_TYPE_BUFFER_S
            if run_end_source <= run_start:
                run_end_source = run_start + _TRAILING_TYPE_BUFFER_S

            out.append(
                TypeEvent(
                    event_type="type",
                    start_ms=_to_ms(offset + run_start),
                    end_ms=_to_ms(offset + run_end_source),
                    code="".join(code_chars),
                )
            )
            i = j
            continue

        if ev.event_type == "highlight_line":
            line = ev.payload.get("line")
            if line is None:
                raise ValueError(f"segment {segment.segment_id}: highlight_line missing 'line' in payload")
            duration = ev.duration if ev.duration is not None else 0.001
            out.append(
                HighlightEvent(
                    event_type="highlight",
                    start_ms=_to_ms(offset + ev.timestamp),
                    end_ms=_to_ms(offset + ev.timestamp + duration),
                    start_line=int(line),
                    end_line=int(line),
                )
            )
            i += 1
            continue

        raise ValueError(
            f"segment {segment.segment_id}: unmapped event_type '{ev.event_type}' — "
            "no shared-schema equivalent defined in timeline_adapter.py"
        )

    return out


def master_timeline_to_shared(master: MasterTimeline) -> Timeline:
    """Convert a synced MasterTimeline into the shared Timeline schema.

    This is the function David's renderer-facing code should call —
    never pass MasterTimeline or NarrationSegment directly to the renderer.

    Args:
        master: Completed MasterTimeline from TimelineSyncer.build_master_timeline().

    Returns:
        Validated Timeline ready for the renderer.
    """
    all_events: List[SharedTimelineEvent] = []
    for segment in master.segments:
        all_events.extend(_segment_to_shared_events(segment))

    all_events.sort(key=lambda e: e.start_ms)
    return Timeline(events=all_events)
