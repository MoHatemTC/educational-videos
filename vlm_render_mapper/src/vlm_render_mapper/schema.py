"""
Pydantic v2 models for the VLM Render Plan schema.
These models mirror render_plan_schema.json and provide validation + serialization.

Schema contract changes (v1.1.0):
- RenderMetadata: `created_at` renamed to `recorded_at`; `total_duration_ms` is int
- Frame: `timestamp_ms` and `duration_ms` are int (milliseconds, no sub-ms precision)
- CursorKeyframe: `timestamp_ms` and `duration_ms` are int
- Caption / Transition / Timeline: all *_ms fields are int
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from datetime import datetime

import jsonschema
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    HOVER = "hover"
    SCROLL = "scroll"
    DRAG = "drag"
    TYPE = "type"
    KEY_PRESS = "key_press"
    NAVIGATE = "navigate"
    PAGE_LOAD = "page_load"
    SCREENSHOT = "screenshot"
    WAIT = "wait"
    FOCUS = "focus"
    BLUR = "blur"


class EasingType(str, Enum):
    LINEAR = "linear"
    EASE_IN = "ease_in"
    EASE_OUT = "ease_out"
    EASE_IN_OUT = "ease_in_out"
    SPRING = "spring"
    SMOOTHSTEP = "smoothstep"  # t² * (3 − 2t)


class CursorStyle(str, Enum):
    DEFAULT = "default"
    POINTER = "pointer"
    TEXT = "text"
    CROSSHAIR = "crosshair"
    GRAB = "grab"
    GRABBING = "grabbing"


class CaptionPosition(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"


class TransitionType(str, Enum):
    FADE = "fade"
    CUT = "cut"
    DISSOLVE = "dissolve"
    WIPE_LEFT = "wipe_left"
    WIPE_RIGHT = "wipe_right"
    ZOOM_IN = "zoom_in"
    ZOOM_OUT = "zoom_out"
    SLIDE = "slide"


class RenderTarget(str, Enum):
    FFMPEG = "ffmpeg"
    REMOTION = "remotion"
    BOTH = "both"


class Modifier(str, Enum):
    CTRL = "ctrl"
    SHIFT = "shift"
    ALT = "alt"
    META = "meta"


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


class Point(BaseModel):
    x: float
    y: float


class Resolution(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class Action(BaseModel):
    type: ActionType
    target: Optional[str] = None
    value: Optional[Any] = None
    coordinates: Optional[Point] = None
    modifiers: list[Modifier] = Field(default_factory=list)
    scroll_delta: Optional[Point] = None
    drag_start: Optional[Point] = None
    drag_end: Optional[Point] = None


# ---------------------------------------------------------------------------
# Cursor
# ---------------------------------------------------------------------------


class CursorState(BaseModel):
    x: float
    y: float
    visible: bool = True
    style: CursorStyle = CursorStyle.DEFAULT


class CursorKeyframe(BaseModel):
    """A single waypoint in the cursor path.

    ``timestamp_ms`` is the segment-relative integer offset from the start
    of the containing frame segment (not session-absolute).  ``duration_ms``
    is the travel time from the *previous* keyframe, also an integer.
    """

    timestamp_ms: int = Field(ge=0)
    x: float
    y: float
    easing: EasingType = EasingType.SMOOTHSTEP
    duration_ms: int = Field(ge=0, default=0)


# ---------------------------------------------------------------------------
# Visual Regions
# ---------------------------------------------------------------------------


class HighlightRegion(BaseModel):
    region_id: str
    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)
    color: str = "#FFD700"
    opacity: float = Field(ge=0, le=1, default=0.4)
    border_radius: float = Field(ge=0, default=4.0)
    border_color: Optional[str] = None
    border_width: Optional[float] = None
    label: Optional[str] = None


class ZoomRegion(BaseModel):
    x: float
    y: float
    width: float = Field(ge=0)
    height: float = Field(ge=0)
    scale: float = Field(ge=1.0)
    easing: EasingType = EasingType.EASE_IN_OUT
    transition_duration_ms: int = Field(ge=0, default=300)


# ---------------------------------------------------------------------------
# Captions
# ---------------------------------------------------------------------------


class CaptionStyle(BaseModel):
    font_size: int = 24
    font_family: str = "Arial"
    color: str = "#FFFFFF"
    background_color: str = "#000000"
    background_opacity: float = Field(ge=0, le=1, default=0.7)
    padding: int = 8


class Caption(BaseModel):
    caption_id: str
    text: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    position: CaptionPosition = CaptionPosition.BOTTOM
    style: CaptionStyle = Field(default_factory=CaptionStyle)

    @model_validator(mode="after")
    def end_after_start(self) -> Caption:
        if self.end_ms <= self.start_ms:
            raise ValueError(f"end_ms ({self.end_ms}) must be > start_ms ({self.start_ms})")
        return self


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


class Transition(BaseModel):
    transition_id: str
    type: TransitionType
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    from_frame_id: Optional[str] = None
    to_frame_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Frames  (FrameDescriptor in the spec)
# ---------------------------------------------------------------------------


class FrameDescriptor(BaseModel):
    """Per-frame descriptor matching the contract spec.

    Fields
    ------
    frame_id            : unique string identifier, e.g. ``"frame_000001"``
    source_action_index : 0-based index into the original session event list
    start_ms            : integer ms offset from session start (t=0)
    duration_ms         : integer display duration in ms
    screenshot_ref      : path or ref to the source screenshot
    action              : the browser action that produced this frame
    cursor              : optional cursor state at the moment of capture
    highlight_regions   : visual attention overlays
    zoom_region         : optional camera-zoom descriptor
    caption_ids         : ids of ``Caption`` objects that are active this frame
    """

    frame_id: str
    source_action_index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    duration_ms: int = Field(ge=0)
    screenshot_ref: str
    action: Action
    cursor: Optional[CursorState] = None
    highlight_regions: list[HighlightRegion] = Field(default_factory=list)
    zoom_region: Optional[ZoomRegion] = None
    caption_ids: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------
    # Back-compat shim: callers reading .timestamp_ms continue to work.
    # ------------------------------------------------------------------
    @property
    def timestamp_ms(self) -> int:
        return self.start_ms


# Keep the old name as an alias so existing internal code that imports
# ``Frame`` continues to work during the transition.
Frame = FrameDescriptor


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


class TimelineSegment(BaseModel):
    segment_id: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    frame_ids: list[str]
    label: Optional[str] = None
    speed_multiplier: float = Field(ge=0.1, default=1.0)

    @model_validator(mode="after")
    def end_after_start(self) -> TimelineSegment:
        if self.end_ms <= self.start_ms:
            raise ValueError(f"end_ms ({self.end_ms}) must be > start_ms ({self.start_ms})")
        return self


class Timeline(BaseModel):
    segments: list[TimelineSegment]
    total_duration_ms: int = Field(ge=0, default=0)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class RenderMetadata(BaseModel):
    """Top-level session metadata.

    ``recorded_at`` (formerly ``created_at``) is the UTC datetime at which
    the render plan was generated.  ``total_duration_ms`` is the integer
    sum of all frame durations.
    """

    session_id: str
    recorded_at: datetime  # renamed from created_at
    total_duration_ms: int = Field(ge=0)
    frame_rate: int = Field(ge=1, le=120)
    resolution: Resolution
    source_session_file: Optional[str] = None
    render_target: RenderTarget = RenderTarget.FFMPEG


class Viewport(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    device_pixel_ratio: float = 1.0


# ---------------------------------------------------------------------------
# Root Document
# ---------------------------------------------------------------------------


class RenderPlan(BaseModel):
    version: str = "1.0.0"
    metadata: RenderMetadata
    viewport: Viewport
    frames: list[FrameDescriptor] = Field(min_length=1)
    timeline: Timeline
    cursor_path: list[CursorKeyframe] = Field(default_factory=list)
    captions: list[Caption] = Field(default_factory=list)
    transitions: list[Transition] = Field(default_factory=list)

    def frame_by_id(self, frame_id: str) -> Optional[FrameDescriptor]:
        for f in self.frames:
            if f.frame_id == frame_id:
                return f
        return None

    def caption_by_id(self, caption_id: str) -> Optional[Caption]:
        for c in self.captions:
            if c.caption_id == caption_id:
                return c
        return None


# ---------------------------------------------------------------------------
# Schema validation  (moved here from cli.py to keep responsibilities clean)
# ---------------------------------------------------------------------------


def _load_json_schema() -> dict:
    """Locate and load render_plan_schema.json.

    Search order:
    1. ``<repo_root>/schemas/render_plan_schema.json``  (new canonical location)
    2. ``<repo_root>/render_plan_schema.json``           (legacy flat location)
    3. Parent dirs (editable-install fallback)
    """
    candidates = [
        Path(__file__).parent.parent.parent.parent / "schemas" / "render_plan_schema.json",
        Path(__file__).parent.parent.parent.parent / "render_plan_schema.json",
        Path(__file__).parent.parent.parent / "render_plan_schema.json",
        Path(__file__).parent / "render_plan_schema.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        "render_plan_schema.json not found. Searched: " + ", ".join(str(c) for c in candidates)
    )


def validate_against_json_schema(plan_dict: dict) -> None:
    """Validate *plan_dict* against render_plan_schema.json.

    Raises ``jsonschema.ValidationError`` on failure so the caller can
    decide whether to abort or warn.
    """
    schema = _load_json_schema()
    jsonschema.validate(instance=plan_dict, schema=schema)


def validate_render_plan(plan: "RenderPlan") -> None:
    """Full validation: Pydantic structural check + JSON Schema contract.

    Raises on the first failure encountered.
    """
    RenderPlan.model_validate(plan.model_dump())
    validate_against_json_schema(json.loads(plan.model_dump_json()))


# ---------------------------------------------------------------------------
# Deterministic session_id derivation from log content
# ---------------------------------------------------------------------------


def session_id_from_events(events: list) -> str:
    """Derive a deterministic session ID from raw event list content.

    Builds a SHA-256 digest over the timestamp/action/screenshot sequence
    so the same log file always produces the same session ID.
    """
    import hashlib

    parts = []
    for ev in events:
        ts = ev.get("timestamp", ev.get("ts", ev.get("time", "")))
        action = ev.get("action", ev.get("type", ev.get("event_type", "")))
        screenshot = (
            ev.get("screenshot") or ev.get("screenshot_path") or ev.get("screenshotRef") or ""
        )
        parts.append(f"{ts}|{action}|{screenshot}")

    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
