"""
vlm_render_mapper
=================
Translates VLM browser-action session logs into structured render plan JSON
for downstream programmatic rendering via FFmpeg or Remotion.

Schema v1.1 changes
-------------------
- ``RenderMetadata.created_at`` renamed to ``recorded_at``
- All ``*_ms`` fields are now strict integers (no sub-ms float precision)
- ``Frame`` renamed to ``FrameDescriptor`` (old alias preserved)
- ``CursorKeyframe.timestamp_ms`` is segment-relative (not session-absolute)
- Default cursor easing is ``smoothstep`` (t² * (3 − 2t))
- Last-frame duration capped by ``max(preceding frame durations)``
- ``cli.cmd_map`` validates output with ``jsonschema.validate`` before saving
"""

from vlm_render_mapper.schema import RenderPlan, FrameDescriptor, Frame
from vlm_render_mapper.parser import parse_session_file, parse_session_text
from vlm_render_mapper.mapper import RenderMapper, MapperConfig
from vlm_render_mapper.timing import TimingConfig

__version__ = "1.1.0"
__all__ = [
    "RenderPlan",
    "FrameDescriptor",
    "Frame",  # backwards-compatible alias
    "RenderMapper",
    "MapperConfig",
    "TimingConfig",
    "parse_session_file",
    "parse_session_text",
]
