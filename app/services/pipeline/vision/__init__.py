"""Canonical vision-agent package for the video pipeline.

``app.services.pipeline.vision`` owns URL capture, screenshot description,
web-explainer narration, coordinate actions, and the graduated DOM-independent
recovery layer. The legacy top-level vision-agent folders are kept only as
fenced prototypes.
"""

from app.services.pipeline.vision.actions import capture_page_with_actions
from app.services.pipeline.vision.agent import VisionComputerUseAgent
from app.services.pipeline.vision.browser import capture_page
from app.services.pipeline.vision.describe import describe_screenshots
from app.services.pipeline.vision.recovery import RecoveryManager
from app.services.pipeline.vision.script import generate_web_script

__all__ = [
    "RecoveryManager",
    "VisionComputerUseAgent",
    "capture_page",
    "capture_page_with_actions",
    "describe_screenshots",
    "generate_web_script",
]
