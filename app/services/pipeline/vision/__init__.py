"""Web-page explainer.

Navigate a URL, screenshot it, describe it with Kimi
vision, and narrate the explanation. Reuses TTS + a Ken-Burns render.
"""

from app.services.pipeline.vision.browser import capture_page
from app.services.pipeline.vision.describe import describe_screenshots
from app.services.pipeline.vision.script import generate_web_script

__all__ = ["capture_page", "describe_screenshots", "generate_web_script"]
