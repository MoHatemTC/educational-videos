"""
vision_agent — Vision-Based UI Agent for DOM-Independent Browser Navigation.

Implements the Observe → Reason → Plan → Act loop:
  1. Observe  : capture a full-page screenshot via Playwright / browser-use CDP backend
  2. Reason   : POST the base64-encoded frame to a VLM (Kimi K2.5 / GPT-4o / Claude CU)
                through a LiteLLM proxy
  3. Plan     : parse the model's structured action decision
  4. Act      : execute click / type / scroll / navigate via Playwright

Public surface
--------------
    from vision_agent import VisionAgent
    from vision_agent.browser import BrowserController
    from vision_agent.actions import Action, ActionType
    from vision_agent.recorder import Recorder
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("vision_agent")
except PackageNotFoundError:
    __version__ = "0.1.0"

from vision_agent.agent import VisionAgent  # noqa: F401 – re-exported

__all__ = ["VisionAgent", "__version__"]
