"""Self-healing code execution.

Run generated Python in a resource-limited
subprocess, parse tracebacks, and loop a Kimi repair prompt until it runs clean.
"""

from app.services.pipeline.sandbox.loop import SelfHealResult, self_heal_code
from app.services.pipeline.sandbox.runner import ExecutionResult, run_code

__all__ = ["run_code", "ExecutionResult", "self_heal_code", "SelfHealResult"]
