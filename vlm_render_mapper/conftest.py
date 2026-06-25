"""
Pytest configuration.
Adds src/ to sys.path so tests can import vlm_render_mapper without installing.
"""

import sys
from pathlib import Path

# Ensure src/ is on the import path
sys.path.insert(0, str(Path(__file__).parent / "src"))
