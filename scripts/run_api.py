"""Run the FastAPI app with a Windows-compatible asyncio event loop."""

import asyncio
import selectors
import sys
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _windows_selector_loop() -> asyncio.AbstractEventLoop:
    """Create a selector event loop for psycopg async support on Windows."""
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


async def _serve() -> None:
    """Serve the FastAPI app."""
    config = uvicorn.Config(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        loop="asyncio",
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.run(_serve(), loop_factory=_windows_selector_loop)
    else:
        asyncio.run(_serve())
