"""Run the FastAPI backend and Streamlit frontend together."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://localhost:8501"


def terminate_process(process: subprocess.Popen[object] | None) -> None:
    """Terminate a child process safely."""
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()


def wait_for_backend(timeout_s: int = 90) -> None:
    """Wait until the backend root endpoint responds."""
    deadline = time.monotonic() + timeout_s
    last_error = ""

    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(BACKEND_URL, timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
            time.sleep(1)

    raise TimeoutError(f"Backend did not become ready within {timeout_s} seconds. Last error: {last_error}")


def main() -> int:
    """Start backend and frontend, then keep both alive."""
    frontend: subprocess.Popen[object] | None = None
    backend = subprocess.Popen([sys.executable, "scripts/run_api.py"], cwd=REPO_ROOT)

    try:
        print("Waiting for backend...")
        wait_for_backend()
        print(f"Backend ready: {BACKEND_URL}")

        frontend_env = os.environ.copy()
        frontend_env["STREAMLIT_SERVER_HEADLESS"] = "true"
        frontend_env["API_BASE_URL"] = "http://127.0.0.1:8000/api/v1"

        frontend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "frontend/streamlit_app.py",
                "--server.headless=true",
            ],
            cwd=REPO_ROOT,
            env=frontend_env,
        )

        time.sleep(4)
        webbrowser.open(FRONTEND_URL)

        print(f"Frontend: {FRONTEND_URL}")
        print("Press Ctrl+C to stop both.")

        while True:
            if backend.poll() is not None:
                print(f"Backend exited with code {backend.returncode}.")
                terminate_process(frontend)
                return int(backend.returncode or 1)

            if frontend.poll() is not None:
                print(f"Frontend exited with code {frontend.returncode}.")
                terminate_process(backend)
                return int(frontend.returncode or 1)

            time.sleep(1)

    except KeyboardInterrupt:
        print("Stopping backend and frontend...")
        terminate_process(frontend)
        terminate_process(backend)
        return 0

    except Exception:
        terminate_process(frontend)
        terminate_process(backend)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
