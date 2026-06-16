#!/usr/bin/env bash
# =============================================================================
# entrypoint.sh — Docker container entrypoint
#
# Responsibilities:
#   1. Start a virtual X display (Xvfb) so Chromium can run headed inside the
#      container without a real monitor.
#   2. Optionally launch ffmpeg to record the Xvfb display to a raw .mp4.
#   3. Execute the vision agent (run.py) with any CLI arguments forwarded in.
#   4. On exit, stop ffmpeg cleanly so the final MP4 is properly finalized.
#
# Environment variables (with defaults):
#   DISPLAY_NUM    – X display number used by Xvfb (default: 99)
#   WINDOW_SIZE    – e.g. 1280x900 (default: 1280x900)
#   RECORD_VIDEO   – true/false – whether to run ffmpeg (default: true)
#   OUTPUT_DIR     – root output path (default: output)
# =============================================================================

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
DISPLAY_NUM="${DISPLAY_NUM:-99}"
WINDOW_SIZE="${WINDOW_SIZE:-1280x900}"
RECORD_VIDEO="${RECORD_VIDEO:-true}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"

# Parse WxH
W=$(echo "$WINDOW_SIZE" | cut -dx -f1)
H=$(echo "$WINDOW_SIZE" | cut -dx -f2)

export DISPLAY=":${DISPLAY_NUM}"

# ─── 1. Start Xvfb ───────────────────────────────────────────────────────────
echo "[entrypoint] Starting Xvfb on display :${DISPLAY_NUM} (${W}x${H}x24)"
Xvfb ":${DISPLAY_NUM}" -screen 0 "${W}x${H}x24" -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Give Xvfb a moment to initialise
sleep 1

if ! kill -0 "$XVFB_PID" 2>/dev/null; then
  echo "[entrypoint] ERROR: Xvfb failed to start." >&2
  exit 1
fi
echo "[entrypoint] Xvfb running (PID ${XVFB_PID})"

# ─── 2. Optional ffmpeg screen capture ───────────────────────────────────────
FFMPEG_PID=""
if [[ "${RECORD_VIDEO,,}" =~ ^(1|true|yes)$ ]]; then
  mkdir -p "${OUTPUT_DIR}/video"
  VIDEO_PATH="${OUTPUT_DIR}/video/screen_capture.mp4"
  echo "[entrypoint] Starting ffmpeg screen capture → ${VIDEO_PATH}"
  ffmpeg -y \
    -f x11grab \
    -video_size "${W}x${H}" \
    -framerate 10 \
    -i ":${DISPLAY_NUM}.0" \
    -c:v libx264 \
    -preset ultrafast \
    -crf 28 \
    "${VIDEO_PATH}" \
    </dev/null \
    >/dev/null 2>&1 &
  FFMPEG_PID=$!
  echo "[entrypoint] ffmpeg running (PID ${FFMPEG_PID})"
fi

# ─── 3. Run the agent ────────────────────────────────────────────────────────
echo "[entrypoint] Launching vision agent: python run.py $*"
python run.py "$@"
AGENT_EXIT=$?

# ─── 4. Cleanup ──────────────────────────────────────────────────────────────
if [[ -n "$FFMPEG_PID" ]] && kill -0 "$FFMPEG_PID" 2>/dev/null; then
  echo "[entrypoint] Stopping ffmpeg (PID ${FFMPEG_PID}) …"
  kill -INT "$FFMPEG_PID" 2>/dev/null || true
  wait "$FFMPEG_PID" 2>/dev/null || true
  echo "[entrypoint] ffmpeg stopped."
fi

if kill -0 "$XVFB_PID" 2>/dev/null; then
  kill "$XVFB_PID" 2>/dev/null || true
fi

echo "[entrypoint] Done (exit ${AGENT_EXIT})."
exit "$AGENT_EXIT"
