#!/usr/bin/env bash
# Records the FULL browser process as a continuous .mp4.
#
# browser-use 0.12 (CDP backend) can't produce a video via record_video_dir, so
# instead we run Chromium *headful* on a virtual X display (Xvfb) and capture
# that display with ffmpeg for the entire run.
set -u

RES="${SCREEN_RESOLUTION:-1280x800}"   # WxH of the recorded area
DEPTH=24
DISPLAY_NUM=99
export DISPLAY=":${DISPLAY_NUM}"

OUT="${OUTPUT_DIR:-output}"
VIDEO_DIR="${OUT}/video"
mkdir -p "${VIDEO_DIR}"
VIDEO_PATH="${VIDEO_DIR}/process.mp4"

# 1) Virtual display
Xvfb ":${DISPLAY_NUM}" -screen 0 "${RES}x${DEPTH}" -ac +extension RANDR >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!
sleep 2

# 2) Screen recorder (continuous capture of the whole virtual display)
ffmpeg -y -loglevel error -f x11grab -draw_mouse 1 -video_size "${RES}" \
  -framerate 12 -i ":${DISPLAY_NUM}" \
  -codec:v libx264 -pix_fmt yuv420p -preset ultrafast "${VIDEO_PATH}" &
FFMPEG_PID=$!

# 3) The agent — headful so Chromium renders into Xvfb; browser-use's own video
#    recorder is off (we use ffmpeg instead).
HEADLESS=false RECORD_VIDEO=false WINDOW_SIZE="${RES}" python agent_poc.py
STATUS=$?

# 4) Stop the recorder cleanly so the mp4 is finalized.
sleep 1
kill -INT "${FFMPEG_PID}" 2>/dev/null
wait "${FFMPEG_PID}" 2>/dev/null
kill "${XVFB_PID}" 2>/dev/null

if [ -s "${VIDEO_PATH}" ]; then
  echo "🎥 Full-process video: ${VIDEO_PATH}"
else
  echo "🎥 Video was not captured — see /tmp/ffmpeg.log"
fi
exit "${STATUS}"
