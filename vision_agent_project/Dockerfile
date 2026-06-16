# =============================================================================
# Dockerfile — Vision-Based UI Agent
#
# Stack:
#   Python 3.11 slim + Playwright Chromium + Xvfb + ffmpeg
#
# Build:
#   docker build -t vision-agent:latest .
#
# Run:
#   docker run --rm --env-file .env vision-agent:latest "Your task here"
# =============================================================================

FROM python:3.11-slim

# ─── System packages ──────────────────────────────────────────────────────────
# xvfb          : virtual X display for headed Chromium inside Docker
# ffmpeg        : screen recording
# wget/gnupg    : needed by Playwright's Chromium installer
# Other libs    : shared libraries required by Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        ffmpeg \
        wget \
        gnupg \
        ca-certificates \
        fonts-liberation \
        libatk-bridge2.0-0 \
        libatk1.0-0 \
        libc6 \
        libcairo2 \
        libcups2 \
        libdbus-1-3 \
        libexpat1 \
        libfontconfig1 \
        libgbm1 \
        libgcc1 \
        libglib2.0-0 \
        libgtk-3-0 \
        libnspr4 \
        libnss3 \
        libpango-1.0-0 \
        libpangocairo-1.0-0 \
        libstdc++6 \
        libx11-6 \
        libx11-xcb1 \
        libxcb1 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxext6 \
        libxfixes3 \
        libxi6 \
        libxrandr2 \
        libxrender1 \
        libxss1 \
        libxtst6 \
        lsb-release \
        xdg-utils \
    && rm -rf /var/lib/apt/lists/*

# ─── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ─── Python dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (Chromium only for smaller image)
RUN playwright install chromium && playwright install-deps chromium

# ─── Application code ────────────────────────────────────────────────────────
COPY vision_agent/ ./vision_agent/
COPY run.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# ─── Output directories ──────────────────────────────────────────────────────
RUN mkdir -p output/screenshots output/video

# ─── Environment defaults (overridden by .env / docker-compose) ──────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HEADLESS=false \
    RECORD_VIDEO=true \
    MAX_STEPS=30 \
    OUTPUT_DIR=output \
    WINDOW_SIZE=1280x900 \
    GIF_FPS=1 \
    LOG_LEVEL=INFO

# ─── Entrypoint ───────────────────────────────────────────────────────────────
ENTRYPOINT ["./entrypoint.sh"]

# Default task (overridden via CMD or TASK_PROMPT env var)
CMD []
