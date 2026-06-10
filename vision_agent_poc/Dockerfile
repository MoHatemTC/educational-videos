# Vision-based UI Agent PoC — container image
# Provides Python, the app, and a Chromium binary that browser-use drives over
# CDP headlessly.
FROM python:3.11-slim

WORKDIR /app

# Chromium + fonts, plus Xvfb (virtual display) and ffmpeg (screen recorder) so
# we can record the full process to mp4. browser-use 0.12 talks to the Chromium
# binary directly (it no longer ships the Playwright CLI).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        chromium fonts-liberation xvfb ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code.
COPY . .
RUN chmod +x entrypoint.sh

# Runs as root in a container, so point at the system Chromium. The browser runs
# headful inside Xvfb (set by entrypoint) so ffmpeg can record it.
ENV CHROME_PATH=/usr/bin/chromium

ENTRYPOINT ["./entrypoint.sh"]
