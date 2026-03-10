# ============================================================
#  MoodBot — Docker image
#  Provides camera, microphone, display, and speaker access
#  for real-time emotion detection + AI companion.
#
#  Build:   docker build -t moodbot .
#  Run:     docker-compose up   (recommended)
# ============================================================

FROM python:3.11-slim

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for OpenCV, PyAudio, espeak (TTS), GTK (imshow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    libgtk-3-0 \
    portaudio19-dev \
    espeak-ng libespeak-ng1 \
    libasound2 libpulse0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config.py emotion_detector.py voice_io.py ai_companion.py main.py ./

# Create data directory for persistent memory
RUN mkdir -p data

# Copy model file if it exists (optional — downloaded at runtime otherwise)
COPY face_landmarker.tas[k] ./

CMD ["python", "main.py"]
