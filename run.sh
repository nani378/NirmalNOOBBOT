#!/bin/bash
# ============================================================
#  MoodBot — One-click launcher for Raspberry Pi / Linux
#  Installs system deps + Python deps (first run) and starts.
#  Usage:  git clone <repo> && cd NirmalNOOBBOT && bash run.sh
# ============================================================

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  MoodBot — AI Emotion Companion"
echo "============================================"

# Install system dependencies if espeak-ng is not found
if ! command -v espeak-ng &>/dev/null; then
    echo "[SETUP] Installing system dependencies (needs sudo)..."
    sudo apt update
    sudo apt install -y \
        python3-pip python3-venv \
        espeak-ng libespeak-ng1 \
        libatlas-base-dev libhdf5-dev \
        libgtk-3-dev libopencv-dev \
        portaudio19-dev python3-pyaudio \
        libjpeg-dev libpng-dev libtiff-dev \
        libasound2-dev alsa-utils \
        bluez bluez-alsa-utils
fi

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install / update Python dependencies
echo "[SETUP] Installing Python dependencies..."
pip install --quiet -r requirements-pi.txt

# Check .env
if [ ! -f ".env" ]; then
    echo ""
    echo "[ERROR] .env file not found!"
    echo "  Create it with:  echo 'GROQ_API_KEY=your_key_here' > .env"
    echo ""
    exit 1
fi

# Run the app
echo ""
echo "[START] Launching MoodBot..."
echo "  Press Q in the video window to quit."
echo ""
python main.py
