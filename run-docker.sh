#!/bin/bash
# ============================================================
#  MoodBot — Docker launcher for Linux / Raspberry Pi
#
#  ONE-LINE SETUP (paste into your Raspberry Pi terminal):
#
#    git clone https://github.com/nani378/NirmalNOOBBOT.git && \
#      cd NirmalNOOBBOT && \
#      echo 'GROQ_API_KEY=your_key_here' > .env && \
#      bash run-docker.sh
#
#  If you already cloned and want to update:
#    cd NirmalNOOBBOT && git pull && bash run-docker.sh
# ============================================================

set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  MoodBot — Docker Launch"
echo "============================================"

# Check Docker is installed
if ! command -v docker &>/dev/null; then
    echo "[SETUP] Docker not found. Installing..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "[INFO] Docker installed. Log out and back in, then re-run this script."
    exit 0
fi

# If user can't talk to Docker daemon, use newgrp or sudo
DOCKER_CMD="docker"
if ! docker info &>/dev/null 2>&1; then
    echo "[INFO] Adding you to the docker group (needs sudo)..."
    sudo usermod -aG docker "$USER"
    DOCKER_CMD="sudo docker"
fi

# Check .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "[ERROR] .env file not found!"
    echo "  Create it with:  echo 'GROQ_API_KEY=your_key_here' > .env"
    echo ""
    exit 1
fi

# Allow X11 connections from Docker containers
xhost +local: 2>/dev/null || true

# Create data dir for persistent memory
mkdir -p data

# Export DISPLAY so sudo docker can see it
export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}

# Set up PulseAudio for speaker/mic passthrough into Docker
mkdir -p ~/.config/pulse
pactl load-module module-native-protocol-unix auth-anonymous=1 2>/dev/null || true
echo "[AUDIO] Available audio sources (microphones):"
pactl list short sources 2>/dev/null || echo "  (pactl not available)"

# List detected cameras
echo "[CAMERA] Detected video devices:"
ls /dev/video* 2>/dev/null || echo "  (no /dev/video* found)"

# Build and run
echo "[BUILD] Building MoodBot Docker image..."
$DOCKER_CMD compose up --build
