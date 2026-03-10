#!/bin/bash
# ============================================================
#  MoodBot — Docker launcher for Linux / Raspberry Pi
#  Usage:  git clone <repo> && cd NirmalNOOBBOT && bash run-docker.sh
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

# Check .env exists
if [ ! -f ".env" ]; then
    echo ""
    echo "[ERROR] .env file not found!"
    echo "  Create it with:  echo 'GROQ_API_KEY=your_key_here' > .env"
    echo ""
    exit 1
fi

# Allow X11 connections from Docker
xhost +local:docker 2>/dev/null || true

# Create data dir for persistent memory
mkdir -p data

# Build and run
echo "[BUILD] Building MoodBot Docker image..."
docker compose up --build
