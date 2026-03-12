#!/bin/bash
# ============================================================
#  MoodBot — Native (no Docker) launcher for Raspberry Pi 5
#
#  USAGE — pass API key directly (no .env file needed):
#    bash scripts/run-native.sh YOUR_GROQ_API_KEY
#
#  ONE-LINE SETUP:
#    git clone https://github.com/ManojSwagath/NirmalNOOBBOT.git && \
#      cd NirmalNOOBBOT && \
#      bash scripts/run-native.sh YOUR_GROQ_API_KEY
#
#  TO UPDATE AND RE-RUN:
#    cd NirmalNOOBBOT && git pull && bash scripts/run-native.sh YOUR_GROQ_API_KEY
# ============================================================

set -e
# Always run from project root regardless of where script is called from
cd "$(dirname "$0")/.."

echo "============================================"
echo "  MoodBot — AI Emotion Companion (Native)"
echo "============================================"

# ── API key: accept from CLI arg or fall back to .env ─────────────────────────
if [ -n "$1" ]; then
    echo "[KEY] Using GROQ_API_KEY from command-line argument."
    echo "GROQ_API_KEY=$1" > .env
elif [ -f ".env" ]; then
    echo "[KEY] Using GROQ_API_KEY from .env file."
else
    echo ""
    echo "[ERROR] No GROQ_API_KEY provided!"
    echo "  Pass it as an argument:  bash scripts/run-native.sh YOUR_KEY"
    echo "  Or create .env:          echo 'GROQ_API_KEY=...' > .env"
    echo ""
    exit 1
fi

# ── System dependencies ───────────────────────────────────────────────────────
if ! command -v espeak-ng &>/dev/null; then
    echo "[SETUP] Installing system dependencies (needs sudo)..."
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        python3-pip python3-venv \
        espeak-ng libespeak-ng1 \
        libatlas-base-dev \
        libgl1 libglib2.0-0 \
        libgtk-3-dev \
        portaudio19-dev python3-pyaudio \
        libjpeg-dev libpng-dev \
        libasound2-dev alsa-utils \
        bluez bluez-alsa-utils \
        pulseaudio-utils \
        v4l-utils
    echo "[SETUP] System dependencies installed."
fi

# ── Python virtual environment ────────────────────────────────────────────────
if [ ! -d "venv" ]; then
    echo "[SETUP] Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# ── Python dependencies ───────────────────────────────────────────────────────
echo "[SETUP] Installing Python packages (first run may take a few minutes)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements-pi.txt

# ── Hardware check ────────────────────────────────────────────────────────────
echo ""
echo "[CAMERA] Detected video devices:"
ls /dev/video* 2>/dev/null || echo "  (no /dev/video* found — plug in USB webcam)"

echo ""
echo "[AUDIO] Available recording devices (webcam built-in mics are skipped at runtime):"
python3 -c "
import speech_recognition as sr
_webcam_kw = ['camera', 'webcam', 'video', 'uvc', 'cam']
_usb_kw    = ['usb', 'microphone', 'mic']
names = sr.Microphone.list_microphone_names()
for i, n in enumerate(names):
    lc = n.lower()
    if any(k in lc for k in _webcam_kw):
        print(f'  [{i}] {n}  [webcam mic - will be skipped]')
    elif any(k in lc for k in _usb_kw):
        print(f'  [{i}] {n}  [standalone USB mic - will be used]')
    else:
        print(f'  [{i}] {n}')
" 2>/dev/null || echo "  (PyAudio not yet ready)"

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "============================================"
echo "  Starting MoodBot..."
echo "  Press Q in the video window to quit."
echo "============================================"
echo ""

python main.py
