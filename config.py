"""
config.py — Central configuration for the AI Emotion Companion.
All tunable constants are here so you never need to hunt through code.
"""
import os
import platform

# ── Platform detection ────────────────────────────────────────────────────────
IS_PI      = platform.machine() in ("armv7l", "aarch64")   # Raspberry Pi
IS_WINDOWS = platform.system() == "Windows"

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX    = int(os.environ.get("CAMERA_INDEX", 0 if IS_PI else 1))
# Raspberry Pi: 320×240 for real-time speed.  Laptop: 640×480 for accuracy.
FRAME_WIDTH     = 320 if IS_PI else 640
FRAME_HEIGHT    = 240 if IS_PI else 480
# Run FER every Nth frame — saves CPU on Pi (set higher = more frames skipped)
ANALYSE_EVERY_N = 6   if IS_PI else 2

# ── Emotion detection ─────────────────────────────────────────────────────────
# Grouped confidence needed to accept a reading
# Lowered to 0.22 — accepts subtler/weaker expressions
MIN_CONFIDENCE       = 0.22
# Number of recent frames used for majority-vote stability check
STABLE_FRAMES        = 6
# Rolling window for temporal score smoothing
EMOTION_BUFFER_SIZE  = 15
# Long-term emotion history (used to detect sustained emotions)
EMOTION_HISTORY_SIZE = 60
# Seconds the stable emotion must be held before triggering a conversation
EMOTION_HOLD_SECONDS = 1.5

# ── Long-duration thresholds (out of EMOTION_HISTORY_SIZE frames) ─────────────
SAD_HISTORY_THRESHOLD   = 25
ANGRY_HISTORY_THRESHOLD = 18

# ── Groq API ──────────────────────────────────────────────────────────────────
GROQ_MODEL    = "llama-3.1-8b-instant"      # fast, ~300 ms latency
WHISPER_MODEL = "whisper-large-v3-turbo"    # Groq-hosted Whisper

# ── Voice I/O ─────────────────────────────────────────────────────────────────
TTS_RATE          = 145   # words per minute
LISTEN_TIMEOUT    = 5     # seconds to wait for speech before giving up
PHRASE_TIME_LIMIT = 8     # max seconds of a single utterance

# Max back-and-forth AI turns per emotion session
CONVERSATION_LIMIT = 6

# ── Display ───────────────────────────────────────────────────────────────────
WINDOW_NAME = "AI Emotion Companion"

# ── Persistent Memory ────────────────────────────────────────────────────────
MEMORY_DATA_DIR = "data"    # folder for persistent .npy emotion sample files
