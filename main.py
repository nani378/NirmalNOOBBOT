"""
=============================================================================
  AI Emotion Companion  —  Modular Edition
=============================================================================
  Modules
  -------
    config.py          — all tunable constants, platform detection
    emotion_detector.py — FER + grouping + temporal smoothing + stability vote
    voice_io.py        — cross-platform TTS (pyttsx3/espeak) + Groq Whisper STT
    ai_companion.py    — Groq LLM conversation logic & message templates
    main.py            — camera loop, overlay drawing, conversation threading

  Quick start (laptop)
  --------------------
    pip install -r requirements.txt
    python main.py

  Raspberry Pi 5 setup
  --------------------
    sudo apt update && sudo apt upgrade -y
    sudo apt install -y espeak-ng libespeak-ng1 libatlas-base-dev \
        libhdf5-dev portaudio19-dev python3-pyaudio
    pip install -r requirements-pi.txt
    python main.py

  Press 'q' in the webcam window to quit.
=============================================================================
"""

import os
import numpy as np
import sys
import time
import platform
import threading
from collections import deque

import cv2
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
from config import (
    IS_WINDOWS, IS_PI,
    FRAME_WIDTH, FRAME_HEIGHT, ANALYSE_EVERY_N,
    EMOTION_HOLD_SECONDS, EMOTION_HISTORY_SIZE,
    SAD_HISTORY_THRESHOLD, ANGRY_HISTORY_THRESHOLD,
    GROQ_MODEL, WHISPER_MODEL,
    TTS_RATE, LISTEN_TIMEOUT, PHRASE_TIME_LIMIT,
    CONVERSATION_LIMIT, WINDOW_NAME,
    MIN_CONFIDENCE,
    CAMERA_INDEX,
)
from emotion_detector import EmotionDetector, EMOTION_COLOURS
from voice_io import speak, listen
from ai_companion import get_greeting, get_long_duration_message, get_ai_reply

# ── Camera ─────────────────────────────────────────────────────────────────────

def open_camera() -> cv2.VideoCapture | None:
    """
    Open the camera at CAMERA_INDEX (set in config.py).
    Uses DirectShow on Windows and V4L2 on Linux/Raspberry Pi.
    Falls back to scanning indices 0–5 if the specified index fails.
    """
    backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2
    backend_name = "DirectShow" if IS_WINDOWS else "V4L2"

    # Try the configured index first
    indices = [CAMERA_INDEX] + [i for i in range(6) if i != CAMERA_INDEX]
    for idx in indices:
        cap = cv2.VideoCapture(idx, backend)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if idx != CAMERA_INDEX:
                    print(f"[CAMERA] Index {CAMERA_INDEX} unavailable, using index={idx}")
                else:
                    print(f"[CAMERA] Opened: index={idx}  backend={backend_name}")
                return cap


# ── Overlay ─────────────────────────────────────────────────────────────────────

def draw_overlay(frame, detection: dict, talking: bool,
                 calibrated: bool = True, calib_progress: int = 40,
                 confirmed_counts: dict | None = None,
                 feedback_msg: str = "",
                 memory_counts: dict | None = None) -> None:
    """
    Render face bounding box, fused score bars, landmark score bars,
    stable-emotion badge, memory sample counts, and status text onto
    the frame in-place.
    """
    box      = detection.get("box")
    emotion  = detection.get("raw_emotion")
    conf     = detection.get("confidence", 0.0)
    fused    = detection.get("smoothed_scores", {})
    lm_raw   = detection.get("landmark_scores", {})
    stable   = detection.get("stable_emotion")
    h_frame, w_frame = frame.shape[:2]

    # 1 — Face bounding box + emotion label (or dimmed best-guess)
    if box is not None:
        x, y, w, h = (max(0, int(v)) for v in box)
        if emotion and conf >= MIN_CONFIDENCE:
            colour = EMOTION_COLOURS.get(emotion, (200, 200, 200))
            label  = f"Detected: {emotion.upper()} ({conf:.0%})"
        elif fused:
            best_emo   = max(fused, key=fused.get)
            best_score = fused[best_emo]
            base_col   = EMOTION_COLOURS.get(best_emo, (180, 180, 180))
            colour     = tuple(max(0, v - 70) for v in base_col)  # dimmed
            label      = f"{best_emo.upper()} ({best_score:.0%}) ?"
        else:
            colour = (180, 180, 180)
            label  = "Detecting..."
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        cv2.putText(frame, label, (x, max(y - 8, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, colour, 2, cv2.LINE_AA)

    # 2 — Fused score bars (right side, top)
    BAR_MAX = 110
    bx = w_frame - BAR_MAX - 50
    by = 14
    cv2.putText(frame, "FUSED", (bx - 2, by - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    for emo, score in fused.items():
        c   = EMOTION_COLOURS.get(emo, (180, 180, 180))
        bl  = int(np.clip(score, 0.0, 1.0) * BAR_MAX)
        cv2.rectangle(frame, (bx, by), (bx + bl, by + 13), c, -1)
        cv2.putText(frame, emo[:3].upper(), (bx - 34, by + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, c, 1, cv2.LINE_AA)
        by += 18

    # 3 — Landmark-only score bars (right side, below fused)
    by += 4
    cv2.putText(frame, "LNDMK", (bx - 2, by - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 140), 1, cv2.LINE_AA)
    for emo, score in lm_raw.items():
        c   = EMOTION_COLOURS.get(emo, (120, 120, 120))
        cl  = tuple(max(0, v - 60) for v in c)   # dimmed version
        bl  = int(np.clip(score, 0.0, 1.0) * BAR_MAX)
        cv2.rectangle(frame, (bx, by), (bx + bl, by + 11), cl, -1)
        cv2.putText(frame, emo[:3].upper(), (bx - 34, by + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, cl, 1, cv2.LINE_AA)
        by += 16

    # 4 — Stable emotion badge (top-left)
    if stable:
        bc = EMOTION_COLOURS.get(stable, (255, 255, 255))
        cv2.putText(frame, f"STABLE: {stable.upper()}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, bc, 2, cv2.LINE_AA)

    # 4b — Confirmed sample counts (in-session, top-left, below stable badge)
    if confirmed_counts is not None:
        cx = 10
        cv2.putText(frame, "CONF:", (cx, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        cx += 44
        for emo, cnt in confirmed_counts.items():
            col_c = EMOTION_COLOURS.get(emo, (180, 180, 180))
            star  = "*" if cnt >= 25 else ""   # * = centroid active
            cv2.putText(frame, f"{emo[:3].upper()}{star}:{cnt}",
                        (cx, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_c, 1, cv2.LINE_AA)
            cx += 58

    # 4c — Persistent memory sample counts (HAPPY: N  SAD: N  ANGRY: N)
    if memory_counts is not None:
        cy = 62
        cx = 10
        cv2.putText(frame, "MEM:", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
        cx += 44
        for emo, cnt in memory_counts.items():
            col_c = EMOTION_COLOURS.get(emo, (180, 180, 180))
            cv2.putText(frame, f"{emo.upper()}: {cnt}",
                        (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col_c, 1, cv2.LINE_AA)
            cx += 80

    # 4d — Feedback confirmation message (on-screen stored-sample notification)
    if feedback_msg:
        cv2.putText(frame, feedback_msg, (10, h_frame // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2, cv2.LINE_AA)

    # 5 — Bottom status
    if not calibrated:
        pct    = int(calib_progress / 40 * 100)
        status = f"CALIBRATING {pct}%  — look neutral at the camera"
        col    = (0, 200, 255)
        bar_w  = int((w_frame - 20) * calib_progress / 40)
        cv2.rectangle(frame, (10, h_frame - 28), (10 + bar_w, h_frame - 20),
                      (0, 200, 255), -1)
        cv2.rectangle(frame, (10, h_frame - 28), (w_frame - 10, h_frame - 20),
                      (80, 80, 80), 1)
    elif talking:
        status, col = "[ TALKING... ]", (200, 200, 200)
    else:
        status = "H=Happy  S=Sad  A=Angry  N=reset  |  Q=quit"
        col    = (160, 160, 160)
    cv2.putText(frame, status, (10, h_frame - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


# ── Conversation thread ──────────────────────────────────────────────────────────

def run_conversation(emotion: str, groq_client: Groq,
                     emotion_history: list, done_flag: dict) -> None:
    """
    Greeting → optional long-duration support message → multi-turn chat.
    Runs in a daemon thread so the camera loop is never blocked.
    """
    def _converse() -> None:
        try:
            # Instant greeting — no API call, zero latency
            speak(get_greeting(emotion), TTS_RATE)

            # Long-duration support messages
            sad_c   = emotion_history.count("sad")
            angry_c = emotion_history.count("angry")
            if emotion == "sad"   and sad_c   >= SAD_HISTORY_THRESHOLD:
                speak(get_long_duration_message("sad"),   TTS_RATE)
            if emotion == "angry" and angry_c >= ANGRY_HISTORY_THRESHOLD:
                speak(get_long_duration_message("angry"), TTS_RATE)

            # Back-and-forth conversation
            history: list = []
            for turn in range(CONVERSATION_LIMIT):
                user_text = listen(groq_client, WHISPER_MODEL,
                                   LISTEN_TIMEOUT, PHRASE_TIME_LIMIT)
                if not user_text:
                    speak("I'm here whenever you're ready. Take care!", TTS_RATE)
                    break

                history.append({"role": "user", "content": user_text})
                reply = get_ai_reply(groq_client, history, GROQ_MODEL, emotion)
                history.append({"role": "assistant", "content": reply})
                speak(reply, TTS_RATE)

                if turn == CONVERSATION_LIMIT - 1:
                    speak("It was lovely talking with you. Take care!", TTS_RATE)

        except Exception as exc:
            print(f"[CONVO ERROR] {exc}")
        finally:
            done_flag["busy"] = False
            print("[CONVO] Session ended.")

    threading.Thread(target=_converse, daemon=True).start()


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[ERROR] GROQ_API_KEY not set. Add it to your .env file.")
        sys.exit(1)

    groq_client = Groq(api_key=api_key)

    cap = open_camera()
    if cap is None:
        print("[ERROR] No camera found. Check connections and try again.")
        sys.exit(1)

    detector = EmotionDetector()

    emotion_history   = deque(maxlen=EMOTION_HISTORY_SIZE)
    last_spoken       = None
    emotion_hold_time = None
    convo_flag        = {"busy": False}
    detection: dict   = {}      # latest process_frame() result
    frame_count       = 0
    last_no_face_time = time.time()
    feedback_msg      = ""      # on-screen confirmation message
    feedback_msg_time = 0.0     # timestamp when feedback_msg was set
    auto_conf_start: dict = {}  # tracks passive auto-confirm timing {emotion, t}

    print("\n" + "=" * 52)
    print("  AI Emotion Companion — Running")
    print(f"  Platform : {'Raspberry Pi' if IS_PI else 'Laptop/Desktop'}")
    print(f"  Camera   : {FRAME_WIDTH}x{FRAME_HEIGHT}  (every {ANALYSE_EVERY_N} frames)")
    print("  Press Q in the video window to quit.")
    print("=" * 52 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame_count += 1

            # ── Emotion detection (every Nth frame) ──────────────────────────
            if frame_count % ANALYSE_EVERY_N == 0:
                detection = detector.process_frame(frame)
                stable    = detection.get("stable_emotion")

                if detection.get("box") is not None:
                    last_no_face_time = time.time()
                elif time.time() - last_no_face_time > 1.5:
                    # Clear stale display after 1.5 s of no face
                    detection = {}

                if stable:
                    emotion_history.append(stable)

                    if stable != last_spoken:
                        last_spoken       = stable
                        emotion_hold_time = time.time()

                    # Start conversation: stable + held ≥ threshold + not busy
                    if (emotion_hold_time
                            and time.time() - emotion_hold_time >= EMOTION_HOLD_SECONDS
                            and not convo_flag["busy"]):

                        print(f"[TRIGGER] Emotion={stable.upper()}")
                        convo_flag["busy"] = True
                        emotion_hold_time  = time.time() + 9999   # block re-trigger
                        detector.reset_votes()

                        run_conversation(
                            stable, groq_client,
                            list(emotion_history), convo_flag,
                        )

                # ── Auto-confirm: build personal centroids passively ──────────
                # When emotion is stable at ≥45% confidence for ≥3 seconds,
                # store a sample automatically — no key press needed.
                auto_conf_emotion = (
                    stable
                    if (stable
                        and detection.get("confidence", 0.0) >= 0.45
                        and detector.is_calibrated
                        and not convo_flag["busy"])
                    else None
                )
                if auto_conf_emotion:
                    if auto_conf_start.get("emotion") != auto_conf_emotion:
                        auto_conf_start = {"emotion": auto_conf_emotion, "t": time.time()}
                    elif time.time() - auto_conf_start["t"] >= 3.0:
                        auto_conf_start["t"] = time.time()
                        detector.confirm_detection(auto_conf_emotion)
                else:
                    auto_conf_start = {}

            # ── Draw overlay every frame ──────────────────────────────────────
            draw_overlay(
                frame, detection, convo_flag["busy"],
                detector.is_calibrated, detector.calib_progress,
                detector.confirmed_counts,
                feedback_msg if (time.time() - feedback_msg_time) < 3.0 else "",
                detector.memory_counts,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quitting…")
                break

            elif key in (ord("h"), ord("H")):
                if detector.confirm_detection("happy"):
                    feedback_msg      = f"HAPPY confirmed  ({detector.confirmed_counts['happy']} samples)"
                    feedback_msg_time = time.time()
                    print(f"[CONFIRM] happy  ({detector.confirmed_counts['happy']} samples)")

            elif key in (ord("s"), ord("S")):
                if detector.confirm_detection("sad"):
                    feedback_msg      = f"SAD confirmed  ({detector.confirmed_counts['sad']} samples)"
                    feedback_msg_time = time.time()
                    print(f"[CONFIRM] sad    ({detector.confirmed_counts['sad']} samples)")

            elif key in (ord("a"), ord("A")):
                if detector.confirm_detection("angry"):
                    feedback_msg      = f"ANGRY confirmed  ({detector.confirmed_counts['angry']} samples)"
                    feedback_msg_time = time.time()
                    print(f"[CONFIRM] angry  ({detector.confirmed_counts['angry']} samples)")

            elif key in (ord("n"), ord("N")):
                detector.reset_votes()
                feedback_msg      = "Votes reset"
                feedback_msg_time = time.time()
                print("[INFO] Votes reset")

    except KeyboardInterrupt:
        print("[INFO] Interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Goodbye!")


if __name__ == "__main__":
    main()
