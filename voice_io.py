"""
voice_io.py — Cross-platform Text-to-Speech and Speech-to-Text.

  TTS:
    Windows      → pyttsx3 (SAPI5).  A new engine instance is created per
                   call so it works safely from any background thread.
    Linux / Pi   → espeak-ng subprocess (no Python bindings needed).

  STT:
    Both platforms → Groq Whisper (whisper-large-v3-turbo).
    Audio is captured with SpeechRecognition, saved to a temp WAV file,
    uploaded to Groq, and the temp file is always cleaned up in a finally block.
    This avoids the flac.exe PermissionError that occurs on Windows when using
    SpeechRecognition's built-in Google engine.
"""

import ctypes
import ctypes.util
import os
import platform
import subprocess
import tempfile
import threading

import speech_recognition as sr

IS_WINDOWS = platform.system() == "Windows"
IS_PI      = platform.machine() in ("armv7l", "aarch64")   # Raspberry Pi

# ALSA output device for the Bluetooth speaker (Pi only).
# Mirrors config.py — defined here to avoid a circular import.
BT_SPEAKER_ALSA_DEVICE = os.environ.get("BT_SPEAKER_ALSA_DEVICE", "bluealsa")

# ── Silence ALSA / JACK noise at C-library level (thread-safe, permanent) ────
# PyAudio probes every virtual ALSA device on init, producing dozens of
# harmless 'Unknown PCM' and 'jack server not running' lines.
# Installing a no-op ALSA error handler + setting JACK_NO_START_SERVER
# suppresses them all safely without any fd manipulation.

def _init_audio_silence() -> None:
    os.environ.setdefault("JACK_NO_START_SERVER", "1")
    if IS_WINDOWS:
        return
    try:
        asound = ctypes.CDLL(
            ctypes.util.find_library("asound") or "libasound.so.2"
        )
        c_handler = ctypes.CFUNCTYPE(
            None,
            ctypes.c_char_p, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        )
        asound.snd_lib_error_set_handler(c_handler(0))
    except Exception:
        pass


_init_audio_silence()

# ── Shutdown control ──────────────────────────────────────────────────────────
_shutdown = threading.Event()            # set by shutdown() when the app exits
_tts_proc: "subprocess.Popen | None" = None  # active espeak-ng proc (Pi only)


# ── Mic detection keyword lists ──────────────────────────────────────────────
# Devices whose names match any WEBCAM keyword are the webcam's built-in mic
# and must be skipped to avoid audio interference.
_WEBCAM_KEYWORDS  = ["camera", "webcam", "video", "uvc", "cam"]
# A device that is NOT a webcam mic but matches any of these is the standalone
# external USB microphone we want to use.
_USB_MIC_KEYWORDS = ["usb", "microphone", "mic"]

# ── Mic index caches ──────────────────────────────────────────────────────────
# Each scan runs only once per process; results are cached in module globals.
_pa_mic_index: int | None = None
_pa_mic_scanned: bool = False
_sr_mic_index: int | None = None
_sr_mic_scanned: bool = False


def _find_standalone_usb_mic_pyaudio() -> int | None:
    """Enumerate input devices with PyAudio and return the standalone USB mic.

    Devices whose names contain webcam-related keywords ('camera', 'webcam',
    'video', 'uvc', 'cam') are explicitly skipped so the webcam's built-in
    mic is never chosen.  The first remaining device whose name contains 'usb',
    'microphone', or 'mic' is selected as the external USB microphone.
    Result is cached — the scan runs only once per process.
    Returns None when no qualifying mic is found or PyAudio is unavailable.
    """
    global _pa_mic_index, _pa_mic_scanned
    if _pa_mic_scanned:
        return _pa_mic_index

    try:
        import pyaudio
    except ImportError:
        print("[MIC/PA] PyAudio not installed — run: pip install pyaudio")
        _pa_mic_scanned = True
        return None

    pa = pyaudio.PyAudio()
    try:
        count = pa.get_device_count()
        print("[MIC/PA] Enumerating input devices via PyAudio:")
        for i in range(count):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                name     = info.get("name", "")
                name_lc  = name.lower()
                is_webcam_mic = any(k in name_lc for k in _WEBCAM_KEYWORDS)
                is_usb_mic    = any(k in name_lc for k in _USB_MIC_KEYWORDS)
                if is_webcam_mic:
                    print(f"        [{i}] {name}  [webcam mic — SKIPPED]")
                elif is_usb_mic:
                    print(f"        [{i}] {name}  [standalone USB mic]")
                    if _pa_mic_index is None:
                        print(f"[MIC/PA] ✓ Selected standalone USB mic → [{i}] {name}")
                        _pa_mic_index = i
                else:
                    print(f"        [{i}] {name}")
    except Exception as exc:
        print(f"[MIC/PA] Enumeration error: {exc}")
    finally:
        pa.terminate()

    if _pa_mic_index is None:
        print("[MIC/PA] No standalone USB mic found — will use system default")
    _pa_mic_scanned = True
    return _pa_mic_index


def _find_standalone_usb_mic_sr() -> int | None:
    """Scan microphone list via SpeechRecognition and return the standalone USB mic.

    Devices whose names contain webcam-related keywords are explicitly skipped
    to avoid selecting the webcam's built-in microphone.  The first remaining
    device containing 'usb', 'microphone', or 'mic' in its name is returned.
    Result is cached — the scan only runs on the very first call.
    Returns None if not found, causing sr.Microphone() to use the system default.
    """
    global _sr_mic_index, _sr_mic_scanned
    if _sr_mic_scanned:
        return _sr_mic_index

    try:
        names = sr.Microphone.list_microphone_names()
        print("[MIC/SR] Scanning available microphones:")
        for i, name in enumerate(names):
            name_lc       = name.lower()
            is_webcam_mic = any(k in name_lc for k in _WEBCAM_KEYWORDS)
            is_usb_mic    = any(k in name_lc for k in _USB_MIC_KEYWORDS)
            if is_webcam_mic:
                print(f"        [{i}] {name}  [webcam mic — SKIPPED]")
            elif is_usb_mic:
                print(f"        [{i}] {name}  [standalone USB mic]")
                if _sr_mic_index is None:
                    print(f"[MIC/SR] ✓ Selected standalone USB mic → [{i}] {name}")
                    _sr_mic_index = i
            else:
                print(f"        [{i}] {name}")
        if _sr_mic_index is None:
            print("[MIC/SR] No standalone USB mic found — using system default")
    except Exception as exc:
        print(f"[MIC/SR] Could not enumerate microphones: {exc}")

    _sr_mic_scanned = True
    return _sr_mic_index


def speak(text: str, rate: int = 145) -> None:
    """
    Speak text synchronously and print it to the terminal.
    Safe to call from any thread on both Windows and Linux/Pi.
    """
    print(f"\n[BOT REPLY] {text}")
    print("-" * 60)
    if IS_WINDOWS:
        import pyttsx3
        # Create a fresh engine instance — required for thread safety on Windows
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    elif IS_PI:
        if _shutdown.is_set():
            return
        # Raspberry Pi TTS pipeline:
        #   Primary  : espeak-ng --stdout | paplay  (PulseAudio → BT speaker)
        #   Fallback : espeak-ng direct             (ALSA default output)
        #
        # CRITICAL — pipe ownership:
        #   After passing espeak_proc.stdout to paplay we MUST close our own
        #   reference to that pipe.  If we don't, the pipe has two readers
        #   (our process + paplay); when paplay exits our process still holds
        #   the read end open, espeak never receives SIGPIPE, fills the 64 KB
        #   kernel pipe buffer, blocks on write, and espeak_proc.wait()
        #   deadlocks permanently — silently killing every turn after turn 1.
        global _tts_proc
        try:
            if _shutdown.is_set():
                return
            espeak_proc = subprocess.Popen(
                ["espeak-ng", "-s", str(rate), "-v", "en+f3", "--stdout", text],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            _tts_proc = espeak_proc
            player = subprocess.Popen(
                ["paplay"],           # PulseAudio player → BT speaker
                stdin=espeak_proc.stdout,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Release OUR copy of the pipe read-end so espeak gets SIGPIPE
            # when paplay exits and can terminate on its own.
            espeak_proc.stdout.close()
            player.wait()
            espeak_proc.wait()
            if player.returncode != 0:
                raise subprocess.SubprocessError(f"paplay exit {player.returncode}")
        except (FileNotFoundError, subprocess.SubprocessError):
            # paplay not installed or failed — fall back to direct ALSA output
            if _shutdown.is_set():
                return
            fallback = subprocess.Popen(
                ["espeak-ng", "-s", str(rate), "-v", "en+f3", text],
                stderr=subprocess.DEVNULL,
            )
            _tts_proc = fallback
            fallback.wait()
        except Exception as exc:
            print(f"[TTS] Error: {exc}")
        finally:
            _tts_proc = None
    else:
        # Generic Linux desktop — en+f3 = English female voice; -s = speed (wpm)
        subprocess.run(
            ["espeak-ng", "-s", str(rate), "-v", "en+f3", text],
            check=False,
        )


def listen(groq_client, whisper_model: str,
           timeout: int = 5, phrase_limit: int = 8) -> str:
    """
    Record one utterance from the default microphone and transcribe
    it via Groq Whisper.

    Returns the transcribed text, or an empty string on timeout / error.
    """
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    # ── Step 1: capture microphone audio ─────────────────────────────────────
    # Always select the standalone external USB mic and explicitly skip the
    # webcam's built-in mic to prevent audio interference.
    # On Pi: PyAudio enumeration is more reliable; fall back to SR scan.
    # On Windows / generic Linux: use SpeechRecognition enumeration.
    if IS_PI:
        mic_index = _find_standalone_usb_mic_pyaudio()
        if mic_index is None:
            mic_index = _find_standalone_usb_mic_sr()
    else:
        mic_index = _find_standalone_usb_mic_sr()
    if _shutdown.is_set():
        return ""
    try:
        with sr.Microphone(device_index=mic_index) as source:
            print("\n[LISTENING] Adjusting for ambient noise…")
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            print(f"[LISTENING] *** Speak now  (up to {phrase_limit}s) ***")
            audio = recognizer.listen(
                source,
                timeout=timeout,
                phrase_time_limit=phrase_limit,
            )
        print("[LISTENING] Audio captured — transcribing…")
    except sr.WaitTimeoutError:
        print("[LISTENING] Timed out — no speech detected.")
        return ""

    # ── Step 2: transcribe via Groq Whisper ───────────────────────────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio.get_wav_data())
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                model=whisper_model,
                file=("audio.wav", f, "audio/wav"),
            )

        text = result.text.strip()
        print(f"[YOU SAID]  \"{text}\"")
        return text

    except Exception as exc:
        print(f"[STT ERROR] {exc}")
        return ""

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def shutdown() -> None:
    """Stop all voice I/O immediately.  Call this on application exit.

    Sets the shutdown event so speak() and listen() return without doing
    anything, and kills the active espeak-ng subprocess (if any) so no
    queued TTS plays after the main loop exits.
    """
    _shutdown.set()
    proc = _tts_proc
    if proc is not None:
        try:
            proc.kill()
        except Exception:
            pass
