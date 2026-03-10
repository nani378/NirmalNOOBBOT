import cv2
from fer.fer import FER
import speech_recognition as sr
from groq import Groq
from collections import deque
import tempfile
import os
import time
import threading
import pyttsx3
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# CONFIGURATION
# -----------------------------

FRAME_WIDTH = 320
FRAME_HEIGHT = 240

MIN_CONFIDENCE = 0.45   # threshold on the COMBINED group score
STABLE_FRAMES = 6

EMOTION_BUFFER_SIZE = 10
EMOTION_HISTORY_SIZE = 50

TRACKED_EMOTIONS = ["happy", "sad", "angry"]

# Maps FER's 7 raw emotions → your 3 target emotions.
# Multiple FER expressions that look like the same feeling are grouped together.
EMOTION_GROUPS = {
    "happy": ["happy", "surprise"],          # smiling, excited, open-mouth joy
    "sad":   ["sad",   "fear"],               # crying, worried, scared, downturned
    "angry": ["angry", "disgust"],            # frowning, frustrated, disgusted
    # "neutral" is intentionally not mapped — ignored
}

EMOTION_COLOURS = {
    "happy": (0, 255, 0),
    "sad":   (255, 0, 0),
    "angry": (0, 0, 255),
}

# -----------------------------
# EMOTION GROUPING
# Combines raw FER scores so different facial variations
# (e.g. scared / worried → sad) all trigger the right emotion.
# -----------------------------

def map_emotion(raw_scores: dict) -> tuple:
    """
    Fold FER's 7 raw scores into our 3 target groups.
    Returns (target_emotion, combined_confidence) or (None, 0) if
    no group reaches MIN_CONFIDENCE.
    """
    group_scores = {}
    for target, members in EMOTION_GROUPS.items():
        group_scores[target] = sum(raw_scores.get(m, 0.0) for m in members)

    best = max(group_scores, key=group_scores.get)
    return best, group_scores[best]


# -----------------------------
# GROQ CLIENT
# -----------------------------

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# -----------------------------
# TEXT TO SPEECH
# Each call creates its own engine so it works safely from any thread
# -----------------------------

def speak(text):
    print("AI:", text)
    engine = pyttsx3.init()
    engine.setProperty("rate", 145)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# -----------------------------
# SPEECH RECOGNITION
# -----------------------------

recognizer = sr.Recognizer()

def listen():

    with sr.Microphone() as source:

        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)

        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=8)
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return ""

    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tmp:
        tmp.write(audio.get_wav_data())
        path = tmp.name

    try:
        with open(path, "rb") as f:
            result = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=("speech.wav", f, "audio/wav")
            )
        text = result.text.strip()
        print("User:", text)
        return text
    except Exception as exc:
        print("STT error:", exc)
        return ""
    finally:
        if os.path.exists(path):
            os.remove(path)

# -----------------------------
# AI REPLY
# -----------------------------

def ai_reply(user_text,emotion_context):

    messages = [
        {
            "role":"system",
            "content":f"You are a caring AI companion. The user currently feels {emotion_context}. Respond with empathy."
        },
        {
            "role":"user",
            "content":user_text
        }
    ]

    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )

    return completion.choices[0].message.content

# -----------------------------
# EMOTION MEMORY
# -----------------------------

emotion_buffer = deque(maxlen=EMOTION_BUFFER_SIZE)
emotion_history = deque(maxlen=EMOTION_HISTORY_SIZE)

# -----------------------------
# CONVERSATION  (runs in background thread so camera never freezes)
# -----------------------------

def run_conversation(emotion, hist_snapshot):
    greetings = {
        "sad":   "You look sad. I'm here if you want to talk.",
        "angry": "You seem angry. Take a deep breath.",
        "happy": "You look happy today!",
    }
    speak(greetings.get(emotion, "How are you feeling?"))

    if hist_snapshot.count("sad") > 20:
        speak("You seem sad for a long time. Do you want to talk about it?")
    if hist_snapshot.count("angry") > 15:
        speak("You seem frustrated. Maybe talking could help.")

    user_text = listen()
    if user_text:
        reply = ai_reply(user_text, emotion)
        speak(reply)


# -----------------------------
# FER EMOTION MODEL
# -----------------------------

detector = FER(mtcnn=False)

# -----------------------------
# CAMERA
# -----------------------------

cap = cv2.VideoCapture(0)
cap.set(3, FRAME_WIDTH)
cap.set(4, FRAME_HEIGHT)

print("AI Companion Started")

last_emotion      = None
emotion_start_time = None
conversation_active = False   # True while background thread is running

# Last known face state — updated each detection, drawn every frame
display_emotion    = None
display_box        = None
display_confidence = 0.0

# -----------------------------
# MAIN LOOP
# -----------------------------

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # --- Emotion detection ---
    detections = detector.detect_emotions(frame)

    if detections:
        top      = detections[0]
        box      = top["box"]        # [x, y, w, h]
        raw      = top["emotions"]   # FER's 7 raw scores

        # Map 7 FER emotions → 3 target emotions using group scores
        emotion, confidence = map_emotion(raw)

        # Always update display so the box tracks the face live
        display_box        = box
        display_emotion    = emotion
        display_confidence = confidence

        if confidence > MIN_CONFIDENCE:
            emotion_buffer.append(emotion)

            if emotion_buffer.count(emotion) >= STABLE_FRAMES:
                emotion_history.append(emotion)

                if emotion != last_emotion:
                    last_emotion       = emotion
                    emotion_start_time = time.time()

                # Trigger conversation only when stable, new, and not already talking
                if (emotion_start_time
                        and time.time() - emotion_start_time > 2
                        and not conversation_active):

                    print("Detected Emotion:", emotion)
                    emotion_start_time = time.time() + 1000  # block re-trigger
                    conversation_active = True

                    def _converse(e=emotion, h=list(emotion_history)):
                        global conversation_active
                        try:
                            run_conversation(e, h)
                        finally:
                            conversation_active = False

                    threading.Thread(target=_converse, daemon=True).start()

    # --- Draw bounding box + emotion label ---
    if display_box is not None and display_emotion is not None:
        x, y, w, h = display_box
        colour = EMOTION_COLOURS.get(display_emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        label = f"{display_emotion.upper()} ({display_confidence:.0%})"
        cv2.putText(frame, label, (x, max(y - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)

    # --- Status bar ---
    status = "TALKING..." if conversation_active else "Press Q to quit"
    cv2.putText(frame, status, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("AI Companion", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()