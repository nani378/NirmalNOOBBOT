"""
ai_companion.py — Groq LLM conversation logic.

Keeps all message templates, system prompts, and Groq API calls in one place
so the rest of the code stays clean and easy to tweak.
"""

# ── Static response templates ─────────────────────────────────────────────────

# First thing spoken when a stable emotion is detected
GREETINGS: dict[str, str] = {
    "happy": "You look really happy today! That's wonderful to see.",
    "sad":   "You seem a bit sad right now. I'm right here — what's on your mind?",
    "angry": "You look upset. Take a slow breath — I'm listening whenever you're ready.",
}

# Spoken when the same negative emotion has persisted for a long time
LONG_DURATION_MESSAGES: dict[str, str] = {
    "sad":   (
        "I've noticed you've been feeling sad for quite a while. "
        "Would you like to talk about what's been bothering you?"
    ),
    "angry": (
        "You've seemed tense for a while now. "
        "Sometimes talking it out really helps — I'm here for you."
    ),
}

# Groq LLM system prompt — personalised per emotion each session
_SYSTEM_PROMPT = (
    "You are a warm, empathetic AI companion. "
    "The person in front of you is currently feeling {emotion}. "
    "Keep every response concise (1–2 sentences). "
    "Be emotionally supportive and end with a gentle follow-up question "
    "to keep the conversation going."
)


# ── Public helpers ────────────────────────────────────────────────────────────

def get_greeting(emotion: str) -> str:
    return GREETINGS.get(emotion, "How are you feeling right now?")


def get_long_duration_message(emotion: str) -> str | None:
    return LONG_DURATION_MESSAGES.get(emotion)


def get_ai_reply(groq_client, conversation_history: list,
                 groq_model: str, emotion: str) -> str:
    """
    Send the full conversation history to Groq and return the reply text.

    conversation_history is a list of {"role": …, "content": …} dicts.
    The system prompt is prepended automatically (not stored in history).
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT.format(emotion=emotion)},
        *conversation_history,
    ]
    try:
        completion = groq_client.chat.completions.create(
            model=groq_model,
            messages=messages,
            max_tokens=120,
            temperature=0.7,
        )
        return completion.choices[0].message.content.strip()
    except Exception as exc:
        print(f"[GROQ ERROR] {exc}")
        return "I'm having a small hiccup right now, but I'm still here for you."
