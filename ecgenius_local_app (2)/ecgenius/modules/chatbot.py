"""
AI Health Assistant chatbot.

Tries a locally-running Ollama server (spec: DeepSeek 8B) first. If Ollama
isn't running/installed, falls back to a small rule-based educational Q&A so
the section still works out of the box.

To enable the full local LLM:
  1. Install Ollama: https://ollama.com
  2. Run:  ollama pull deepseek-r1:8b   (or your preferred 8B model)
  3. Make sure `ollama serve` is running (default http://localhost:11434)
"""

import os

SYSTEM_PROMPT = (
    "You are ECGENIUS AI, an educational cardiac-health assistant. You explain "
    "ECG concepts, heart-rhythm terminology, and general symptoms in plain "
    "language. You must NEVER diagnose a specific person, prescribe medication, "
    "or replace professional medical advice. Always include a brief reminder to "
    "consult a qualified healthcare professional for any personal concern."
)

RULE_BASED = {
    "en": {
        "arrhythmia": "Arrhythmia means an irregular heartbeat — it can beat too fast, too slow, or erratically. Many types are harmless, but some need medical attention.",
        "heart rate": "Heart rate is the number of times your heart beats per minute (bpm). A typical resting adult range is 60–100 bpm.",
        "atrial fibrillation": "Atrial fibrillation (AFib) is a common irregular heart rhythm originating in the atria, causing an erratic pulse.",
        "afib": "Atrial fibrillation (AFib) is a common irregular heart rhythm originating in the atria, causing an erratic pulse.",
        "bradycardia": "Bradycardia means a resting heart rate slower than 60 bpm.",
        "tachycardia": "Tachycardia means a resting heart rate faster than 100 bpm.",
        "explain": "I analyze R-R intervals and beat morphology from the ECG to estimate heart rate and rhythm regularity, then a trained model compares those features against learned patterns to suggest a category.",
        "default": "I can explain ECG terms like arrhythmia, heart rate, AFib, bradycardia or tachycardia — ask away! (Educational information only, not a diagnosis.)",
    },
    "hi": {
        "arrhythmia": "अतालता का अर्थ है अनियमित दिल की धड़कन — यह बहुत तेज़, धीमी या अनियमित हो सकती है।",
        "अतालता": "अतालता का अर्थ है अनियमित दिल की धड़कन — यह बहुत तेज़, धीमी या अनियमित हो सकती है।",
        "heart rate": "हृदय गति प्रति मिनट धड़कनों की संख्या है। सामान्य विश्राम सीमा 60–100 bpm है।",
        "हृदय गति": "हृदय गति प्रति मिनट धड़कनों की संख्या है। सामान्य विश्राम सीमा 60–100 bpm है।",
        "atrial fibrillation": "आलिंद फिब्रिलेशन (AFib) एक सामान्य अनियमित हृदय लय है।",
        "afib": "आलिंद फिब्रिलेशन (AFib) एक सामान्य अनियमित हृदय लय है।",
        "bradycardia": "ब्रैडीकार्डिया का अर्थ है 60 bpm से धीमी हृदय गति।",
        "tachycardia": "टैचीकार्डिया का अर्थ है 100 bpm से तेज़ हृदय गति।",
        "explain": "मैं R-R अंतराल और धड़कन आकृति का विश्लेषण करके हृदय गति और लय नियमितता का अनुमान लगाता हूँ।",
        "default": "आप मुझसे अतालता, हृदय गति, AFib, ब्रैडीकार्डिया या टैचीकार्डिया के बारे में पूछ सकते हैं! (केवल शैक्षिक जानकारी।)",
    },
}


def rule_based_answer(question: str, lang: str = "en") -> str:
    q = question.lower()
    table = RULE_BASED.get(lang, RULE_BASED["en"])
    for key, ans in table.items():
        if key != "default" and key in q:
            return ans
    return table["default"]


def _get_secret(name: str):
    """Look for the key in Streamlit secrets first, then environment variables."""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and name in st.secrets:
            key = st.secrets[name]
            if key:
                return key
    except Exception:
        pass
    return os.environ.get(name)


def cloud_chat(question: str, lang: str = "en", history=None, model: str = "claude-haiku-4-5-20251001"):
    """Cloud LLM (works on Streamlit Cloud). Returns (answer, error)."""
    key = _get_secret("ANTHROPIC_API_KEY")
    if not key:
        return None, "No ANTHROPIC_API_KEY found in Streamlit secrets."
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        messages = []
        for role, msg in (history or [])[-8:]:
            # strip the "(source)" tags we append to displayed answers
            messages.append({"role": role, "content": msg.split("  \n_(")[0]})
        messages.append({"role": "user", "content": question})
        resp = client.messages.create(
            model=model,
            max_tokens=500,
            system=SYSTEM_PROMPT + (" Respond in Hindi." if lang == "hi" else " Respond in English.")
                   + " Keep answers under 120 words.",
            messages=messages,
        )
        text = "".join(b.text for b in resp.content if getattr(b, "text", None)).strip()
        return (text, None) if text else (None, "Empty response")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def gemini_chat(question: str, lang: str = "en", history=None, model: str = "gemini-3.1-flash-lite"):
    """FREE cloud LLM via Google AI Studio key. Returns (answer, error)."""
    key = _get_secret("GEMINI_API_KEY")
    if not key:
        return None, ("No GEMINI_API_KEY found. On Streamlit Cloud: open your app -> "
                       "Manage app -> Settings -> Secrets, and add a line exactly like "
                       'GEMINI_API_KEY = "your-key-here" (with quotes), then save and reboot the app.')
    try:
        import requests
        contents = []
        for role, msg in (history or [])[-8:]:
            contents.append({"role": "user" if role == "user" else "model",
                             "parts": [{"text": msg.split("  \n_(")[0]}]})
        contents.append({"role": "user", "parts": [{"text": question}]})
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
            json={
                "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT
                    + (" Respond in Hindi." if lang == "hi" else " Respond in English.")
                    + " Keep answers under 120 words."}]},
                "contents": contents,
                "generationConfig": {"maxOutputTokens": 600},
            },
            timeout=30,
        )
        if resp.status_code == 404 and model != "gemini-2.5-flash-lite":
            # model id not available on this key/account yet -> retry on the older stable model
            return gemini_chat(question, lang, history, model="gemini-2.5-flash-lite")
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}: {resp.text[:200]}"
        parts = resp.json()["candidates"][0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts).strip()
        return (text, None) if text else (None, "Empty response")
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def get_answer(question: str, lang: str = "en", history=None):
    """Try Gemini (free) -> Claude -> local Ollama -> rule-based. Returns (answer, source_label)."""
    errors = []
    answer, err = gemini_chat(question, lang, history)
    if answer:
        return answer, "Gemini AI"
    errors.append(err)
    answer, err = cloud_chat(question, lang, history)
    if answer:
        return answer, "Claude AI"
    errors.append(err)
    answer, used = ollama_chat(question, lang)
    if used:
        return answer, "local LLM"
    return answer, "rule-based fallback — " + " | ".join(e for e in errors if e)[:300]


def ollama_chat(question: str, lang: str = "en", model: str = "deepseek-r1:8b", host: str = "http://localhost:11434"):
    """Returns (answer, used_llm: bool). Falls back to rule-based on any failure."""
    try:
        import requests
        resp = requests.post(
            f"{host}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT + (
                        " Respond in Hindi." if lang == "hi" else " Respond in English.")},
                    {"role": "user", "content": question},
                ],
                "stream": False,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("message", {}).get("content", "").strip()
        if answer:
            return answer, True
    except Exception:
        pass
    return rule_based_answer(question, lang), False
