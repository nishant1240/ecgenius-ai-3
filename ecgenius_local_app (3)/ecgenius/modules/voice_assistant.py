"""
Voice Assistant module (cloud-friendly).

- Speech-to-text runs IN THE BROWSER (Web Speech API via streamlit-mic-recorder),
  so nothing heavy is needed on the server. Use Chrome or Edge.
- Text-to-speech uses gTTS (Google TTS) which works on Streamlit Cloud and
  supports Hindi. Returns MP3 bytes that st.audio can play directly.
"""
import io


def speak_text(text: str, lang: str = "en"):
    """Returns (mp3_bytes, error)."""
    try:
        from gtts import gTTS
    except ImportError:
        return None, "Text-to-speech needs `gTTS`. Add it to requirements.txt."
    try:
        buf = io.BytesIO()
        gTTS(text=text, lang="hi" if lang == "hi" else "en").write_to_fp(buf)
        return buf.getvalue(), None
    except Exception as e:
        return None, f"Text-to-speech failed: {e}"
