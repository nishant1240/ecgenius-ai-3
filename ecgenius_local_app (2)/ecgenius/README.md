# ECGENIUS AI — Local / Cloud Prototype

AI-assisted ECG screening and health-assistant prototype. **Educational/research
prototype only — not a medical diagnostic device.**

## 1. Setup (local)

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 2. Setup (Streamlit Community Cloud)

1. Push this folder to a public GitHub repo (`app.py` at the top level).
2. Deploy at share.streamlit.io, main file `app.py`, **Python version 3.12**.
3. **Set your Gemini API key** so the chatbot/voice assistant use real AI instead of the rule-based fallback:
   - Get a free key at [aistudio.google.com](https://aistudio.google.com) → "Get API key"
   - In your Streamlit Cloud app → **Manage app → Settings → Secrets**, add exactly:
     ```
     GEMINI_API_KEY = "your-key-here"
     ```
   - Save, then reboot the app from the Manage app menu.
   - If the chatbot still shows "(rule-based fallback — ...)", the text after the dash tells you the exact error (missing key, wrong model, quota, etc).

## 3. What works out of the box (no extra setup)

- ECG demo library (5 built-in cases) + CSV/TXT upload
- Real noise-reduction pipeline: bandpass + 50Hz notch (powerline) + baseline-wander correction, with a before/after view and estimated SNR improvement
- ML classification (scikit-learn RandomForest, auto-upgrades to XGBoost if installed) with confidence + risk level
- Rule-based Explainable AI
- **Dataset & Validation** page — shows exactly what data trained the model, class balance, sample rows, and held-out accuracy/confusion matrix
- **Emergency Contacts** page — quick-reference numbers, shown automatically on any High-risk result
- Weekly progress reports (SQLite + Markdown/PDF export)
- Full English/Hindi UI, bright accessible color palette
- Live simulation mode
- In-browser mic speech-to-text (Chrome/Edge) + spoken answers (gTTS)

## 4. What needs the Gemini key (or falls back gracefully without it)

- AI Health Chatbot and Voice Assistant use Gemini (`gemini-3.1-flash-lite`, free tier) when `GEMINI_API_KEY` is set, otherwise a small built-in rule-based Q&A answers instead — never a hard error.

## 5. Project structure

```
ecgenius/
├── app.py
├── requirements.txt
├── .streamlit/config.toml   # bright theme
├── data/                    # created at runtime: ecgenius.db
└── modules/
    ├── signal_utils.py      # synthetic ECG, filtering, notch filter, peak detection
    ├── classifier.py        # feature extraction, training, held-out evaluation
    ├── explainability.py    # rule-based + optional SHAP
    ├── voice_assistant.py   # gTTS text-to-speech
    ├── chatbot.py           # Gemini / Claude / Ollama / rule-based, in that priority order
    ├── emergency.py         # emergency numbers + disclaimer
    ├── reports.py           # SQLite logging, weekly summary, PDF/Markdown export
    ├── demo_library.py      # 5 built-in demo cases
    └── i18n.py              # English/Hindi UI strings
```

## 6. Presenting at an ideathon

Suggested flow: **Overview → Demo Library ("Atrial Fibrillation") → Visualization
→ Classification & Alerts (point out the Emergency Contacts callout on High risk)
→ Explainable AI → Signal Quality & Noise Reduction → Voice Assistant (speak a
question) → Chatbot → Dataset & Validation (be upfront it's synthetic) → Weekly
Reports → Future Expansion.**
