"""
ECGENIUS AI — locally-runnable Streamlit prototype.

Run with:
    streamlit run app.py

Educational/research prototype only. NOT a medical diagnostic device.
"""
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from modules import signal_utils as su
from modules import classifier as clf
from modules import explainability as xai
from modules import voice_assistant as va
from modules import chatbot as bot
from modules import reports as rep
from modules import demo_library as lib
from modules import emergency as em
from modules.i18n import t

st.set_page_config(page_title="ECGENIUS AI", page_icon="💓", layout="wide")

# ---------------------------------------------------------------- session state
defaults = {
    "lang": "en",
    "raw_signal": None,
    "signal": None,
    "noise_steps": None,
    "snr_db": None,
    "peaks": None,
    "features": None,
    "category": None,
    "confidence": None,
    "prob_dict": None,
    "risk": None,
    "source_label": None,
    "history_added": False,
    "chat_log": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


@st.cache_resource(show_spinner="Training ECG classifier on synthetic dataset (first run only)...")
def get_model():
    df = clf.build_training_set(n_per_class=120)
    bundle, backend, classes = clf.train_model(df)
    return bundle, backend, classes


MODEL_BUNDLE, MODEL_BACKEND, MODEL_CLASSES = get_model()
rep.init_db()

L = st.session_state.lang


def analyze_signal(sig: np.ndarray, source_label: str):
    raw = sig
    filtered, noise_steps, snr_db = su.clean_signal(raw)
    peaks = su.detect_r_peaks(filtered)
    features = clf.extract_features(filtered, peaks)
    category, confidence, prob_dict = clf.predict(MODEL_BUNDLE, MODEL_BACKEND, MODEL_CLASSES, features)
    risk = clf.risk_level(category, features)

    st.session_state.raw_signal = raw
    st.session_state.signal = filtered
    st.session_state.noise_steps = noise_steps
    st.session_state.snr_db = snr_db
    st.session_state.peaks = peaks
    st.session_state.features = features
    st.session_state.category = category
    st.session_state.confidence = confidence
    st.session_state.prob_dict = prob_dict
    st.session_state.risk = risk
    st.session_state.source_label = source_label

    rep.log_analysis(source_label, features["heart_rate"], category, risk, confidence)


# ---------------------------------------------------------------- nav keys (stable across language switches)
NAV_ITEMS = [
    ("overview", "nav_overview"),
    ("source", "nav_source"),
    ("viz", "nav_viz"),
    ("class", "nav_class"),
    ("xai", "nav_xai"),
    ("noise", "nav_noise"),
    ("voice", "nav_voice"),
    ("chat", "nav_chat"),
    ("reports", "nav_reports"),
    ("dataset", "nav_dataset"),
    ("emergency", "nav_emergency"),
    ("settings", "nav_settings"),
    ("future", "nav_future"),
]
if "page" not in st.session_state:
    st.session_state.page = "overview"

# button styling + bright medical-dashboard palette
st.markdown("""
<style>
:root {
    --primary: #4ECDC4;     /* teal */
    --secondary: #5B8DEF;   /* blue */
    --bg: #F8FAFC;          /* light background */
    --surface: #FFFFFF;
    --success: #34D399;
    --warning: #FB923C;
    --critical: #F87171;
    --text: #1F2937;
}
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stHeader"] { background: var(--bg) !important; }
.stApp p, .stApp li, .stApp label, .stApp span { color: var(--text); }
[data-testid="stTextInput"] input, [data-testid="stChatInput"] textarea,
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #FFFFFF !important; color: var(--text) !important;
    border: 1px solid #CBD5E1 !important; border-radius: 8px !important;
}
.stButton > button {
    background: var(--primary) !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important; font-weight: 500 !important;
}
.stButton > button:hover { background: var(--secondary) !important; }

h1, h2, h3 { color: var(--secondary) !important; }
h4, h5, strong { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] { color: #6B7280 !important; }
hr { border-color: #E2E8F0 !important; }

/* sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #F0FDFA 100%);
    border-right: 1px solid #E2E8F0;
}
section[data-testid="stSidebar"] button {
    text-align: left !important;
    justify-content: flex-start !important;
    border-radius: 8px !important;
    margin-bottom: 4px !important;
    font-weight: 400 !important;
    background: #FFFFFF !important;
    color: var(--text) !important;
    border: 1px solid #E2E8F0 !important;
}
section[data-testid="stSidebar"] button:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}
section[data-testid="stSidebar"] button[kind="primary"] {
    font-weight: 700 !important;
    background: var(--primary) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--primary) !important;
}

/* main-area buttons */
.stButton > button[kind="primary"], .stDownloadButton > button {
    background: var(--secondary) !important; color: #fff !important; border: none !important;
}
.stDownloadButton > button:hover { background: var(--primary) !important; }

/* alerts, chat, metrics */
[data-testid="stMetricValue"] { color: var(--secondary); }
[data-testid="stChatMessage"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
}
[data-testid="stAlert"] { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- sidebar / nav
with st.sidebar:
    st.markdown("### 💓 ECGENIUS AI")

    lc1, lc2 = st.columns(2)
    if lc1.button("English", use_container_width=True,
                  type="primary" if st.session_state.lang == "en" else "secondary"):
        st.session_state.lang = "en"
        st.rerun()
    if lc2.button("हिंदी", use_container_width=True,
                  type="primary" if st.session_state.lang == "hi" else "secondary"):
        st.session_state.lang = "hi"
        st.rerun()

    L = st.session_state.lang
    st.caption(f"Classifier backend: **{MODEL_BACKEND}**")
    st.divider()
    st.caption("Navigate" if L == "en" else "नेविगेट करें")

    for key, label_key in NAV_ITEMS:
        if st.button(t(label_key, L), key=f"nav_{key}", use_container_width=True,
                     type="primary" if st.session_state.page == key else "secondary"):
            st.session_state.page = key
            st.rerun()

page = st.session_state.page

st.markdown(f"## {t('app_title', L)}")
st.caption(t("app_tag", L))
st.warning(t("disclaimer", L))

# ================================================================== OVERVIEW
if page == "overview":
    st.subheader("Project Overview" if L == "en" else "परियोजना अवलोकन")
    st.write(
        "ECGENIUS AI is a locally-run, section-by-section implementation of the full spec: "
        "ECG processing, visualization, AI classification, explainability, emergency alerts, "
        "a voice assistant, a health chatbot, weekly reports, multilingual support, a demo "
        "library, and live simulation — plus a roadmap for real hardware integration."
        if L == "en" else
        "ईसीजीनियस एआई पूरे स्पेक का एक स्थानीय रूप से चलने वाला कार्यान्वयन है: ईसीजी प्रोसेसिंग, "
        "विज़ुअलाइज़ेशन, एआई वर्गीकरण, व्याख्या, आपातकालीन अलर्ट, आवाज़ सहायक, स्वास्थ्य चैटबॉट, "
        "साप्ताहिक रिपोर्ट, बहुभाषी समर्थन, डेमो लाइब्रेरी और लाइव सिमुलेशन।"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Classifier", MODEL_BACKEND)
    c2.metric("Classes", len(MODEL_CLASSES))
    hist = rep.load_history()
    c3.metric("Analyses logged" if L == "en" else "विश्लेषण दर्ज", len(hist))
    st.info(
        "Use the sidebar to move through every section. Start with **Upload & Demo Library** "
        "to load an ECG, then check Classification, Explainability, Voice, Chatbot, and Reports."
        if L == "en" else
        "हर सेक्शन देखने के लिए साइडबार का उपयोग करें। पहले 'अपलोड और डेमो लाइब्रेरी' से एक ईसीजी लोड करें।"
    )

# ================================================================== SOURCE (upload + demo library)
elif page == "source":
    st.subheader(t("nav_source", L))
    tab1, tab2 = st.tabs(["📚 Demo Library", "📤 Upload File"] if L == "en" else ["📚 डेमो लाइब्रेरी", "📤 फ़ाइल अपलोड करें"])

    with tab1:
        st.caption("Built-in cases so judges can test functionality without uploading files.")
        cols = st.columns(len(lib.DEMO_CASES))
        for i, case in enumerate(lib.DEMO_CASES):
            label = case[L]
            if cols[i].button(label, use_container_width=True):
                sig = lib.get_case_signal(case["key"])
                analyze_signal(sig, label)
                st.success(f"Loaded: {label}")

    with tab2:
        st.caption("Supported: CSV / TXT with numeric ECG samples (one column or comma/space separated).")
        uploaded = st.file_uploader("Upload ECG file", type=["csv", "txt"])
        if uploaded is not None:
            text = uploaded.read().decode("utf-8", errors="ignore")
            sig = su.parse_uploaded_text(text)
            if sig is None:
                st.error("Could not find enough numeric samples in this file.")
            else:
                analyze_signal(sig, f"Uploaded: {uploaded.name}")
                st.success(f"Loaded and analyzed: {uploaded.name}")
        st.caption("WFDB (.dat/.hea) records from MIT-BIH/PhysioNet are also supported via "
                   "`modules.signal_utils.load_wfdb_record` if you `pip install wfdb`.")

# ================================================================== VISUALIZATION / LIVE SIM
elif page == "viz":
    st.subheader(t("nav_viz", L))
    if st.session_state.signal is None:
        st.info(t("no_signal", L))
    else:
        sig = st.session_state.signal
        peaks = st.session_state.peaks
        mode = st.radio("Mode", ["Static waveform", "Live simulation"] if L == "en" else ["स्थिर तरंग", "लाइव सिमुलेशन"], horizontal=True)

        if mode in ("Static waveform", "स्थिर तरंग"):
            fig = go.Figure()
            t_axis = np.arange(len(sig)) / su.FS
            fig.add_trace(go.Scatter(x=t_axis, y=sig, mode="lines", name="Filtered ECG",
                                      line=dict(color="#4ECDC4", width=1.5)))
            fig.add_trace(go.Scatter(x=peaks / su.FS, y=sig[peaks], mode="markers", name="R-peaks",
                                      marker=dict(color="#F87171", size=7)))
            fig.update_layout(height=380, xaxis_title="Time (s)", yaxis_title="Amplitude (mV, normalized)",
                               plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color="#1F2937",
                               margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Prerecorded signal scrolled continuously to mimic a hospital monitor.")
            speed = st.slider("Playback speed (windows/sec)" if L == "en" else "गति", 1, 10, 4)
            placeholder = st.empty()
            window = su.FS * 6
            run = st.toggle("▶ Play" if L == "en" else "▶ चलाएं", value=False)
            if run:
                import time
                offset = 0
                for _ in range(60):  # bounded loop so the page doesn't hang forever
                    idx = (np.arange(offset, offset + window) % len(sig))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=sig[idx], mode="lines", line=dict(color="#4ECDC4", width=1.5)))
                    fig.update_layout(height=300, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                                       font_color="#1F2937", margin=dict(l=10, r=10, t=10, b=10),
                                       xaxis=dict(visible=False), yaxis=dict(visible=False))
                    placeholder.plotly_chart(fig, use_container_width=True, key=f"live_{_}")
                    offset = (offset + su.FS // speed) % len(sig)
                    time.sleep(1.0 / speed)

# ================================================================== CLASSIFICATION & ALERTS
elif page == "class":
    st.subheader(t("nav_class", L))
    if st.session_state.category is None:
        st.info(t("no_signal", L))
    else:
        cat = st.session_state.category
        risk = st.session_state.risk
        feats = st.session_state.features
        color = {"Low": "🟢", "Moderate": "🟠", "High": "🔴"}[risk]

        c1, c2, c3 = st.columns(3)
        c1.metric("Category" if L == "en" else "श्रेणी", su.CASE_TYPES.get(cat, cat))
        c2.metric("Heart Rate" if L == "en" else "हृदय गति", f"{feats['heart_rate']:.0f} bpm")
        c3.metric("Confidence" if L == "en" else "विश्वास", f"{st.session_state.confidence*100:.0f}%")

        if risk == "Low":
            st.success(f"{color} Risk: {risk} — Pattern consistent with normal sinus rhythm (model estimate).")
        else:
            st.error(f"{color} Risk: {risk} — ⚠️ Potential abnormal ECG pattern detected. "
                     "This system is an educational prototype and cannot provide medical diagnosis. "
                     "Please consult a qualified healthcare professional.")
        if risk == "High":
            with st.container(border=True):
                st.markdown("#### 🚨 " + ("Emergency Contacts" if L == "en" else "आपातकालीन संपर्क"))
                for c in em.EMERGENCY_NUMBERS[L]:
                    st.write(f"**{c['label']}:** {c['number']}")
                st.caption(em.DISCLAIMER[L])

        st.write("**Class probabilities:**" if L == "en" else "**श्रेणी संभावनाएं:**")
        prob_df = pd.DataFrame({
            "Class": [su.CASE_TYPES.get(k, k) for k in st.session_state.prob_dict.keys()],
            "Probability": list(st.session_state.prob_dict.values()),
        }).sort_values("Probability", ascending=False)
        st.bar_chart(prob_df.set_index("Class"))

# ================================================================== EXPLAINABLE AI
elif page == "xai":
    st.subheader(t("nav_xai", L))
    if st.session_state.features is None:
        st.info(t("no_signal", L))
    else:
        reasons = xai.rule_based_reasons(st.session_state.features, st.session_state.category)
        st.write("**Reasons for this prediction:**" if L == "en" else "**इस भविष्यवाणी के कारण:**")
        for r in reasons:
            st.markdown(f"- {r}")

        shap_pairs = xai.shap_explain(MODEL_BUNDLE, clf.FEATURE_NAMES, st.session_state.features)
        if shap_pairs:
            st.write("**SHAP feature contributions:**")
            shap_df = pd.DataFrame(shap_pairs, columns=["Feature", "Impact"]).set_index("Feature")
            st.bar_chart(shap_df)
        else:
            st.caption("Install `shap` (`pip install shap`) to also see quantitative SHAP feature "
                       "attributions here, in addition to the rule-based reasons above.")

# ================================================================== SIGNAL QUALITY / NOISE REDUCTION
elif page == "noise":
    st.subheader(t("nav_noise", L))
    if st.session_state.raw_signal is None:
        st.info(t("no_signal", L))
    else:
        st.write("**Noise-reduction steps applied to every signal:**" if L == "en" else
                  "**हर सिग्नल पर लागू शोर-कमी चरण:**")
        for i, step in enumerate(st.session_state.noise_steps, 1):
            st.markdown(f"{i}. {step}")

        st.metric("Estimated SNR improvement" if L == "en" else "अनुमानित SNR सुधार",
                   f"{st.session_state.snr_db:.1f} dB")

        st.write("**Before vs. after (first 5 seconds):**" if L == "en" else
                  "**पहले बनाम बाद (पहले 5 सेकंड):**")
        n = min(len(st.session_state.raw_signal), su.FS * 5)
        t_axis = np.arange(n) / su.FS
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_axis, y=st.session_state.raw_signal[:n], mode="lines",
                                  name="Raw", line=dict(color="#F87171", width=1)))
        fig.add_trace(go.Scatter(x=t_axis, y=st.session_state.signal[:n], mode="lines",
                                  name="Cleaned", line=dict(color="#4ECDC4", width=1.5)))
        fig.update_layout(height=340, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                           font_color="#1F2937", xaxis_title="Time (s)",
                           margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Why these steps? (the 4 main ECG noise sources)" if L == "en" else
                          "ये चरण क्यों? (4 मुख्य ईसीजी शोर स्रोत)"):
            st.markdown("""
| Noise type | Cause | Fix used here |
|---|---|---|
| Baseline wander | Breathing, body movement | Bandpass + median baseline correction |
| Powerline interference | AC mains (50/60Hz) | Notch filter |
| Muscle artifact (EMG) | Muscle movement/tension | Bandpass filter (high-frequency cutoff) |
| Electrode motion | Loose/moving electrode contact | Mainly a hardware fix — good skin contact, shielded leads, right-leg-drive circuit on real hardware |
            """ if L == "en" else """
| शोर प्रकार | कारण | यहाँ उपयोग किया गया समाधान |
|---|---|---|
| बेसलाइन वांडर | सांस लेना, शरीर की गति | बैंडपास + मीडियन बेसलाइन सुधार |
| पावरलाइन हस्तक्षेप | AC मेंस (50/60Hz) | नॉच फ़िल्टर |
| मांसपेशी कलाकृति (EMG) | मांसपेशी गति/तनाव | बैंडपास फ़िल्टर |
| इलेक्ट्रोड गति | ढीला/हिलता इलेक्ट्रोड संपर्क | मुख्यतः हार्डवेयर समाधान |
            """)

# ================================================================== VOICE ASSISTANT
elif page == "voice":
    st.subheader(t("nav_voice", L))
    st.caption("Speak directly into your microphone (use Chrome or Edge and allow mic access). "
               "The answer is spoken back to you." if L == "en" else
               "सीधे माइक्रोफ़ोन में बोलें (Chrome या Edge उपयोग करें)। उत्तर बोलकर सुनाया जाएगा।")

    st.markdown("**🔊 Hear the current result**" if L == "en" else "**🔊 वर्तमान परिणाम सुनें**")
    if st.session_state.category is None:
        st.info(t("no_signal", L))
    else:
        if st.button("Speak result" if L == "en" else "परिणाम बोलें"):
            reasons = xai.rule_based_reasons(st.session_state.features, st.session_state.category)
            text = f"{su.CASE_TYPES.get(st.session_state.category)}. {reasons[0]}"
            audio_bytes, err = va.speak_text(text, L)
            if err:
                st.warning(err)
            else:
                st.audio(audio_bytes, format="audio/mp3", autoplay=True)

    st.divider()
    st.markdown("**🎤 Ask a question by voice**" if L == "en" else "**🎤 आवाज़ से प्रश्न पूछें**")
    from streamlit_mic_recorder import speech_to_text
    spoken = speech_to_text(
        language="hi-IN" if L == "hi" else "en-US",
        start_prompt="🎤 Click and speak" if L == "en" else "🎤 क्लिक करें और बोलें",
        stop_prompt="⏹ Stop" if L == "en" else "⏹ रोकें",
        just_once=True, use_container_width=True, key=f"stt_{L}",
    )
    if spoken:
        st.write(f"**You said:** {spoken}")
        with st.spinner("Thinking..."):
            answer, source = bot.get_answer(spoken, L)
        st.write(f"**Answer** ({source}): {answer}")
        audio_bytes, err = va.speak_text(answer, L)
        if err:
            st.warning(err)
        else:
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

# ================================================================== CHATBOT
elif page == "chat":
    st.subheader(t("nav_chat", L))

    for role, msg in st.session_state.chat_log:
        with st.chat_message(role):
            st.write(msg)

    q = st.chat_input("Ask about arrhythmia, heart rate, AFib..." if L == "en" else "अतालता, हृदय गति, AFib के बारे में पूछें...")
    if q:
        history = list(st.session_state.chat_log)
        st.session_state.chat_log.append(("user", q))
        with st.spinner("Thinking..."):
            answer, source = bot.get_answer(q, L, history)
        st.session_state.chat_log.append(("assistant", f"{answer}  \n_({source})_"))
        st.rerun()

# ================================================================== WEEKLY REPORTS
elif page == "reports":
    st.subheader(t("nav_reports", L))
    hist = rep.load_history()
    if hist.empty:
        st.info("No analyses logged yet. Analyze a demo case or upload first." if L == "en" else
                "अभी तक कोई विश्लेषण दर्ज नहीं। पहले एक मामला विश्लेषण करें।")
    else:
        summary = rep.weekly_summary(hist)
        c1, c2, c3 = st.columns(3)
        c1.metric("Analyses this week" if L == "en" else "इस सप्ताह विश्लेषण", summary["count"])
        c2.metric("Average HR" if L == "en" else "औसत HR", f"{summary['avg_hr']} bpm")
        c3.metric("Most common class" if L == "en" else "सबसे सामान्य श्रेणी",
                   su.CASE_TYPES.get(summary["most_common_class"], summary["most_common_class"]))

        st.write("**Trend (average HR per day):**" if L == "en" else "**प्रवृत्ति (प्रति दिन औसत HR):**")
        if len(summary["by_day"]) > 0:
            st.line_chart(summary["by_day"])

        st.write("**Full history:**" if L == "en" else "**पूरा इतिहास:**")
        st.dataframe(hist, use_container_width=True)

        if st.button("📄 Export weekly report" if L == "en" else "📄 साप्ताहिक रिपोर्ट निर्यात करें"):
            out = rep.export_pdf(summary, "/tmp/ecgenius_weekly_report.pdf")
            with open(out, "rb") as f:
                st.download_button("Download report", f, file_name=os.path.basename(out))

# ================================================================== DATASET & VALIDATION
elif page == "dataset":
    st.subheader(t("nav_dataset", L))
    st.write(
        "The classifier is trained on a **synthetically generated** ECG dataset, not real "
        "patient records. Full transparency on that below." if L == "en" else
        "वर्गीकरणकर्ता एक **सिंथेटिक रूप से जनरेट किए गए** ईसीजी डेटासेट पर प्रशिक्षित है, वास्तविक "
        "रोगी रिकॉर्ड पर नहीं। नीचे पूरी पारदर्शिता दी गई है।"
    )

    with st.spinner("Generating dataset preview..." if L == "en" else "डेटासेट पूर्वावलोकन बना रहे हैं..."):
        preview_df = clf.build_training_set(n_per_class=120)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total samples" if L == "en" else "कुल नमूने", len(preview_df))
    c2.metric("Classes" if L == "en" else "श्रेणियाँ", preview_df["label"].nunique())
    c3.metric("Samples / class" if L == "en" else "नमूने / श्रेणी", 120)

    st.write("**Class distribution:**" if L == "en" else "**श्रेणी वितरण:**")
    st.bar_chart(preview_df["label"].value_counts())

    st.write("**Sample rows (features extracted per signal):**" if L == "en" else
              "**नमूना पंक्तियाँ:**")
    st.dataframe(preview_df.sample(min(10, len(preview_df)), random_state=1), use_container_width=True)

    st.divider()
    st.write("**Held-out validation (25% test split, never seen during training):**" if L == "en" else
              "**होल्ड-आउट सत्यापन (25% टेस्ट स्प्लिट):**")
    with st.spinner("Running validation..." if L == "en" else "सत्यापन चल रहा है..."):
        metrics = clf.evaluate_model(preview_df)
    st.metric("Accuracy on held-out data" if L == "en" else "होल्ड-आउट डेटा पर सटीकता",
               f"{metrics['accuracy']*100:.1f}%")
    st.caption(f"Trained on {metrics['n_train']} samples, tested on {metrics['n_test']} unseen samples "
               f"using {metrics['backend']}." if L == "en" else
               f"{metrics['n_train']} नमूनों पर प्रशिक्षित, {metrics['n_test']} अनदेखे नमूनों पर परीक्षण किया गया।")

    cm_df = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
    st.write("**Confusion matrix** (rows = actual, columns = predicted):" if L == "en" else
              "**कन्फ्यूजन मैट्रिक्स:**")
    st.dataframe(cm_df, use_container_width=True)

    st.warning(
        "⚠️ Because both training and test data are synthetic, this accuracy reflects how well the "
        "model learned the synthetic signal generator's patterns — not real-world clinical accuracy. "
        "See Future Expansion for the path to training on real MIT-BIH/PhysioNet data."
        if L == "en" else
        "⚠️ चूंकि प्रशिक्षण और परीक्षण डेटा दोनों सिंथेटिक हैं, यह सटीकता वास्तविक नैदानिक सटीकता को नहीं दर्शाती।"
    )

# ================================================================== EMERGENCY CONTACTS
elif page == "emergency":
    st.subheader(t("nav_emergency", L))
    st.error(em.DISCLAIMER[L])
    for c in em.EMERGENCY_NUMBERS[L]:
        st.markdown(f"### {c['label']}: `{c['number']}`")
    st.divider()
    st.write("**Your personal emergency contacts (this session only, not saved to disk):**" if L == "en" else
              "**आपके व्यक्तिगत आपातकालीन संपर्क (केवल इस सत्र के लिए):**")
    if "personal_contacts" not in st.session_state:
        st.session_state.personal_contacts = []
    with st.form("add_contact", clear_on_submit=True):
        cc1, cc2 = st.columns(2)
        name = cc1.text_input("Name" if L == "en" else "नाम")
        number = cc2.text_input("Phone number" if L == "en" else "फ़ोन नंबर")
        if st.form_submit_button("Add" if L == "en" else "जोड़ें") and name and number:
            st.session_state.personal_contacts.append((name, number))
    for name, number in st.session_state.personal_contacts:
        st.write(f"📞 **{name}** — {number}")

# ================================================================== MULTILINGUAL SETTINGS
elif page == "settings":
    st.subheader(t("nav_settings", L))
    st.write("Currently supported: **English**, **हिंदी (Hindi)**." if L == "en" else
              "वर्तमान में समर्थित: **English**, **हिंदी**।")
    st.caption("Adding a language: add a new key to modules/i18n.py's STR dict and to the "
               "per-module translation tables (chatbot.py, and the reasons in explainability.py "
               "if you want localized XAI text too).")

# ================================================================== FUTURE EXPANSION
elif page == "future":
    st.subheader(t("nav_future", L))
    st.markdown("""
- **Real ECG hardware / wearables**: replace `signal_utils.generate_synthetic_ecg` and
  `parse_uploaded_text` with a live serial/BLE reader feeding the same `analyze_signal()` pipeline.
- **Real datasets**: use `signal_utils.load_wfdb_record` (`pip install wfdb`) to train/evaluate on
  MIT-BIH, PTB Diagnostic ECG, or other PhysioNet databases instead of synthetic signals.
- **Cloud deployment**: containerize this app (Dockerfile) and deploy to any Streamlit-compatible host.
- **Mobile app**: reuse the `modules/` package as a backend API (FastAPI) behind a mobile front-end.
- **IoT integration**: stream live vitals into the SQLite log used by the Weekly Reports section.
    """ if L == "en" else """
- **वास्तविक ईसीजी हार्डवेयर**: लाइव सेंसर डेटा को उसी `analyze_signal()` पाइपलाइन में फीड करें।
- **वास्तविक डेटासेट**: MIT-BIH/PTB के लिए `wfdb` का उपयोग करें।
- **क्लाउड डिप्लॉयमेंट**: इस ऐप को Docker में पैक करें।
- **मोबाइल ऐप**: `modules/` को FastAPI बैकएंड के रूप में पुनः उपयोग करें।
- **IoT एकीकरण**: लाइव वाइटल्स को SQLite लॉग में स्ट्रीम करें।
    """)

st.divider()
st.caption("ECGENIUS AI — laboratory-level educational/research prototype. Not a medical device. "
           "All outputs are for demonstration purposes only.")
