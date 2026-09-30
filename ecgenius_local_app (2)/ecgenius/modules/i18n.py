"""Minimal English/Hindi translation dictionary for static UI labels."""

STR = {
    "en": {
        "app_title": "ECGENIUS AI",
        "app_tag": "AI-Powered ECG Screening & Health Assistant — Educational Prototype",
        "disclaimer": "⚕️ This is an educational/research screening prototype. It is NOT a medical diagnostic device. Always consult a qualified healthcare professional.",
        "nav_overview": "🏠 Overview",
        "nav_source": "📤 Upload & Demo Library",
        "nav_viz": "📈 ECG Visualization & Live Simulation",
        "nav_class": "🧠 AI Classification & Alerts",
        "nav_xai": "🔍 Explainable AI",
        "nav_voice": "🎙️ Voice Assistant",
        "nav_chat": "💬 AI Health Chatbot",
        "nav_reports": "📊 Weekly Progress Reports",
        "nav_settings": "🌐 Multilingual Settings",
        "nav_noise": "🧹 Signal Quality & Noise Reduction",
        "nav_dataset": "📂 Dataset & Validation",
        "nav_emergency": "🚨 Emergency Contacts",
        "nav_future": "🔮 Future Expansion",
        "no_signal": "No ECG loaded yet. Go to 'Upload & Demo Library' to load a case.",
    },
    "hi": {
        "app_title": "ईसीजीनियस एआई",
        "app_tag": "एआई-संचालित ईसीजी स्क्रीनिंग और स्वास्थ्य सहायक — शैक्षिक प्रोटोटाइप",
        "disclaimer": "⚕️ यह एक शैक्षिक/अनुसंधान स्क्रीनिंग प्रोटोटाइप है। यह चिकित्सा निदान उपकरण नहीं है। कृपया किसी योग्य स्वास्थ्य विशेषज्ञ से सलाह लें।",
        "nav_overview": "🏠 अवलोकन",
        "nav_source": "📤 अपलोड और डेमो लाइब्रेरी",
        "nav_viz": "📈 ईसीजी विज़ुअलाइज़ेशन और लाइव सिमुलेशन",
        "nav_class": "🧠 एआई वर्गीकरण और अलर्ट",
        "nav_xai": "🔍 व्याख्यात्मक एआई",
        "nav_voice": "🎙️ आवाज़ सहायक",
        "nav_chat": "💬 एआई स्वास्थ्य चैटबॉट",
        "nav_reports": "📊 साप्ताहिक प्रगति रिपोर्ट",
        "nav_settings": "🌐 बहुभाषी सेटिंग्स",
        "nav_noise": "🧹 सिग्नल गुणवत्ता और शोर में कमी",
        "nav_dataset": "📂 डेटासेट और सत्यापन",
        "nav_emergency": "🚨 आपातकालीन संपर्क",
        "nav_future": "🔮 भविष्य विस्तार",
        "no_signal": "अभी तक कोई ईसीजी लोड नहीं हुआ। कृपया 'अपलोड और डेमो लाइब्रेरी' पर जाएं।",
    },
}


def t(key: str, lang: str = "en") -> str:
    return STR.get(lang, STR["en"]).get(key, key)
