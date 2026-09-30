"""
Emergency Contacts module.

Static, India-focused emergency numbers (swap/extend EMERGENCY_NUMBERS for
other regions), plus a personal contact list the user can note down for the
session. This is informational only — it does NOT call, text, or dispatch
anyone. It exists so a High-risk alert always points the user to real help.
"""

EMERGENCY_NUMBERS = {
    "en": [
        {"label": "National Emergency Number (India)", "number": "112"},
        {"label": "Ambulance", "number": "108"},
        {"label": "Ambulance (alt.)", "number": "102"},
        {"label": "Police", "number": "100"},
        {"label": "Fire", "number": "101"},
    ],
    "hi": [
        {"label": "राष्ट्रीय आपातकालीन नंबर (भारत)", "number": "112"},
        {"label": "एम्बुलेंस", "number": "108"},
        {"label": "एम्बुलेंस (वैकल्पिक)", "number": "102"},
        {"label": "पुलिस", "number": "100"},
        {"label": "फायर ब्रिगेड", "number": "101"},
    ],
}

DISCLAIMER = {
    "en": ("These numbers are shown for quick reference only — this app cannot "
           "call anyone automatically. If you or someone near you is experiencing "
           "chest pain, severe shortness of breath, fainting, or any emergency "
           "symptom, call emergency services immediately. Do not wait for or rely "
           "on this app's classification."),
    "hi": ("ये नंबर केवल त्वरित संदर्भ के लिए दिखाए गए हैं — यह ऐप स्वचालित रूप से "
           "किसी को कॉल नहीं कर सकता। यदि आपको या आपके पास किसी को सीने में दर्द, "
           "सांस लेने में गंभीर कठिनाई, बेहोशी, या कोई भी आपातकालीन लक्षण हो, तो तुरंत "
           "आपातकालीन सेवाओं को कॉल करें। इस ऐप के वर्गीकरण पर निर्भर न रहें।"),
}
