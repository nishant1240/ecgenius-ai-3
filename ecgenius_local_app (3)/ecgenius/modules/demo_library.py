"""ECG Demo Library: pre-defined cases so judges can test without uploading files."""
from . import signal_utils as su

DEMO_CASES = [
    {"key": "Normal", "en": "Case 1 — Normal ECG", "hi": "मामला 1 — सामान्य ईसीजी"},
    {"key": "Arrhythmia", "en": "Case 2 — Arrhythmia", "hi": "मामला 2 — अतालता"},
    {"key": "AFib", "en": "Case 3 — Atrial Fibrillation", "hi": "मामला 3 — आलिंद फिब्रिलेशन"},
    {"key": "Brady", "en": "Case 4 — Bradycardia", "hi": "मामला 4 — ब्रैडीकार्डिया"},
    {"key": "Tachy", "en": "Case 5 — Tachycardia", "hi": "मामला 5 — टैचीकार्डिया"},
]


def get_case_signal(key: str, seconds: float = 10.0):
    return su.generate_synthetic_ecg(key, seconds=seconds)
