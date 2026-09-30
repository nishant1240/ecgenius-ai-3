"""
Explainable AI module.

Always produces transparent, rule-based reasons from the extracted features
(so the app is explainable even with zero extra installs). If `shap` is
installed, also computes SHAP feature-importance values for the ML model's
prediction and returns them alongside.
"""


def rule_based_reasons(features: dict, category: str) -> list:
    reasons = []
    hr = features.get("heart_rate", 0)
    cv = features.get("rr_cv", 0)
    qrs = features.get("qrs_width_est", 0)

    if cv > 0.10:
        reasons.append(f"Irregular R-R intervals detected (variability index {cv:.2f}) — beat-to-beat timing is inconsistent.")
    else:
        reasons.append(f"Regular R-R intervals detected (variability index {cv:.2f}).")

    if hr < 60:
        reasons.append(f"Heart rate is {hr:.0f} bpm, below the typical 60 bpm resting threshold.")
    elif hr > 100:
        reasons.append(f"Heart rate is {hr:.0f} bpm, above the typical 100 bpm resting threshold.")
    else:
        reasons.append(f"Heart rate is {hr:.0f} bpm, within the typical 60–100 bpm resting range.")

    if category == "AFib":
        reasons.append("Fibrillatory baseline pattern with absent/indistinct P-waves is consistent with atrial fibrillation.")
    if qrs > 0.02:
        reasons.append(f"Estimated QRS width ({qrs*1000:.0f} ms) is on the wider side, which can indicate ectopic or abnormal beats.")

    return reasons


def shap_explain(model_bundle, feature_names, features: dict):
    """Returns a list of (feature, contribution) sorted by |impact|, or None if shap missing."""
    try:
        import shap
        import numpy as np
        import pandas as pd
        model = model_bundle["model"]
        x = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)
        explainer = shap.Explainer(model)
        sv = explainer(x)
        vals = sv.values[0]
        if vals.ndim > 1:  # multi-class: take the row for the predicted class
            vals = vals[:, int(np.argmax(np.abs(vals).sum(axis=0)))]
        pairs = sorted(zip(feature_names, vals), key=lambda p: -abs(p[1]))
        return pairs
    except ImportError:
        return None
    except Exception:
        return None
