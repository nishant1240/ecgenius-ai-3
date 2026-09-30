"""
AI-based ECG classification.

Trains on synthetically generated, labeled ECG signals (see signal_utils) so
the app works fully offline out of the box. Prefers XGBoost if installed,
falls back to scikit-learn's RandomForestClassifier otherwise.

To train on real data instead: point `build_training_set` at real MIT-BIH /
PTB records loaded via signal_utils.load_wfdb_record, extract the same
features, and refit.
"""
import numpy as np
import pandas as pd

from . import signal_utils as su

FEATURE_NAMES = ["heart_rate", "rr_mean", "rr_std", "rr_cv", "qrs_width_est", "n_beats"]


def extract_features(sig: np.ndarray, peaks: np.ndarray, fs: int = su.FS) -> dict:
    if len(peaks) < 3:
        return dict(heart_rate=0.0, rr_mean=0.0, rr_std=0.0, rr_cv=0.0,
                    qrs_width_est=0.0, n_beats=len(peaks))
    rr = np.diff(peaks) / fs
    rr_mean = float(np.mean(rr))
    rr_std = float(np.std(rr))
    rr_cv = rr_std / rr_mean if rr_mean > 0 else 0.0
    hr = 60.0 / rr_mean if rr_mean > 0 else 0.0

    # crude QRS width estimate: samples around each peak above half the peak's amplitude
    widths = []
    for p in peaks:
        lo, hi = max(0, p - int(0.06 * fs)), min(len(sig), p + int(0.06 * fs))
        seg = sig[lo:hi]
        if len(seg) == 0:
            continue
        half = sig[p] / 2.0
        above = np.where(seg > half)[0]
        if len(above) > 0:
            widths.append((above[-1] - above[0]) / fs)
    qrs_width_est = float(np.mean(widths)) if widths else 0.03

    return dict(heart_rate=hr, rr_mean=rr_mean, rr_std=rr_std, rr_cv=rr_cv,
                qrs_width_est=qrs_width_est, n_beats=len(peaks))


def build_training_set(n_per_class: int = 120, seconds: float = 8.0) -> pd.DataFrame:
    """Generate a synthetic, labeled training set across all 5 classes."""
    rows = []
    for kind in su.CASE_TYPES.keys():
        for i in range(n_per_class):
            sig = su.generate_synthetic_ecg(kind, seconds=seconds, seed=i * 7 + hash(kind) % 1000)
            peaks = su.detect_r_peaks(sig)
            feats = extract_features(sig, peaks)
            feats["label"] = kind
            rows.append(feats)
    return pd.DataFrame(rows)


def evaluate_model(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> dict:
    """Train/test split evaluation so the app can show an honest accuracy
    number on held-out synthetic data (not the same rows used for training)."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix

    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["label"].astype(str).to_numpy(dtype=object)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    train_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    train_df["label"] = y_train
    bundle, backend, classes = train_model(train_df)

    preds = []
    for row in X_test:
        feats = dict(zip(FEATURE_NAMES, row))
        cat, _, _ = predict(bundle, backend, classes, feats)
        preds.append(cat)

    acc = accuracy_score(y_test, preds)
    labels_sorted = sorted(set(y_test) | set(preds))
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)
    return {
        "accuracy": acc,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "labels": labels_sorted,
        "confusion_matrix": cm,
        "backend": backend,
    }


def train_model(df: pd.DataFrame):
    """Train and return (model, backend_name, classes_)."""
    # .to_numpy(...) instead of .values: pandas >=3.0 defaults some columns to
    # Arrow-backed arrays, which scikit-learn's indexing can't handle directly.
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    y = df["label"].astype(str).to_numpy(dtype=object)

    try:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        model = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.15,
            objective="multi:softprob", eval_metric="mlogloss",
        )
        model.fit(X, y_enc)
        return {"model": model, "label_encoder": le}, "XGBoost", list(le.classes_)
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)
        model.fit(X, y)
        return {"model": model, "label_encoder": None}, "RandomForest (fallback)", list(model.classes_)


def predict(bundle, backend, classes, features: dict):
    """Return (category, confidence, probability_dict)."""
    x = np.array([[features[f] for f in FEATURE_NAMES]])
    model = bundle["model"]
    proba = model.predict_proba(x)[0]

    if bundle["label_encoder"] is not None:
        idx = int(np.argmax(proba))
        category = bundle["label_encoder"].inverse_transform([idx])[0]
        prob_dict = {c: float(p) for c, p in zip(bundle["label_encoder"].classes_, proba)}
    else:
        idx = int(np.argmax(proba))
        category = model.classes_[idx]
        prob_dict = {c: float(p) for c, p in zip(model.classes_, proba)}

    confidence = float(proba[idx])
    return category, confidence, prob_dict


def risk_level(category: str, features: dict) -> str:
    """Simple, transparent risk-banding on top of the ML category (mirrors spec's alert levels)."""
    hr = features.get("heart_rate", 0)
    cv = features.get("rr_cv", 0)
    if category == "Normal":
        return "Low"
    if category == "AFib":
        return "High"
    if category == "Arrhythmia":
        return "High" if cv > 0.16 else "Moderate"
    if category == "Brady":
        return "High" if hr < 45 else "Moderate"
    if category == "Tachy":
        return "High" if hr > 130 else "Moderate"
    return "Moderate"
