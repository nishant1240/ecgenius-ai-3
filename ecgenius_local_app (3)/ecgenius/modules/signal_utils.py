"""
ECG signal utilities: synthetic generation (for demo library / model training),
filtering, R-peak detection, and file loading (CSV/TXT, optional WFDB).
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, iirnotch

FS = 250  # sampling frequency (Hz) used across the app

CASE_TYPES = {
    "Normal": "Normal ECG",
    "Arrhythmia": "Arrhythmia",
    "AFib": "Atrial Fibrillation",
    "Brady": "Bradycardia",
    "Tachy": "Tachycardia",
}


def _beat_shape(phase, p_wave=True, qrs_width=0.012, qrs_scale=1.0):
    """One synthetic PQRST cycle as a sum of Gaussian bumps. phase in [0,1)."""
    def g(c, w, a):
        return a * np.exp(-((phase - c) ** 2) / (2 * w ** 2))
    v = 0.0
    if p_wave:
        v += g(0.18, 0.025, 0.15)
    v += g(0.375, 0.010, -0.15 * qrs_scale)
    v += g(0.40, qrs_width, 1.0 * qrs_scale)
    v += g(0.425, 0.012, -0.25 * qrs_scale)
    v += g(0.62, 0.045, 0.30)
    return v


def generate_synthetic_ecg(kind: str, seconds: float = 10.0, fs: int = FS, seed=None) -> np.ndarray:
    """Generate a labeled synthetic ECG signal for a given class.

    NOTE: this is a lightweight stand-in for real patient data, used so the
    demo library and the classifier can run fully offline. Swap in real
    MIT-BIH / PhysioNet records (see load_wfdb_record) for production use.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    arr = np.zeros(n, dtype=np.float32)

    params = {
        "Normal":     dict(rr_mean=0.83, rr_jitter=0.02,  noise=0.010, ectopic=0.0,  p_wave=True,  wide_p=0.0),
        "Brady":      dict(rr_mean=1.15, rr_jitter=0.02,  noise=0.010, ectopic=0.0,  p_wave=True,  wide_p=0.0),
        "Tachy":      dict(rr_mean=0.48, rr_jitter=0.015, noise=0.015, ectopic=0.0,  p_wave=True,  wide_p=0.0),
        "Arrhythmia": dict(rr_mean=0.80, rr_jitter=0.14,  noise=0.020, ectopic=0.15, p_wave=True,  wide_p=0.4),
        "AFib":       dict(rr_mean=0.70, rr_jitter=0.28,  noise=0.050, ectopic=0.0,  p_wave=False, wide_p=0.0),
    }.get(kind, None)
    if params is None:
        raise ValueError(f"Unknown ECG kind: {kind}")

    idx = 0
    while idx < n:
        rr = max(0.28, params["rr_mean"] + rng.uniform(-1, 1) * params["rr_jitter"])
        beat_samples = int(round(rr * fs))
        is_ectopic = rng.random() < params["ectopic"]
        p_wave = params["p_wave"] and not is_ectopic
        qrs_width = 0.026 if (params["wide_p"] and rng.random() < params["wide_p"]) else 0.012
        qrs_scale = 1.3 if is_ectopic else 1.0
        for i in range(beat_samples):
            if idx >= n:
                break
            phase = i / beat_samples
            val = _beat_shape(phase, p_wave, qrs_width, qrs_scale)
            val += rng.uniform(-1, 1) * params["noise"]
            if kind == "AFib":
                val += rng.uniform(-1, 1) * 0.04  # fibrillatory baseline wobble
            arr[idx] = val
            idx += 1
    return arr


def bandpass_filter(sig: np.ndarray, fs: int = FS, low=0.5, high=40.0, order=3) -> np.ndarray:
    """Simple Butterworth bandpass filter to clean baseline wander / high-freq noise."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig  # signal too short to filter; return unfiltered


def notch_filter(sig: np.ndarray, fs: int = FS, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Remove powerline interference (mains hum) at a single frequency (50Hz in
    India/EU, 60Hz in the US). This is the classic third noise source that a
    plain bandpass filter does NOT remove, since 50/60Hz sits inside the
    0.5-40Hz passband alongside real ECG content."""
    nyq = fs / 2.0
    if freq >= nyq:
        return sig
    b, a = iirnotch(freq / nyq, q)
    try:
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def median_baseline_removal(sig: np.ndarray, fs: int = FS, window_s: float = 0.6) -> np.ndarray:
    """Estimate and subtract slow baseline wander (breathing/movement) using a
    sliding median, which removes drift without distorting the QRS shape the
    way a steep high-pass filter can."""
    import pandas as pd
    win = max(3, int(window_s * fs) | 1)  # odd window
    baseline = pd.Series(sig).rolling(win, center=True, min_periods=1).median().to_numpy()
    return sig - baseline


def estimate_snr(raw: np.ndarray, clean: np.ndarray) -> float:
    """Rough signal-to-noise ratio improvement estimate (dB), comparing the
    energy of what filtering removed against the cleaned signal's energy."""
    if len(raw) != len(clean) or len(raw) == 0:
        return 0.0
    noise = raw - clean
    sig_power = float(np.mean(clean ** 2)) + 1e-12
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    return 10 * np.log10(sig_power / noise_power)


def clean_signal(raw: np.ndarray, fs: int = FS, mains_freq: float = 50.0):
    """Full noise-reduction pipeline used everywhere in the app:
      1) Bandpass (0.5-40Hz)   -> removes baseline wander + high-freq EMG noise
      2) Notch (50/60Hz)       -> removes powerline interference
      3) Median baseline pass  -> mops up any remaining slow drift
    Returns (clean_signal, steps_applied: list[str], snr_db: float).
    """
    step1 = bandpass_filter(raw, fs=fs)
    step2 = notch_filter(step1, fs=fs, freq=mains_freq)
    step3 = median_baseline_removal(step2, fs=fs)
    snr = estimate_snr(raw, step3)
    steps = [
        "Bandpass filter (0.5–40 Hz) — removes baseline wander and muscle (EMG) noise",
        f"Notch filter ({mains_freq:.0f} Hz) — removes powerline interference",
        "Sliding-median baseline correction — removes residual slow drift",
    ]
    return step3, steps, snr


def detect_r_peaks(sig: np.ndarray, fs: int = FS):
    """Basic R-peak detector: adaptive threshold + minimum refractory distance."""
    if len(sig) < 3:
        return np.array([], dtype=int)
    mean = np.mean(sig)
    sd = np.std(sig)
    thresh = mean + 0.7 * sd
    distance = max(1, int(0.25 * fs))  # refractory ~250ms
    peaks, _ = find_peaks(sig, height=thresh, distance=distance)
    return peaks


def parse_uploaded_text(text: str) -> np.ndarray:
    """Parse a CSV/TXT upload: takes all numeric tokens, normalizes to ~[-0.2, 1.0]."""
    import re
    tokens = re.split(r"[,\s;]+", text.strip())
    nums = []
    for t in tokens:
        try:
            nums.append(float(t))
        except ValueError:
            continue
    if len(nums) < 20:
        return None
    arr = np.array(nums, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    rng = (hi - lo) or 1.0
    return ((arr - lo) / rng) * 1.2 - 0.2


def load_wfdb_record(record_path: str):
    """Optional: load a real PhysioNet/MIT-BIH record via the `wfdb` package.

    Requires: pip install wfdb, and either a local record or internet access
    to PhysioNet. Returns (signal, fs) or raises if wfdb is not installed.
    """
    import wfdb  # noqa: local import so the app runs without this optional dep
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal[:, 0].astype(np.float32)
    fs = record.fs
    return sig, fs
