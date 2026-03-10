"""
Fixed data preparation and evaluation for marine acoustic autoresearch.
DO NOT MODIFY — this is the ground truth data loader and evaluation harness.

Adapted from Karpathy's autoresearch pattern for marine bioacoustics.
Runs on Apple Silicon (MPS), CPU, or CUDA — auto-detected.

Data: SoundTrap hydrophone recordings from the Gulf of San Cristobal, Galapagos.
  - Unit 5783: 144 kHz, ~20 min files
  - Unit 6478: 96 kHz, ~10 min files
  - Pilot: 48 kHz, ~5 min files
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_device():
    """Auto-detect best available compute device."""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"

DEVICE = detect_device()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TARGET_SR = 48000           # resample everything to 48 kHz for consistency
SEGMENT_SECONDS = 10        # each analysis segment is 10 seconds
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_SECONDS
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 50                  # min freq Hz — below is self-noise
F_MAX = 24000               # max freq Hz — Nyquist at 48kHz

TIME_BUDGET = 180           # experiment time budget in seconds (3 minutes)

# Marine frequency bands (ecological significance)
BAND_LOW = (50, 2000)       # ships, fish vocalizations, whale calls
BAND_MID = (2000, 20000)    # snapping shrimp, dolphin whistles
BAND_HIGH = (20000, 24000)  # echolocation clicks

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

UNITS = {
    "5783": {"sample_rate": 144000, "subdir": "5783"},
    "6478": {"sample_rate": 96000,  "subdir": "6478"},
    "pilot": {"sample_rate": 48000, "subdir": "Music_Soundtrap_Pilot"},
}

# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------

def find_wav_files(data_dir=None):
    """Find all WAV files across all units. Returns list of dicts."""
    if data_dir is None:
        data_dir = RAW_DIR
    recordings = []
    for unit_name, unit_info in UNITS.items():
        unit_dir = os.path.join(data_dir, unit_info["subdir"])
        if not os.path.isdir(unit_dir):
            continue
        for fname in sorted(os.listdir(unit_dir)):
            if not fname.lower().endswith(".wav"):
                continue
            path = os.path.join(unit_dir, fname)
            info = sf.info(path)
            recordings.append({
                "path": path, "filename": fname, "unit": unit_name,
                "sample_rate": info.samplerate, "duration_s": info.duration,
                "channels": info.channels, "frames": info.frames,
            })
    return recordings


def load_audio(path, sr=TARGET_SR, duration_s=None, offset_s=0.0):
    """Load WAV, resample to target SR, return (audio_float32, sr)."""
    orig_sr = sf.info(path).samplerate
    start = int(offset_s * orig_sr)
    stop = int((offset_s + duration_s) * orig_sr) if duration_s else None
    audio, _ = sf.read(path, dtype="float32", start=start, stop=stop)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio, sr


def segment_audio(audio, sr=TARGET_SR, segment_seconds=SEGMENT_SECONDS, overlap=0.0):
    """Split audio into fixed-length segments. Last is zero-padded."""
    seg_len = int(sr * segment_seconds)
    hop = int(seg_len * (1 - overlap))
    segments = []
    for start in range(0, len(audio), hop):
        seg = audio[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg)
    return segments


def highpass_filter(audio, sr=TARGET_SR, cutoff_hz=50, order=4):
    """Remove DC offset and low-frequency self-noise."""
    from scipy.signal import butter, sosfilt
    sos = butter(order, cutoff_hz, btype="high", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature utilities (available to experiment.py)
# ---------------------------------------------------------------------------

def compute_melspec(audio, sr=TARGET_SR):
    """Mel spectrogram in dB. Returns 2D array (n_mels, time_frames)."""
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=F_MIN, fmax=F_MAX,
    )
    return librosa.power_to_db(S, ref=np.max)


def compute_band_power(audio, sr=TARGET_SR):
    """Mean power (dB) in each ecological frequency band."""
    from scipy.signal import welch
    freqs, psd = welch(audio, fs=sr, nperseg=N_FFT)
    powers = {}
    for name, (flo, fhi) in [("low", BAND_LOW), ("mid", BAND_MID), ("high", BAND_HIGH)]:
        mask = (freqs >= flo) & (freqs < fhi)
        powers[name] = float(10 * np.log10(np.mean(psd[mask]) + 1e-12)) if mask.any() else -120.0
    return powers


def compute_rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


# ---------------------------------------------------------------------------
# Evaluation metrics (DO NOT CHANGE)
# ---------------------------------------------------------------------------

def evaluate_clustering(labels, features, method_name=""):
    """
    Evaluate clustering quality. Higher composite_score is better.

    composite_score = 0.5 * silhouette + 0.3 * ch_norm + 0.2 * coverage
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score

    labels = np.asarray(labels)
    features = np.asarray(features)
    n_total = len(labels)
    mask = labels >= 0
    n_clustered = mask.sum()
    n_noise = n_total - n_clustered
    unique_labels = set(labels[mask])
    n_clusters = len(unique_labels)
    coverage = n_clustered / n_total if n_total > 0 else 0.0

    result = {
        "method": method_name, "n_clusters": n_clusters,
        "n_noise": int(n_noise), "n_total": n_total, "coverage": coverage,
    }

    if n_clusters < 2 or n_clustered < 2:
        result.update(silhouette=-1.0, calinski_harabasz=0.0, composite_score=-1.0)
        return result

    sil = silhouette_score(features[mask], labels[mask])
    ch = calinski_harabasz_score(features[mask], labels[mask])
    ch_norm = 1.0 - 1.0 / (1.0 + ch / 100.0)
    composite = 0.5 * sil + 0.3 * ch_norm + 0.2 * coverage

    result.update(silhouette=float(sil), calinski_harabasz=float(ch),
                  composite_score=float(composite))
    return result


def evaluate_discovery(labels, metadata, features, segments=None):
    """
    Evaluate ecological discovery quality beyond raw clustering metrics.
    Returns dict with discovery insights. This is for logging — NOT the
    primary keep/discard metric, but valuable for the hackathon presentation.

    Measures:
    - temporal_spread: do clusters span multiple time periods?
    - unit_diversity: do clusters contain data from multiple hydrophones?
    - band_separation: do clusters separate frequency bands?
    - n_interesting: segments with high acoustic activity
    """
    labels = np.asarray(labels)
    n_clusters = len(set(labels[labels >= 0]))

    # Temporal spread: how many unique files does each cluster span?
    temporal_spreads = []
    unit_diversities = []
    for c in set(labels[labels >= 0]):
        cmask = labels == c
        cluster_meta = [metadata[i] for i in range(len(metadata)) if cmask[i]]
        unique_files = len(set(m["file"] for m in cluster_meta))
        unique_units = len(set(m["unit"] for m in cluster_meta))
        temporal_spreads.append(unique_files)
        unit_diversities.append(unique_units)

    # Band separation: compute per-cluster mean band powers if we have segments
    band_info = {}
    if segments is not None:
        for c in sorted(set(labels[labels >= 0])):
            cmask = labels == c
            cluster_segs = [segments[i] for i in range(len(segments)) if cmask[i]]
            sample = cluster_segs[:10]  # sample for speed
            powers = [compute_band_power(s) for s in sample]
            band_info[f"cluster_{c}"] = {
                "low_db": float(np.mean([p["low"] for p in powers])),
                "mid_db": float(np.mean([p["mid"] for p in powers])),
                "high_db": float(np.mean([p["high"] for p in powers])),
                "n_segments": int(cmask.sum()),
            }

    return {
        "n_clusters": n_clusters,
        "mean_temporal_spread": float(np.mean(temporal_spreads)) if temporal_spreads else 0,
        "mean_unit_diversity": float(np.mean(unit_diversities)) if unit_diversities else 0,
        "band_profiles": band_info,
    }


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(data_dir=None, max_files=None, segment_seconds=SEGMENT_SECONDS):
    """
    Load all recordings, segment them, return (segments, metadata).
    segments: list of np.array (each SEGMENT_SAMPLES long)
    metadata: list of dicts {file, unit, segment_idx, offset_s, path}
    """
    recordings = find_wav_files(data_dir)
    if max_files:
        recordings = recordings[:max_files]

    segments = []
    metadata = []
    print(f"Loading {len(recordings)} recordings...")
    for rec in recordings:
        print(f"  {rec['unit']}/{rec['filename']} ({rec['duration_s']:.0f}s @ {rec['sample_rate']}Hz)")
        audio, sr = load_audio(rec["path"])
        audio = highpass_filter(audio, sr)
        file_segments = segment_audio(audio, sr, segment_seconds)
        for i, seg in enumerate(file_segments):
            segments.append(seg)
            metadata.append({
                "file": rec["filename"], "unit": rec["unit"],
                "segment_idx": i, "offset_s": i * segment_seconds,
                "path": rec["path"],
            })
    print(f"Total segments: {len(segments)} ({segment_seconds}s each)")
    return segments, metadata


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def get_cache_path(key, suffix=".npy"):
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{key}{suffix}")


# ---------------------------------------------------------------------------
# Main (data prep / verification)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Marine Acoustic Autoresearch — Data Preparation")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    recordings = find_wav_files()
    if not recordings:
        print(f"\nNo WAV files found in {RAW_DIR}")
        print(f"Expected: {RAW_DIR}/5783/*.wav, {RAW_DIR}/6478/*.wav, {RAW_DIR}/Music_Soundtrap_Pilot/*.wav")
        sys.exit(1)

    print(f"\nFound {len(recordings)} recordings:")
    total_duration = 0
    for rec in recordings:
        print(f"  [{rec['unit']:>5}] {rec['filename']:30s} {rec['duration_s']:7.1f}s @ {rec['sample_rate']:6d}Hz")
        total_duration += rec["duration_s"]
    print(f"\nTotal audio: {total_duration/3600:.1f} hours")

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nBuilding segments (sr={TARGET_SR}, seg={SEGMENT_SECONDS}s)...")
    segments, metadata = build_dataset()

    with open(os.path.join(CACHE_DIR, "segment_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("Ready to run experiments: python3 experiment.py")
    print("=" * 60)
