"""
Marine Sound Catalog — Identify sounds in each file with timestamps + PANN embeddings.
Outputs:
  - marine_catalog.json: per-segment sound identification with timestamps
  - marine_embeddings.npy: PANNs 2048-dim embeddings for all segments
  - marine_catalog_summary.txt: human-readable summary
"""

import os
import json
import re
import time
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt, welch

# ---------------------------------------------------------------------------
# Constants (from marine prepare.py)
# ---------------------------------------------------------------------------
TARGET_SR = 48000
SEGMENT_SECONDS = 10
SEGMENT_SAMPLES = TARGET_SR * SEGMENT_SECONDS
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 50
F_MAX = 24000
BAND_LOW = (50, 2000)
BAND_MID = (2000, 20000)
BAND_HIGH = (20000, 24000)
PANNS_SR = 32000

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

UNITS = {
    "5783": {"sample_rate": 144000, "subdir": "5783"},
    "6478": {"sample_rate": 96000, "subdir": "6478"},
    "pilot": {"sample_rate": 48000, "subdir": "Music_Soundtrap_Pilot"},
}


def find_wav_files(data_dir=None):
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
            })
    return recordings


def load_audio(path, sr=TARGET_SR, duration_s=None, offset_s=0.0):
    orig_sr = sf.info(path).samplerate
    start = int(offset_s * orig_sr)
    stop = int((offset_s + duration_s) * orig_sr) if duration_s else None
    audio, _ = sf.read(path, dtype="float32", start=start, stop=stop)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio, sr


def segment_audio(audio, sr=TARGET_SR, segment_seconds=SEGMENT_SECONDS):
    seg_len = int(sr * segment_seconds)
    segments = []
    for start in range(0, len(audio), seg_len):
        seg = audio[start:start + seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg)
    return segments


def highpass_filter(audio, sr=TARGET_SR, cutoff_hz=50, order=4):
    sos = butter(order, cutoff_hz, btype="high", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def compute_band_power(audio, sr=TARGET_SR):
    freqs, psd = welch(audio, fs=sr, nperseg=N_FFT)
    powers = {}
    for name, (flo, fhi) in [("low", BAND_LOW), ("mid", BAND_MID), ("high", BAND_HIGH)]:
        mask = (freqs >= flo) & (freqs < fhi)
        powers[name] = float(10 * np.log10(np.mean(psd[mask]) + 1e-12)) if mask.any() else -120.0
    return powers


def compute_rms(audio):
    return float(np.sqrt(np.mean(audio ** 2)))


def build_dataset():
    recordings = find_wav_files()
    segments = []
    metadata = []
    print(f"Loading {len(recordings)} recordings...")
    for rec in recordings:
        print(f"  {rec['unit']}/{rec['filename']} ({rec['duration_s']:.0f}s @ {rec['sample_rate']}Hz)")
        audio, sr = load_audio(rec["path"])
        audio = highpass_filter(audio, sr)
        file_segments = segment_audio(audio, sr)
        for i, seg in enumerate(file_segments):
            segments.append(seg)
            metadata.append({
                "file": rec["filename"], "unit": rec["unit"],
                "segment_idx": i, "offset_s": i * SEGMENT_SECONDS,
                "path": rec["path"],
            })
    print(f"Total segments: {len(segments)} ({SEGMENT_SECONDS}s each)")
    return segments, metadata

# Marine-relevant AudioSet classes grouped by ecological category
MARINE_CLASSES = {
    "Water": "ambient", "Ocean": "ambient", "Rain": "ambient",
    "Stream": "ambient", "Waves, surf": "ambient",
    "Boat, Water vehicle": "anthropogenic", "Ship": "anthropogenic",
    "Engine": "anthropogenic", "Motor vehicle (road)": "anthropogenic",
    "Vehicle": "anthropogenic", "Motorboat, speedboat": "anthropogenic",
    "Mechanical fan": "anthropogenic",
    "Whale vocalization": "biological", "Animal": "biological",
    "Bird": "biological", "Insect": "biological",
    "Click": "biological", "Squeak": "biological",
    "Chirp, tweet": "biological", "Splash, splashing": "ambient",
    "Rumble": "ambiguous", "Hum": "ambiguous",
    "White noise": "ambient", "Noise": "ambient",
    "Silence": "ambient", "Static": "ambient",
}


def parse_timestamp(filename):
    """Extract timestamp from SoundTrap filename like '6478.230723191251.wav'."""
    match = re.search(r'\.(\d{12})\.', filename)
    if match:
        ts = match.group(1)
        year = 2000 + int(ts[0:2])
        month = int(ts[2:4])
        day = int(ts[4:6])
        hour = int(ts[6:8])
        minute = int(ts[8:10])
        second = int(ts[10:12])
        return f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
    # Pilot files: 190806_3761.wav — date but no time
    match = re.search(r'(\d{6})_\d+\.wav', filename)
    if match:
        ds = match.group(1)
        year = 2000 + int(ds[0:2])
        month = int(ds[2:4])
        day = int(ds[4:6])
        return f"{year}-{month:02d}-{day:02d}"
    return "unknown"


def classify_by_band(seg):
    """Classify segment using band power analysis."""
    bp = compute_band_power(seg)
    rms = compute_rms(seg)
    bio_lin = 10 ** (bp["mid"] / 10)
    anthro_lin = 10 ** (bp["low"] / 10)
    ndsi = (bio_lin - anthro_lin) / (bio_lin + anthro_lin + 1e-12)

    mid_low_ratio = bp["mid"] - bp["low"]
    high_mid_ratio = bp["high"] - bp["mid"]

    if rms < 0.001:
        label = "near-silence"
    elif bp["low"] > -30 and mid_low_ratio < -15:
        label = "boat/ship engine"
    elif mid_low_ratio < -20 and rms > 0.01:
        label = "vessel noise"
    elif ndsi > 0.3 and rms > 0.005:
        label = "strong biological (reef/shrimp)"
    elif ndsi > -0.2 and bp["mid"] > -85:
        label = "biological (shrimp/reef/fish)"
    elif bp["high"] > -80 and high_mid_ratio > 5:
        label = "echolocation clicks"
    elif bp["mid"] > -100 and bp["mid"] < -80 and rms > 0.005:
        label = "moderate biological"
    elif rms > 0.01 and bp["low"] > -80 and bp["mid"] < -100:
        label = "low-freq acoustic event"
    elif bp["low"] < -100 and bp["mid"] < -110:
        label = "quiet ambient"
    else:
        label = "mixed soundscape"

    return {
        "acoustic_label": label,
        "ndsi": round(ndsi, 4),
        "rms": round(rms, 6),
        "band_low_db": round(bp["low"], 1),
        "band_mid_db": round(bp["mid"], 1),
        "band_high_db": round(bp["high"], 1),
    }


def main():
    t0 = time.time()

    # Load data
    print("Loading audio segments...")
    segments, metadata = build_dataset()
    n_segments = len(segments)
    print(f"Loaded {n_segments} segments from {len(set(m['file'] for m in metadata))} files\n")

    # Band-power classification for all segments
    print("Classifying segments by frequency bands...")
    band_results = []
    for i, seg in enumerate(segments):
        band_results.append(classify_by_band(seg))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{n_segments}")

    # PANNs inference on ALL segments
    print("\nLoading PANNs model...")
    import torch
    from panns_inference import AudioTagging
    from panns_inference.config import labels as audioset_labels

    at = AudioTagging(checkpoint_path=None, device='cpu')

    all_embeddings = []
    all_panns_labels = []

    print(f"Running PANNs on {n_segments} segments...")
    for i, seg in enumerate(segments):
        resampled = librosa.resample(seg, orig_sr=TARGET_SR, target_sr=PANNS_SR)
        audio_tensor = resampled[np.newaxis, :]
        with torch.no_grad():
            clip_probs, emb = at.inference(torch.from_numpy(audio_tensor))

        all_embeddings.append(emb[0])  # 2048-dim embedding

        top_indices = np.argsort(clip_probs[0])[-10:][::-1]
        top_labels = [(audioset_labels[idx], round(float(clip_probs[0][idx]), 4))
                      for idx in top_indices]
        all_panns_labels.append(top_labels)

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{n_segments} ({elapsed:.0f}s)")

    # Save embeddings
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    emb_path = os.path.join(os.path.dirname(__file__), "marine_embeddings.npy")
    np.save(emb_path, embeddings_array)
    print(f"\nSaved embeddings: {emb_path} — shape {embeddings_array.shape}")

    # Build catalog
    catalog = {}  # keyed by filename
    for i, m in enumerate(metadata):
        fname = m["file"]
        if fname not in catalog:
            catalog[fname] = {
                "file": fname,
                "unit": m["unit"],
                "recording_timestamp": parse_timestamp(fname),
                "path": m["path"],
                "segments": [],
            }

        # PANNs ecological category
        panns_top = all_panns_labels[i]
        bio_score = sum(p for l, p in panns_top if MARINE_CLASSES.get(l) == "biological")
        anthro_score = sum(p for l, p in panns_top if MARINE_CLASSES.get(l) == "anthropogenic")
        ambient_score = sum(p for l, p in panns_top if MARINE_CLASSES.get(l) == "ambient")

        if anthro_score > 0.1:
            panns_category = "anthropogenic"
        elif bio_score > 0.05:
            panns_category = "biological"
        elif ambient_score > 0.2:
            panns_category = "ambient"
        else:
            panns_category = "unknown"

        seg_info = {
            "segment_idx": m["segment_idx"],
            "start_s": m["offset_s"],
            "end_s": m["offset_s"] + SEGMENT_SECONDS,
            "panns_top5": panns_top[:5],
            "panns_category": panns_category,
            **band_results[i],
        }
        catalog[fname]["segments"].append(seg_info)

    # Save catalog JSON
    catalog_path = os.path.join(os.path.dirname(__file__), "marine_catalog.json")
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Saved catalog: {catalog_path}")

    # Print human-readable summary
    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("MARINE SOUND CATALOG — Galápagos Hydrophone Recordings")
    summary_lines.append("=" * 80)

    for fname, fdata in catalog.items():
        summary_lines.append(f"\n{'─' * 70}")
        summary_lines.append(f"FILE: {fname}")
        summary_lines.append(f"  Unit: {fdata['unit']}  |  Recorded: {fdata['recording_timestamp']}")
        summary_lines.append(f"  Segments: {len(fdata['segments'])}")
        summary_lines.append("")
        summary_lines.append(f"  {'Time':>12s}  {'Acoustic Label':25s}  {'PANNs Category':15s}  {'PANNs Top Sound':25s}  {'NDSI':>6s}")
        summary_lines.append(f"  {'─'*12}  {'─'*25}  {'─'*15}  {'─'*25}  {'─'*6}")

        for seg in fdata["segments"]:
            start = seg["start_s"]
            end = seg["end_s"]
            time_str = f"{int(start//60):02d}:{int(start%60):02d}-{int(end//60):02d}:{int(end%60):02d}"
            top_sound = seg["panns_top5"][0][0] if seg["panns_top5"] else "?"
            top_prob = seg["panns_top5"][0][1] if seg["panns_top5"] else 0
            summary_lines.append(
                f"  {time_str:>12s}  {seg['acoustic_label']:25s}  {seg['panns_category']:15s}  "
                f"{top_sound[:22]:25s}  {seg['ndsi']:>6.3f}"
            )

    summary_text = "\n".join(summary_lines)
    summary_path = os.path.join(os.path.dirname(__file__), "marine_catalog_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"Saved summary: {summary_path}")

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(summary_text)


if __name__ == "__main__":
    main()
