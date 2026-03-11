"""
Fixed data preparation and evaluation for BRUV fish counting autoresearch.
DO NOT MODIFY — this is the ground truth data loader and evaluation harness.

Adapted from Karpathy's autoresearch pattern for BRUV fish counting.
Runs on CUDA, MPS, or CPU — auto-detected.

Data: Baited Remote Underwater Video (BRUV) from MigraMar deployments.
  - 18 sub-videos across 2 stations (~11:47 each at 30fps)
  - Target: count Caranx caballus (green jack) MaxN per video
"""

import os
import sys
import json
import time
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import cv2

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_device():
    """Auto-detect best available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:
        return "cpu"

DEVICE = detect_device()

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TARGET_FPS = 2              # process at 2 fps (from 30fps source)
SOURCE_FPS = 30             # original video fps
FRAME_SKIP = SOURCE_FPS // TARGET_FPS  # every 15th frame
SUB_VIDEO_DURATION_S = 11 * 60 + 47    # ~11:47 per sub-video
SUB_VIDEO_DURATION_MIN = 11.783         # minutes per sub-video segment

# Image dimensions for processing
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080

TIME_BUDGET = 600           # experiment time budget in seconds (10 minutes)

# Target species
TARGET_SPECIES = "Caranx caballus"
GROUND_TRUTH_MAXN = {
    "video_1": 251,  # max fish in single frame for video 1
    "video_2": 52,   # max fish in single frame for video 2
}

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
FRAME_DIR = os.path.join(DATA_DIR, "frames")
LABEL_DIR = os.path.join(DATA_DIR, "labels")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------

def find_video_files(video_dir=None):
    """Find all video files. Returns list of dicts with metadata."""
    if video_dir is None:
        video_dir = VIDEO_DIR
    videos = []
    for ext in ["*.mp4", "*.avi", "*.mov", "*.MP4", "*.AVI", "*.MOV"]:
        for path in sorted(glob.glob(os.path.join(video_dir, "**", ext), recursive=True)):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_s = n_frames / fps if fps > 0 else 0
            cap.release()
            videos.append({
                "path": path, "filename": os.path.basename(path),
                "fps": fps, "n_frames": n_frames,
                "width": width, "height": height,
                "duration_s": duration_s,
            })
    return videos


def find_frame_dirs(frame_dir=None):
    """Find directories of pre-extracted frames. Returns list of dicts."""
    if frame_dir is None:
        frame_dir = FRAME_DIR
    frame_sets = []
    if not os.path.isdir(frame_dir):
        return frame_sets
    for subdir in sorted(os.listdir(frame_dir)):
        subdir_path = os.path.join(frame_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        frames = sorted(glob.glob(os.path.join(subdir_path, "*.jpg")) +
                       glob.glob(os.path.join(subdir_path, "*.png")))
        if frames:
            frame_sets.append({
                "dir": subdir_path, "name": subdir,
                "n_frames": len(frames), "frames": frames,
            })
    return frame_sets


def load_labels(label_dir=None):
    """Load ground truth CSVs. Returns dict of DataFrames."""
    if label_dir is None:
        label_dir = LABEL_DIR
    labels = {}
    cumulative_path = os.path.join(label_dir, "CumulativeMaxN.csv")
    if os.path.exists(cumulative_path):
        labels["cumulative_maxn"] = pd.read_csv(cumulative_path)
    timefirst_path = os.path.join(label_dir, "TimeFirstSeen.csv")
    if os.path.exists(timefirst_path):
        labels["time_first_seen"] = pd.read_csv(timefirst_path)
    return labels


def get_ground_truth_maxn(labels, species=TARGET_SPECIES):
    """Extract ground truth MaxN for target species from labels."""
    if "cumulative_maxn" not in labels:
        return GROUND_TRUTH_MAXN
    df = labels["cumulative_maxn"]
    # Try to find species column — adapt to actual CSV structure
    species_cols = [c for c in df.columns if "species" in c.lower() or "taxon" in c.lower()]
    count_cols = [c for c in df.columns if "maxn" in c.lower() or "count" in c.lower() or "n" == c.lower()]
    if species_cols and count_cols:
        mask = df[species_cols[0]].str.contains(species, case=False, na=False)
        if mask.any():
            return {"labeled": int(df.loc[mask, count_cols[0]].max())}
    return GROUND_TRUTH_MAXN


def extract_frames_from_video(video_path, output_dir=None, fps=TARGET_FPS, max_frames=None):
    """Extract frames from video at target FPS. Returns list of frame paths."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    skip = max(1, int(source_fps / fps))

    if output_dir is None:
        name = Path(video_path).stem
        output_dir = os.path.join(FRAME_DIR, name)
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            path = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(path, frame)
            frame_paths.append(path)
            saved += 1
            if max_frames and saved >= max_frames:
                break
        frame_idx += 1

    cap.release()
    return frame_paths


def load_frame(path, resize=None):
    """Load a frame as BGR numpy array. Optionally resize."""
    frame = cv2.imread(path)
    if frame is None:
        raise ValueError(f"Cannot read frame: {path}")
    if resize:
        frame = cv2.resize(frame, resize)
    return frame


def global_time_to_local(time_mins):
    """Convert global timeline minutes to (sub_video_index, local_time_mins)."""
    sub_idx = int(time_mins / SUB_VIDEO_DURATION_MIN)
    local_time = time_mins - sub_idx * SUB_VIDEO_DURATION_MIN
    return sub_idx, local_time


# ---------------------------------------------------------------------------
# Image processing utilities (available to experiment.py)
# ---------------------------------------------------------------------------

def compute_foreground_mask(frames, history=500, var_threshold=16):
    """Apply MOG2 background subtraction. Returns list of foreground masks."""
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=history, varThreshold=var_threshold, detectShadows=False
    )
    masks = []
    for frame in frames:
        mask = bg_sub.apply(frame)
        masks.append(mask)
    return masks


def count_contours(mask, min_area=100, max_area=50000):
    """Count contours in a binary mask within area bounds."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    return len(valid), valid


def apply_morphology(mask, kernel_size=5, operations=("close", "open")):
    """Apply morphological operations to clean binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    result = mask.copy()
    for op in operations:
        if op == "close":
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        elif op == "open":
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        elif op == "dilate":
            result = cv2.dilate(result, kernel)
        elif op == "erode":
            result = cv2.erode(result, kernel)
    return result


def generate_color_views(frame):
    """
    Generate multiple color-space views of a frame ('scuba glasses' approach).
    Each view highlights different aspects of fish appearance against water.
    Returns dict of {view_name: image_array}.
    """
    views = {"bgr": frame}

    # HSV channels — hue isolates color independent of lighting
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    views["hsv_h"] = hsv[:, :, 0]
    views["hsv_s"] = hsv[:, :, 1]
    views["hsv_v"] = hsv[:, :, 2]

    # LAB channels — L is lightness, A is green-red, B is blue-yellow
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    views["lab_l"] = lab[:, :, 0]
    views["lab_a"] = lab[:, :, 1]
    views["lab_b"] = lab[:, :, 2]

    # CLAHE enhanced (adaptive histogram equalization)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    views["clahe"] = clahe.apply(gray)

    # CLAHE on LAB L-channel (better contrast in murky water)
    lab_clahe = lab.copy()
    lab_clahe[:, :, 0] = clahe.apply(lab[:, :, 0])
    views["lab_clahe"] = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    # Grayscale
    views["gray"] = gray

    # Blue channel ratio (fish often contrast against blue water)
    b, g, r = cv2.split(frame.astype(np.float32) + 1e-6)
    views["blue_ratio"] = np.uint8(np.clip(b / (r + g + b) * 255, 0, 255))
    views["green_minus_blue"] = np.uint8(np.clip((g - b) + 128, 0, 255))

    return views


# ---------------------------------------------------------------------------
# Evaluation metrics (DO NOT CHANGE)
# ---------------------------------------------------------------------------

def evaluate_counting(predicted_maxn, ground_truth_maxn=None, per_frame_counts=None,
                      method_name=""):
    """
    Evaluate MaxN counting quality. Higher maxn_score is better.

    maxn_score = 1.0 - (|predicted - truth| / truth)  clamped to [0, 1]

    Also computes temporal consistency if per_frame_counts provided.
    """
    if ground_truth_maxn is None:
        # Use first available ground truth
        ground_truth_maxn = list(GROUND_TRUTH_MAXN.values())[0]

    if ground_truth_maxn == 0:
        maxn_score = 1.0 if predicted_maxn == 0 else 0.0
    else:
        error_frac = abs(predicted_maxn - ground_truth_maxn) / ground_truth_maxn
        maxn_score = max(0.0, 1.0 - error_frac)

    mae = abs(predicted_maxn - ground_truth_maxn)

    result = {
        "method": method_name,
        "predicted_maxn": int(predicted_maxn),
        "ground_truth_maxn": int(ground_truth_maxn),
        "maxn_score": float(maxn_score),
        "mae": float(mae),
    }

    # Temporal consistency: how smooth are frame-level counts?
    if per_frame_counts is not None and len(per_frame_counts) > 1:
        counts = np.array(per_frame_counts, dtype=float)
        result["count_std"] = float(np.std(counts))
        result["count_max"] = int(np.max(counts))
        result["count_mean"] = float(np.mean(counts))
        result["n_frames"] = len(counts)
        # Temporal smoothness: lower jitter is better
        diffs = np.abs(np.diff(counts))
        result["temporal_jitter"] = float(np.mean(diffs))

    return result


def evaluate_discovery(per_frame_counts, frame_metadata=None):
    """
    Evaluate ecological discovery beyond raw counting.
    Returns dict with discovery insights for hackathon presentation.

    Measures:
    - temporal_pattern: arrival/departure dynamics
    - spatial_distribution: where in frame fish concentrate
    - peak_analysis: characteristics of high-count frames
    """
    counts = np.array(per_frame_counts, dtype=float) if per_frame_counts else np.array([0])

    discovery = {
        "n_frames_analyzed": len(counts),
        "max_count": int(np.max(counts)) if len(counts) > 0 else 0,
        "mean_count": float(np.mean(counts)) if len(counts) > 0 else 0,
    }

    if len(counts) > 10:
        # Find peak window
        window = min(30, len(counts) // 4)
        if window > 0:
            rolling_max = np.array([
                np.max(counts[max(0, i - window):i + window])
                for i in range(len(counts))
            ])
            peak_idx = int(np.argmax(rolling_max))
            discovery["peak_frame_idx"] = peak_idx
            discovery["peak_window_max"] = int(rolling_max[peak_idx])

        # Arrival dynamics: when does count first exceed 50% of max?
        max_count = np.max(counts)
        if max_count > 0:
            threshold = max_count * 0.5
            above = np.where(counts >= threshold)[0]
            if len(above) > 0:
                discovery["arrival_frame"] = int(above[0])
                discovery["departure_frame"] = int(above[-1])
                discovery["high_count_duration_frames"] = int(above[-1] - above[0])

    return discovery


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_frame_dataset(max_frames_per_video=None, resize=None):
    """
    Load all available frames from pre-extracted frame directories or videos.
    Returns (frames, metadata) where frames is list of BGR arrays.
    """
    frame_sets = find_frame_dirs()
    all_frames = []
    all_metadata = []

    if frame_sets:
        print(f"Found {len(frame_sets)} pre-extracted frame sets")
        for fs in frame_sets:
            frame_paths = fs["frames"]
            if max_frames_per_video:
                frame_paths = frame_paths[:max_frames_per_video]
            for i, fp in enumerate(frame_paths):
                frame = load_frame(fp, resize=resize)
                all_frames.append(frame)
                all_metadata.append({
                    "path": fp, "source": fs["name"],
                    "frame_idx": i, "set_name": fs["name"],
                })
            print(f"  {fs['name']}: {len(frame_paths)} frames")
    else:
        # Fall back to extracting from videos
        videos = find_video_files()
        if not videos:
            print("No videos or pre-extracted frames found.")
            return all_frames, all_metadata
        print(f"Found {len(videos)} videos, extracting frames...")
        for vid in videos:
            print(f"  Extracting from {vid['filename']}...")
            frame_paths = extract_frames_from_video(
                vid["path"], fps=TARGET_FPS,
                max_frames=max_frames_per_video
            )
            for i, fp in enumerate(frame_paths):
                frame = load_frame(fp, resize=resize)
                all_frames.append(frame)
                all_metadata.append({
                    "path": fp, "source": vid["filename"],
                    "frame_idx": i, "set_name": Path(vid["filename"]).stem,
                })

    print(f"Total frames loaded: {len(all_frames)}")
    return all_frames, all_metadata


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
    print("BRUV Fish Counting Autoresearch — Data Preparation")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Check for videos
    videos = find_video_files()
    frame_sets = find_frame_dirs()
    labels = load_labels()

    print(f"\nVideos found: {len(videos)}")
    for v in videos:
        print(f"  {v['filename']:40s} {v['duration_s']:7.1f}s @ {v['fps']:.0f}fps  {v['width']}x{v['height']}")

    print(f"\nPre-extracted frame sets: {len(frame_sets)}")
    for fs in frame_sets:
        print(f"  {fs['name']:40s} {fs['n_frames']} frames")

    print(f"\nLabels loaded: {list(labels.keys())}")
    if labels:
        gt = get_ground_truth_maxn(labels)
        print(f"Ground truth MaxN: {gt}")
    else:
        print(f"Using default ground truth: {GROUND_TRUTH_MAXN}")

    if not videos and not frame_sets:
        print(f"\nNo data found!")
        print(f"Place videos in: {VIDEO_DIR}/")
        print(f"Or pre-extracted frames in: {FRAME_DIR}/<video_name>/*.jpg")
        print(f"And labels in: {LABEL_DIR}/CumulativeMaxN.csv")
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 60)
    print("Ready to run experiments: python3 experiment.py")
    print("=" * 60)
