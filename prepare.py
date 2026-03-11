"""
Fixed data preparation and evaluation for BRUV fish counting autoresearch.
DO NOT MODIFY — this is the ground truth data loader and evaluation harness.

Data: BRUV video from MigraMar reef monitoring in Galapagos.
  2 video series, 18 sub-videos, ~65 GB total.
  Labels: CumulativeMaxN.csv from Kaggle competition.
Task: Predict MaxN (max fish count in a single frame) for Caranx caballus.
Metric: composite_score from evaluate_maxn_predictions() — higher is better.
"""

import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def detect_device():
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

TARGET_SPECIES_GENUS = "Caranx"
TARGET_SPECIES_NAME = "caballus"
TARGET_COMMON_NAME = "Green jack"

SUB_VIDEO_DURATION_MIN = 11.783
SUB_VIDEO_DURATION_SEC = SUB_VIDEO_DURATION_MIN * 60
NATIVE_FPS = 30

SERIES_INFO = {
    1: {"suffix": "0001", "sub_videos": 9},
    2: {"suffix": "0002", "sub_videos": 9},
}

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
VIDEO_DIR = os.path.join(DATA_DIR, "videos")
LABEL_DIR = os.path.join(DATA_DIR, "labels")
FRAME_DIR = os.path.join(DATA_DIR, "frames")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
RESULTS_DIR = os.path.join(DATA_DIR, "results")

TIME_BUDGET = 300


def load_labels(label_dir=None):
    if label_dir is None:
        label_dir = LABEL_DIR

    csv_path = os.path.join(label_dir, "CumulativeMaxN.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        print("Download from Kaggle: kaggle competitions download -c marine-conservation-with-migra-mar")
        return None

    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df)} rows from CumulativeMaxN.csv")
    print(f"  Columns: {list(df.columns)}")

    col_map = {}
    for c in df.columns:
        cl = c.lower().strip().replace(" ", "_")
        if "filename" in cl:
            col_map[c] = "filename"
        elif cl == "frame":
            col_map[c] = "frame"
        elif "time" in cl and "min" in cl:
            col_map[c] = "time_mins"
        elif cl == "family":
            col_map[c] = "family"
        elif cl == "genus":
            col_map[c] = "genus"
        elif cl == "species":
            col_map[c] = "species"
        elif "maxn" in cl.replace(" ", ""):
            col_map[c] = "cumulative_maxn"
    df = df.rename(columns=col_map)

    return df


def get_target_species_data(df):
    if df is None:
        return None

    mask = pd.Series(False, index=df.index)

    if "species" in df.columns and "genus" in df.columns:
        genus_match = df["genus"].astype(str).str.strip().str.lower() == TARGET_SPECIES_GENUS.lower()
        species_match = df["species"].astype(str).str.strip().str.lower() == TARGET_SPECIES_NAME.lower()
        mask = genus_match & species_match

    if mask.sum() == 0 and "species" in df.columns:
        mask = df["species"].astype(str).str.contains(TARGET_SPECIES_NAME, case=False, na=False)

    filtered = df[mask].copy()
    print(f"  Target species ({TARGET_SPECIES_GENUS} {TARGET_SPECIES_NAME}) rows: {len(filtered)}")
    return filtered


def parse_series_id(filename):
    name = Path(filename).stem
    if len(name) >= 9:
        suffix = name[5:9]
        if suffix == "0001":
            return 1
        elif suffix == "0002":
            return 2
    return 0


def parse_subvideo_index(filename):
    name = Path(filename).stem
    if len(name) >= 5:
        try:
            return int(name[3:5])
        except ValueError:
            pass
    return 0


def time_to_subvideo(time_mins, series_id):
    sub_idx = int(time_mins / SUB_VIDEO_DURATION_MIN) + 1
    local_mins = time_mins - (sub_idx - 1) * SUB_VIDEO_DURATION_MIN
    local_secs = local_mins * 60

    suffix = SERIES_INFO.get(series_id, {}).get("suffix", "0001")
    filename = f"LGH{sub_idx:02d}{suffix}.MP4"

    return filename, local_secs


def get_maxn_per_subvideo(df=None, label_dir=None):
    if df is None:
        df = load_labels(label_dir)

    target_df = get_target_species_data(df)
    if target_df is None or len(target_df) == 0:
        return {}

    maxn_dict = {}
    if "filename" in target_df.columns and "cumulative_maxn" in target_df.columns:
        for fname, group in target_df.groupby("filename"):
            maxn_dict[fname] = int(group["cumulative_maxn"].max())

    return maxn_dict


def get_series_maxn(maxn_per_subvideo):
    series_maxn = {}
    for fname, maxn in maxn_per_subvideo.items():
        series_id = parse_series_id(fname)
        if series_id not in series_maxn or maxn > series_maxn[series_id]:
            series_maxn[series_id] = maxn
    return series_maxn


def get_key_frames(df=None, label_dir=None):
    if df is None:
        df = load_labels(label_dir)

    target_df = get_target_species_data(df)
    if target_df is None or len(target_df) == 0:
        return []

    if "filename" not in target_df.columns or "cumulative_maxn" not in target_df.columns:
        return []

    key_frames = []
    for fname, group in target_df.groupby("filename"):
        sort_col = "frame" if "frame" in group.columns else group.columns[0]
        group = group.sort_values(sort_col)
        prev_maxn = 0
        for _, row in group.iterrows():
            curr_maxn = int(row["cumulative_maxn"])
            if curr_maxn > prev_maxn:
                key_frames.append({
                    "filename": fname,
                    "frame": int(row.get("frame", 0)),
                    "time_mins": float(row.get("time_mins", 0)),
                    "count": curr_maxn,
                })
                prev_maxn = curr_maxn

    return key_frames


def find_available_videos(video_dir=None):
    if video_dir is None:
        video_dir = VIDEO_DIR

    if not os.path.isdir(video_dir):
        return []

    videos = sorted(Path(video_dir).glob("*.MP4")) + sorted(Path(video_dir).glob("*.mp4"))
    return [str(v) for v in videos]


def extract_frame(video_path, time_sec=None, frame_number=None):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    if time_sec is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
    elif frame_number is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    return frame if ret else None


def extract_frames_at_fps(video_path, sample_fps=1, max_frames=None):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / native_fps if native_fps > 0 else 0

    frame_interval = max(1, int(native_fps / sample_fps))

    print(f"  Video: {Path(video_path).name}, {native_fps:.1f}fps, "
          f"{total_frames} frames, {duration_sec:.1f}s")
    print(f"  Sampling every {frame_interval} frames ({sample_fps} fps)")

    count = 0
    frame_idx = 0
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        time_sec = frame_idx / native_fps
        yield time_sec, frame
        count += 1

        if max_frames and count >= max_frames:
            break

        frame_idx += frame_interval

    cap.release()
    print(f"  Extracted {count} frames")


def get_video_info(video_path):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration_sec"] = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0
    cap.release()
    return info


def evaluate_maxn_predictions(pred_maxn, true_maxn):
    common_keys = set(pred_maxn.keys()) & set(true_maxn.keys())
    if not common_keys:
        return {
            "composite_score": 0.0,
            "n_videos": 0,
            "error": "no common video keys between predictions and ground truth",
        }

    true_vals = np.array([true_maxn[k] for k in sorted(common_keys)])
    pred_vals = np.array([pred_maxn[k] for k in sorted(common_keys)])

    abs_errors = np.abs(pred_vals - true_vals)
    mae = float(np.mean(abs_errors))

    rel_errors = abs_errors / np.maximum(true_vals, 1).astype(float)
    mre = float(np.mean(rel_errors))

    log_true = np.log1p(true_vals.astype(float))
    log_pred = np.log1p(pred_vals.astype(float))
    log_mae = float(np.mean(np.abs(log_pred - log_true)))

    if len(true_vals) >= 3:
        corr = float(np.corrcoef(true_vals, pred_vals)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    log_mae_score = 1.0 / (1.0 + log_mae)
    mre_score = 1.0 / (1.0 + mre)
    corr_score = max(0.0, corr)

    composite = 0.4 * log_mae_score + 0.4 * mre_score + 0.2 * corr_score

    per_video = {}
    for k in sorted(common_keys):
        per_video[k] = {
            "true_maxn": int(true_maxn[k]),
            "pred_maxn": int(pred_maxn[k]),
            "abs_error": int(abs(pred_maxn[k] - true_maxn[k])),
        }

    return {
        "composite_score": float(composite),
        "mae": mae,
        "mre": mre,
        "log_mae": log_mae,
        "log_mae_score": float(log_mae_score),
        "mre_score": float(mre_score),
        "correlation": corr,
        "n_videos": len(common_keys),
        "per_video": per_video,
    }


def evaluate_frame_counts(pred_counts, true_counts):
    pred = np.asarray(pred_counts, dtype=float)
    true = np.asarray(true_counts, dtype=float)

    mae = float(np.mean(np.abs(pred - true)))
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))

    if len(pred) >= 3:
        corr = float(np.corrcoef(pred, true)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    return {
        "frame_mae": mae,
        "frame_rmse": rmse,
        "frame_correlation": corr,
        "n_frames": len(pred),
    }


def print_evaluation(eval_result):
    print("\n---")
    print(f"composite_score:  {eval_result.get('composite_score', 0.0):.6f}")
    print(f"mae:              {eval_result.get('mae', 0.0):.2f}")
    print(f"mre:              {eval_result.get('mre', 0.0):.4f}")
    print(f"log_mae:          {eval_result.get('log_mae', 0.0):.4f}")
    print(f"correlation:      {eval_result.get('correlation', 0.0):.4f}")
    print(f"n_videos:         {eval_result.get('n_videos', 0)}")

    if "per_video" in eval_result:
        print("\nPer-video results:")
        for vid, info in eval_result["per_video"].items():
            print(f"  {vid}: true={info['true_maxn']}, pred={info['pred_maxn']}, "
                  f"err={info['abs_error']}")


def build_dataset(label_dir=None, video_dir=None):
    if label_dir is None:
        label_dir = LABEL_DIR
    if video_dir is None:
        video_dir = VIDEO_DIR

    print("Loading labels...")
    labels_df = load_labels(label_dir)
    if labels_df is None:
        return None

    target_df = get_target_species_data(labels_df)
    maxn_per_sub = get_maxn_per_subvideo(labels_df)
    series_maxn = get_series_maxn(maxn_per_sub)
    key_frames = get_key_frames(labels_df)

    print(f"\nMaxN per sub-video:")
    for fname in sorted(maxn_per_sub.keys()):
        print(f"  {fname}: {maxn_per_sub[fname]}")

    print(f"\nMaxN per series:")
    for sid, maxn in sorted(series_maxn.items()):
        print(f"  Series {sid}: {maxn}")

    print(f"\nKey frames (where MaxN increases): {len(key_frames)}")

    available_videos = find_available_videos(video_dir)
    print(f"\nAvailable videos: {len(available_videos)}")
    for v in available_videos:
        print(f"  {Path(v).name}")

    return {
        "labels_df": labels_df,
        "target_df": target_df,
        "maxn_per_subvideo": maxn_per_sub,
        "series_maxn": series_maxn,
        "key_frames": key_frames,
        "available_videos": available_videos,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("BRUV Fish Counting — Data Preparation")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    for d in [DATA_DIR, VIDEO_DIR, LABEL_DIR, FRAME_DIR, CACHE_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

    result = build_dataset()
    if result is None:
        print(f"\nNo label files found in {LABEL_DIR}")
        print("Place CumulativeMaxN.csv in data/labels/")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Ready to run experiments: python3 experiment.py")
    print("=" * 60)
