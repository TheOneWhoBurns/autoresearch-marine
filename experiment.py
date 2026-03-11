"""
BRUV Fish Counting Autoresearch — Experiment Script
THIS IS THE ONLY FILE THE AGENT MODIFIES.

Starting point: Tier 1 MOG2 background subtraction baseline.
The agent evolves this through Tiers 1-3, improving the composite_score.

Metric: composite_score from evaluate_maxn_predictions() — higher is better.

Usage: python3 experiment.py > run.log 2>&1
"""

import os
import time
import json
import numpy as np
from pathlib import Path

from prepare import (
    DEVICE, TIME_BUDGET, RESULTS_DIR, NATIVE_FPS,
    TARGET_SPECIES_GENUS, TARGET_SPECIES_NAME,
    build_dataset, evaluate_maxn_predictions, print_evaluation,
    extract_frames_at_fps, find_available_videos,
    parse_series_id, parse_subvideo_index,
)

TIER = 1

SAMPLE_FPS = 1
MOG2_HISTORY = 120
MOG2_VAR_THRESHOLD = 50
MOG2_DETECT_SHADOWS = True
MIN_CONTOUR_AREA = 200
MAX_CONTOUR_AREA = 50000
MORPH_KERNEL_SIZE = 5
BLUR_SIZE = 5


def count_fish_background_subtraction(video_path, sample_fps=SAMPLE_FPS):
    import cv2

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )

    frame_counts = []
    max_count = 0

    for time_sec, frame in extract_frames_at_fps(video_path, sample_fps):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

        fg_mask = bg_sub.apply(blurred)

        if MOG2_DETECT_SHADOWS:
            fg_mask = (fg_mask == 255).astype(np.uint8) * 255

        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_CONTOUR_AREA <= area <= MAX_CONTOUR_AREA:
                count += 1

        frame_counts.append((time_sec, count))
        if count > max_count:
            max_count = count

    return max_count, frame_counts


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"BRUV Fish Counting — Tier {TIER}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    config = {
        "tier": TIER,
        "method": "background_subtraction",
        "sample_fps": SAMPLE_FPS,
        "mog2_history": MOG2_HISTORY,
        "mog2_var_threshold": MOG2_VAR_THRESHOLD,
        "min_contour_area": MIN_CONTOUR_AREA,
        "max_contour_area": MAX_CONTOUR_AREA,
        "morph_kernel_size": MORPH_KERNEL_SIZE,
        "blur_size": BLUR_SIZE,
    }
    print(f"\nConfig: {json.dumps(config, indent=2)}")

    print("\n--- Loading dataset ---")
    dataset = build_dataset()
    if dataset is None:
        print("ERROR: No data found.")
        return

    true_maxn = dataset["maxn_per_subvideo"]
    available_videos = dataset["available_videos"]

    if not available_videos:
        print("ERROR: No video files found in data/videos/")
        print("Download at least one sub-video from R2.")
        return

    t_load = time.time() - t_start
    print(f"Data loaded: {t_load:.1f}s")

    print("\n--- Processing videos ---")
    pred_maxn = {}

    for video_path in available_videos:
        video_name = Path(video_path).name
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET - 30:
            print(f"  Approaching time budget, stopping at {video_name}")
            break

        print(f"\n  Processing: {video_name}")
        t1 = time.time()

        max_count, frame_counts = count_fish_background_subtraction(video_path)

        pred_maxn[video_name] = max_count
        print(f"  MaxN prediction: {max_count} ({time.time()-t1:.1f}s)")
        print(f"  Frames processed: {len(frame_counts)}")

        if frame_counts:
            counts = [c for _, c in frame_counts]
            print(f"  Count stats: mean={np.mean(counts):.1f}, "
                  f"max={np.max(counts)}, std={np.std(counts):.1f}")

    print("\n--- Evaluation ---")
    eval_result = evaluate_maxn_predictions(pred_maxn, true_maxn)
    print_evaluation(eval_result)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RESULTS_DIR, f"result_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump({"config": config, "evaluation": eval_result}, f, indent=2, default=str)

    t_total = time.time() - t_start

    print("\n---")
    print(f"composite_score:  {eval_result.get('composite_score', 0.0):.6f}")
    print(f"mae:              {eval_result.get('mae', 0.0):.2f}")
    print(f"mre:              {eval_result.get('mre', 0.0):.4f}")
    print(f"correlation:      {eval_result.get('correlation', 0.0):.4f}")
    print(f"n_videos:         {eval_result.get('n_videos', 0)}")
    print(f"tier:             {TIER}")
    print(f"method:           background_subtraction")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"device:           {DEVICE}")


if __name__ == "__main__":
    main()
