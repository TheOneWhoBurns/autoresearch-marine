"""
BRUV Fish Counting Autoresearch — Experiment Script
THIS IS THE ONLY FILE THE AGENT MODIFIES.

Starting point: Tier 1 — MOG2 background subtraction + contour counting.
The agent evolves this through Tiers 1-3, choosing the best approach.

Metric: maxn_score from prepare.evaluate_counting() — higher is better.

Usage: python3 experiment.py > run.log 2>&1
"""

import os
import time
import json
import numpy as np
import cv2

from prepare import (
    TARGET_FPS, SOURCE_FPS, FRAME_SKIP, FRAME_WIDTH, FRAME_HEIGHT,
    TIME_BUDGET, CACHE_DIR, RESULTS_DIR, DEVICE,
    TARGET_SPECIES, GROUND_TRUTH_MAXN,
    find_video_files, find_frame_dirs, load_labels, get_ground_truth_maxn,
    load_frame, extract_frames_from_video,
    compute_foreground_mask, count_contours, apply_morphology,
    generate_color_views,
    evaluate_counting, evaluate_discovery,
    build_frame_dataset,
)

# ---------------------------------------------------------------------------
# TIER: current approach (agent updates this as it progresses)
# ---------------------------------------------------------------------------
TIER = 1  # 1=background subtraction, 2=object detection, 3=crowd counting

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Background subtraction
MOG2_HISTORY = 500
MOG2_VAR_THRESHOLD = 16
SHADOW_DETECTION = False

# Morphology
MORPH_KERNEL_SIZE = 5
MORPH_OPERATIONS = ("close", "open")

# Contour filtering
MIN_CONTOUR_AREA = 100      # minimum blob area (pixels)
MAX_CONTOUR_AREA = 50000    # maximum blob area (pixels)

# Frame processing
MAX_FRAMES = None           # None = all available frames
RESIZE = None               # None = original resolution, or (W, H)

# Multi-tint (scuba glasses) — use color space views for better detection
USE_MULTI_TINT = True
TINT_VIEWS = ["clahe", "lab_a", "hsv_s"]  # which views to use for detection

# ---------------------------------------------------------------------------
# Counting methods (Tier 1: background subtraction)
# ---------------------------------------------------------------------------

def count_with_mog2(frames):
    """
    Count fish per frame using MOG2 background subtraction + contour counting.
    Returns list of per-frame counts.
    """
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=SHADOW_DETECTION
    )

    per_frame_counts = []
    for i, frame in enumerate(frames):
        # Apply background subtraction
        fg_mask = bg_sub.apply(frame)

        # Clean up mask with morphology
        cleaned = apply_morphology(fg_mask, MORPH_KERNEL_SIZE, MORPH_OPERATIONS)

        # Count contours
        n_fish, contours = count_contours(cleaned, MIN_CONTOUR_AREA, MAX_CONTOUR_AREA)

        # If multi-tint enabled, also count on alternate views and take max
        if USE_MULTI_TINT:
            views = generate_color_views(frame)
            for tint_name in TINT_VIEWS:
                if tint_name in views:
                    tint_img = views[tint_name]
                    if tint_img.ndim == 2:
                        # Single channel — threshold-based counting
                        _, binary = cv2.threshold(tint_img, 0, 255,
                                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        binary_cleaned = apply_morphology(binary, MORPH_KERNEL_SIZE,
                                                           MORPH_OPERATIONS)
                        n_tint, _ = count_contours(binary_cleaned,
                                                    MIN_CONTOUR_AREA, MAX_CONTOUR_AREA)
                        n_fish = max(n_fish, n_tint)

        per_frame_counts.append(n_fish)

        if (i + 1) % 100 == 0:
            print(f"  Processed: {i+1}/{len(frames)} frames, current count: {n_fish}")

    return per_frame_counts


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_counts(per_frame_counts, metadata=None):
    """Print count summary with temporal analysis."""
    counts = np.array(per_frame_counts)
    print(f"\n{'='*60}")
    print(f"Count Analysis")
    print(f"{'='*60}")
    print(f"  Frames analyzed: {len(counts)}")
    print(f"  Max count (MaxN): {np.max(counts)}")
    print(f"  Mean count: {np.mean(counts):.1f}")
    print(f"  Median count: {np.median(counts):.1f}")
    print(f"  Std dev: {np.std(counts):.1f}")

    # Find peak frames
    peak_idx = np.argmax(counts)
    print(f"  Peak at frame index: {peak_idx}")

    # Temporal distribution
    if len(counts) > 20:
        quarters = np.array_split(counts, 4)
        for i, q in enumerate(quarters):
            print(f"  Quarter {i+1}: max={np.max(q)}, mean={np.mean(q):.1f}")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    t_start = time.time()
    print("=" * 60)
    print(f"BRUV Fish Counting Autoresearch — Tier {TIER}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    config = {
        "tier": TIER,
        "mog2_history": MOG2_HISTORY, "mog2_var_threshold": MOG2_VAR_THRESHOLD,
        "morph_kernel": MORPH_KERNEL_SIZE, "morph_ops": list(MORPH_OPERATIONS),
        "min_contour_area": MIN_CONTOUR_AREA, "max_contour_area": MAX_CONTOUR_AREA,
        "use_multi_tint": USE_MULTI_TINT, "tint_views": TINT_VIEWS,
        "max_frames": MAX_FRAMES, "resize": RESIZE,
    }
    print(f"\nConfig: {json.dumps(config, indent=2)}")

    # Load data
    print("\n--- Loading data ---")
    frames, metadata = build_frame_dataset(
        max_frames_per_video=MAX_FRAMES, resize=RESIZE
    )
    if not frames:
        print("ERROR: No frames available. Place data in data/videos/ or data/frames/")
        print(f"maxn_score:       0.000000")
        return

    t_load = time.time() - t_start
    print(f"Data loaded: {len(frames)} frames, {t_load:.1f}s")

    # Load ground truth
    labels = load_labels()
    gt = get_ground_truth_maxn(labels)
    gt_maxn = list(gt.values())[0] if gt else 251
    print(f"Ground truth MaxN: {gt_maxn}")

    # Count fish
    print(f"\n--- Counting fish (Tier {TIER}) ---")
    t1 = time.time()
    per_frame_counts = count_with_mog2(frames)
    print(f"Counting: {time.time()-t1:.1f}s")

    predicted_maxn = max(per_frame_counts) if per_frame_counts else 0

    # Evaluate (primary metric)
    eval_result = evaluate_counting(
        predicted_maxn=predicted_maxn,
        ground_truth_maxn=gt_maxn,
        per_frame_counts=per_frame_counts,
        method_name=f"tier{TIER}_mog2_contour"
    )

    # Discovery insights
    discovery = evaluate_discovery(per_frame_counts)

    # Analysis
    analyze_counts(per_frame_counts, metadata)

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(RESULTS_DIR, f"result_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump({"config": config, "evaluation": eval_result, "discovery": discovery}, f, indent=2)

    # Final summary (parseable — the autoresearch loop reads these)
    t_total = time.time() - t_start
    print("\n---")
    print(f"maxn_score:       {eval_result['maxn_score']:.6f}")
    print(f"mae:              {eval_result['mae']:.1f}")
    print(f"predicted_maxn:   {eval_result['predicted_maxn']}")
    print(f"ground_truth:     {eval_result['ground_truth_maxn']}")
    print(f"n_frames_processed:{len(per_frame_counts)}")
    print(f"method:           tier{TIER}_mog2_contour")
    print(f"tier:             {TIER}")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"device:           {DEVICE}")

    # Discovery summary
    print(f"\n--- Discovery ---")
    print(f"max_count:        {discovery['max_count']}")
    print(f"mean_count:       {discovery['mean_count']:.1f}")
    if "peak_frame_idx" in discovery:
        print(f"peak_frame_idx:   {discovery['peak_frame_idx']}")
    if "arrival_frame" in discovery:
        print(f"arrival_frame:    {discovery['arrival_frame']}")
        print(f"departure_frame:  {discovery['departure_frame']}")


if __name__ == "__main__":
    main()
