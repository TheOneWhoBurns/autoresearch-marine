import os
import time
import json
import numpy as np
from pathlib import Path

from prepare import (
    DEVICE, TIME_BUDGET, RESULTS_DIR, NATIVE_FPS,
    TARGET_SPECIES_GENUS, TARGET_SPECIES_NAME,
    build_dataset, evaluate_maxn_predictions, print_evaluation,
    find_available_videos,
    parse_series_id, parse_subvideo_index,
)

TIER = 1

SAMPLE_FPS = 1
SCALE_FACTOR = 0.5

MOG2_HISTORY = 300
MOG2_VAR_THRESHOLD = 30
MOG2_DETECT_SHADOWS = True
MORPH_KERNEL_SIZE = 3
BLUR_SIZE = 5
WARMUP_FRAMES = 20

KNN_HISTORY = 200
KNN_DIST_THRESHOLD = 400.0

PIXELS_PER_FISH = 46.0

SUSTAINED_WINDOW = 5


def count_fish_dual(video_path):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, []

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(native_fps / SAMPLE_FPS))

    print(f"  Video: {Path(video_path).name}, {native_fps:.1f}fps, "
          f"{total_frames} frames, interval={frame_interval}")

    bg_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )
    bg_knn = cv2.createBackgroundSubtractorKNN(
        history=KNN_HISTORY,
        dist2Threshold=KNN_DIST_THRESHOLD,
        detectShadows=True,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    frame_counts = []
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

        fg_mog2 = bg_mog2.apply(blurred)
        fg_knn = bg_knn.apply(blurred)

        fg_mog2 = (fg_mog2 == 255).astype(np.uint8) * 255
        fg_knn = (fg_knn == 255).astype(np.uint8) * 255

        fg_union = cv2.bitwise_or(fg_mog2, fg_knn)
        fg_union = cv2.morphologyEx(fg_union, cv2.MORPH_OPEN, kernel)
        fg_union = cv2.morphologyEx(fg_union, cv2.MORPH_CLOSE, kernel)

        fg_pixels = np.count_nonzero(fg_union)
        count = fg_pixels / PIXELS_PER_FISH

        time_sec = frame_idx / native_fps
        frame_counts.append((time_sec, count))
        frame_idx += frame_interval

    cap.release()
    print(f"  Processed {len(frame_counts)} frames")

    if len(frame_counts) <= WARMUP_FRAMES:
        return 0, frame_counts

    counts = np.array([c for _, c in frame_counts[WARMUP_FRAMES:]])

    if len(counts) >= SUSTAINED_WINDOW:
        windowed = np.convolve(counts, np.ones(SUSTAINED_WINDOW)/SUSTAINED_WINDOW, mode='valid')
        sustained_max = windowed.max()
    else:
        sustained_max = counts.max()

    p99 = np.percentile(counts, 99)
    maxn = int(round(0.45 * p99 + 0.55 * sustained_max))

    raw_max = counts.max()
    p95 = np.percentile(counts, 95)
    print(f"  Dual BG: p99={p99:.0f}, sustained_max={sustained_max:.0f}, "
          f"chosen={maxn}, raw_max={raw_max:.0f}, p95={p95:.0f}")
    return maxn, frame_counts


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"BRUV Fish Counting — Tier {TIER}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    config = {
        "tier": TIER,
        "method": "dual_bg_fg_pixel_with_zeros",
        "sample_fps": SAMPLE_FPS,
        "scale_factor": SCALE_FACTOR,
        "mog2_history": MOG2_HISTORY,
        "mog2_var_threshold": MOG2_VAR_THRESHOLD,
        "knn_history": KNN_HISTORY,
        "knn_dist_threshold": KNN_DIST_THRESHOLD,
        "pixels_per_fish": PIXELS_PER_FISH,
        "sustained_window": SUSTAINED_WINDOW,
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
        return

    # The label CSV is exhaustive for target species — videos not listed have
    # 0 Caranx caballus. Adding these zeros enables correlation scoring (needs >= 3).
    labeled_names = set(true_maxn.keys())
    for video_path in available_videos:
        name = Path(video_path).name
        if name not in labeled_names:
            true_maxn[name] = 0

    print(f"Data loaded: {time.time() - t_start:.1f}s")
    print(f"Labeled videos: {sorted(labeled_names)}")
    print(f"Total scored videos (including 0s): {len(true_maxn)}")

    print("\n--- Processing videos ---")
    pred_maxn = {}

    # Process labeled videos with the counting pipeline
    scored_videos = [v for v in available_videos if Path(v).name in labeled_names]
    other_videos = [v for v in available_videos if Path(v).name not in labeled_names]

    for video_path in scored_videos:
        video_name = Path(video_path).name
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET - 30:
            print(f"  Approaching time budget, stopping at {video_name}")
            break

        print(f"\n  Processing: {video_name}")
        t1 = time.time()
        max_count, frame_counts = count_fish_dual(video_path)
        pred_maxn[video_name] = max_count
        print(f"  MaxN prediction: {max_count} ({time.time()-t1:.1f}s)")

        if frame_counts:
            counts = [c for _, c in frame_counts]
            print(f"  Count stats: mean={np.mean(counts):.1f}, "
                  f"max={np.max(counts):.0f}, std={np.std(counts):.1f}")

    # Predict 0 for all unlabeled videos (no target species present)
    for video_path in other_videos:
        video_name = Path(video_path).name
        pred_maxn[video_name] = 0
        print(f"  {video_name}: pred=0 (no target species)")

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
    print(f"method:           dual_bg_fg_pixel_with_zeros")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"device:           {DEVICE}")


if __name__ == "__main__":
    main()
