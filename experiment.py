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

TIER = 2

SAMPLE_FPS = 1
SCALE_FACTOR = 0.5
WARMUP_FRAMES = 20

MOG2_HISTORY = 200
MOG2_VAR_THRESHOLD = 40
MOG2_DETECT_SHADOWS = True
MIN_CONTOUR_AREA = 100
MORPH_KERNEL_SIZE = 3
BLUR_SIZE = 5
SINGLE_FISH_AREA = 234

FLOW_THRESHOLD = 2.0
COMBINED_WEIGHT_MOG2 = 0.7
COMBINED_WEIGHT_FLOW = 0.3


def count_fish_combined(video_path):
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, []

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(native_fps / SAMPLE_FPS))

    print(f"  Video: {Path(video_path).name}, {native_fps:.1f}fps, "
          f"{total_frames} frames, interval={frame_interval}")

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    sf2 = SCALE_FACTOR * SCALE_FACTOR
    scaled_min_area = int(MIN_CONTOUR_AREA * sf2)
    scaled_fish_area = SINGLE_FISH_AREA * sf2

    frame_counts = []
    frame_idx = 0
    max_count = 0
    prev_gray = None

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_SIZE, BLUR_SIZE), 0)

        fg_mask = bg_sub.apply(blurred)
        if MOG2_DETECT_SHADOWS:
            fg_mask = (fg_mask == 255).astype(np.uint8) * 255
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mog2_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= scaled_min_area:
                fish_in_blob = max(1, int(round(area / scaled_fish_area)))
                mog2_count += fish_in_blob

        flow_count = 0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            motion_mask = (mag > FLOW_THRESHOLD).astype(np.uint8) * 255
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
            flow_contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in flow_contours:
                area = cv2.contourArea(cnt)
                if area >= scaled_min_area:
                    fish_in_blob = max(1, int(round(area / scaled_fish_area)))
                    flow_count += fish_in_blob

        combined = int(COMBINED_WEIGHT_MOG2 * mog2_count + COMBINED_WEIGHT_FLOW * flow_count)

        prev_gray = gray.copy()
        time_sec = frame_idx / native_fps
        frame_counts.append((time_sec, combined))
        if len(frame_counts) > WARMUP_FRAMES and combined > max_count:
            max_count = combined

        frame_idx += frame_interval

    cap.release()
    print(f"  Processed {len(frame_counts)} frames")

    return max_count, frame_counts


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"BRUV Fish Counting — Tier {TIER}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    config = {
        "tier": TIER,
        "method": "mog2_optflow_combined",
        "sample_fps": SAMPLE_FPS,
        "scale_factor": SCALE_FACTOR,
        "single_fish_area": SINGLE_FISH_AREA,
        "flow_threshold": FLOW_THRESHOLD,
        "weights": f"{COMBINED_WEIGHT_MOG2}/{COMBINED_WEIGHT_FLOW}",
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

    print(f"Data loaded: {time.time() - t_start:.1f}s")

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
        max_count, frame_counts = count_fish_combined(video_path)
        pred_maxn[video_name] = max_count
        print(f"  MaxN prediction: {max_count} ({time.time()-t1:.1f}s)")

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
    print(f"method:           mog2_optflow_combined")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"device:           {DEVICE}")


if __name__ == "__main__":
    main()
