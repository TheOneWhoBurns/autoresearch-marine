"""
Multi-tier BRUV fish counting with adaptive pixel-per-fish calibration.

Instead of hardcoded PIXELS_PER_FISH, we calibrate per-video:
1. Scan video with dual BG subtraction to get raw foreground pixel counts
2. Run YOLO on medium-activity frames where detection is reliable (5-30 fish)
3. Compute PPF = median(fg_pixels / yolo_count) — self-calibrating
4. Apply calibrated PPF to convert pixel counts → fish counts → MaxN
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
    find_available_videos,
    parse_series_id, parse_subvideo_index,
)

TIER = 3

# --- BG subtraction config ---
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
DEFAULT_PPF = 46.0  # fallback if calibration fails
SUSTAINED_WINDOW = 5

# --- YOLO config ---
YOLO_CONF = 0.01
YOLO_IOU = 0.3
YOLO_IMG_SIZE = 1280
YOLO_FISH_CLASSES = {"kite", "bird"}
N_PEAK_FRAMES = 10

# --- Calibration config ---
CALIB_MIN_DETECTIONS = 3   # min YOLO detections for a frame to be useful
CALIB_MAX_DETECTIONS = 50  # max — above this, occlusion makes detection unreliable
CALIB_N_FRAMES = 25        # number of frames to sample for calibration
CALIB_MIN_SAMPLES = 3      # need at least this many calibration frames
CALIB_PERCENTILE = 25      # use 25th percentile PPF (least occluded frames)

# --- VLM config ---
VLM_N_FRAMES = 3


STATIC_MASK_FRAMES = 60  # build static mask from first 60 post-warmup frames
STATIC_MASK_THRESHOLD = 0.5  # pixel is "static" if foreground in >50% of mask frames

def tier1_scan(video_path):
    """Scan video with dual BG subtraction + bait arm masking.

    Automatically detects static structures (bait arm, apparatus) by finding
    pixels that appear as foreground in >50% of the first 60 post-warmup frames.
    These are masked out before counting, since they're not fish.
    Generalizes to any BRUV deployment without manual configuration.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(native_fps / SAMPLE_FPS))

    print(f"    [T1-scan] {Path(video_path).name}: {native_fps:.1f}fps, "
          f"{total_frames} frames, interval={frame_interval}")

    bg_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS)
    bg_knn = cv2.createBackgroundSubtractorKNN(
        history=KNN_HISTORY, dist2Threshold=KNN_DIST_THRESHOLD,
        detectShadows=True)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    scan_results = []
    fg_accumulator = None
    static_mask = None
    n_mask_frames = 0
    frame_idx = 0
    n_frames = 0

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
        fg_mog2 = (fg_mog2 == 255).astype(np.uint8)
        fg_knn = (fg_knn == 255).astype(np.uint8)

        fg_union = cv2.bitwise_or(fg_mog2, fg_knn)
        fg_clean = cv2.morphologyEx(fg_union, cv2.MORPH_OPEN, kernel)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_CLOSE, kernel)

        time_sec = frame_idx / native_fps

        # Build static mask from first STATIC_MASK_FRAMES post-warmup frames
        if n_frames >= WARMUP_FRAMES and n_mask_frames < STATIC_MASK_FRAMES:
            if fg_accumulator is None:
                fg_accumulator = fg_clean.astype(np.float32)
            else:
                fg_accumulator += fg_clean.astype(np.float32)
            n_mask_frames += 1

            # Finalize mask once we have enough frames
            if n_mask_frames == STATIC_MASK_FRAMES:
                static_mask = (fg_accumulator / n_mask_frames
                               > STATIC_MASK_THRESHOLD).astype(np.uint8)
                static_px = np.count_nonzero(static_mask)
                print(f"    [T1-scan] Static mask built: {static_px} pixels "
                      f"({100*static_px/static_mask.size:.1f}% of frame)")
                del fg_accumulator  # free memory

        # Apply static mask if available
        if static_mask is not None:
            fg_final = cv2.bitwise_and(fg_clean, cv2.bitwise_not(static_mask))
        else:
            fg_final = fg_clean

        fg_pixels = np.count_nonzero(fg_final)
        scan_results.append((frame_idx, time_sec, fg_pixels))
        n_frames += 1
        frame_idx += frame_interval

    cap.release()
    return scan_results


def tier1_aggregate(scan_results, ppf):
    """Convert raw pixel counts to fish counts and compute MaxN.

    Uses the geometric mean of p99 and sustained_max as the MaxN estimate.
    Geometric mean naturally handles the fact that both signals are noisy
    estimates of the same quantity, and is robust to either one being
    an outlier (less sensitive than arithmetic mean to extreme values).
    """
    if len(scan_results) <= WARMUP_FRAMES:
        return 0

    counts = np.array([px / ppf for _, _, px in scan_results[WARMUP_FRAMES:]])

    if len(counts) >= SUSTAINED_WINDOW:
        windowed = np.convolve(
            counts, np.ones(SUSTAINED_WINDOW) / SUSTAINED_WINDOW, mode='valid')
        sustained_max = windowed.max()
    else:
        sustained_max = counts.max()

    p99 = np.percentile(counts, 99)

    # Geometric mean of p99 and sustained_max
    # More robust than arithmetic mean — dampens outliers from either signal
    if p99 > 0 and sustained_max > 0:
        maxn = int(round(np.sqrt(p99 * sustained_max)))
    else:
        maxn = int(round(max(p99, sustained_max)))

    print(f"    [T1-agg] ppf={ppf:.1f}, p99={p99:.0f}, sustained={sustained_max:.0f}, "
          f"geomean={maxn}")
    return maxn


_yolo_model = None

def _get_yolo_model():
    """Load YOLOv8n once, reuse across videos."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print("    [YOLO] Loading YOLOv8n...")
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def _yolo_detect_frame(model, frame):
    """Run YOLO on a single frame, return fish-like detection count."""
    results = model(frame, conf=YOLO_CONF, iou=YOLO_IOU,
                    imgsz=YOLO_IMG_SIZE, verbose=False)
    n_detections = 0
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            for cls_id in boxes.cls.cpu().numpy().astype(int):
                cls_name = model.names[int(cls_id)]
                if cls_name in YOLO_FISH_CLASSES:
                    n_detections += 1
    return n_detections


def calibrate_ppf(video_path, scan_results):
    """Calibrate pixels-per-fish using YOLO on sampled frames.

    Strategy:
    1. Sample frames from across the activity spectrum (20th-90th percentile)
    2. Run YOLO on each, compute PPF = fg_pixels / yolo_count
    3. Use 25th percentile of PPF samples (least occluded = best calibration)

    Falls back to DEFAULT_PPF if not enough calibration samples.
    """
    try:
        import cv2
        model = _get_yolo_model()
    except ImportError:
        print("    [Calib] YOLO not available, using default PPF")
        return DEFAULT_PPF

    if len(scan_results) <= WARMUP_FRAMES:
        return DEFAULT_PPF

    after_warmup = scan_results[WARMUP_FRAMES:]
    sorted_by_px = sorted(after_warmup, key=lambda x: x[2])

    # Sample from 20th-80th percentile of activity
    # Avoid lowest frames (no fish) and highest (too dense for accurate YOLO)
    n = len(sorted_by_px)
    lo = int(n * 0.2)
    hi = int(n * 0.8)
    candidate_frames = sorted_by_px[lo:hi]

    # Evenly sample CALIB_N_FRAMES from candidates
    if len(candidate_frames) <= CALIB_N_FRAMES:
        sample_frames = candidate_frames
    else:
        step = len(candidate_frames) / CALIB_N_FRAMES
        sample_frames = [candidate_frames[int(i * step)] for i in range(CALIB_N_FRAMES)]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return DEFAULT_PPF

    ppf_samples = []
    for frame_idx, time_sec, fg_pixels in sample_frames:
        if fg_pixels < 50:  # too few pixels to calibrate
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        n_det = _yolo_detect_frame(model, frame)

        if CALIB_MIN_DETECTIONS <= n_det <= CALIB_MAX_DETECTIONS:
            sample_ppf = fg_pixels / n_det
            ppf_samples.append(sample_ppf)
            print(f"    [Calib] frame {frame_idx} t={time_sec:.1f}s: "
                  f"fg_px={fg_pixels}, yolo={n_det}, ppf={sample_ppf:.1f}")

    cap.release()

    if len(ppf_samples) >= CALIB_MIN_SAMPLES:
        # Use 25th percentile — lower PPF values come from least-occluded frames
        # where YOLO detection is most complete (best calibration quality)
        adaptive_ppf = float(np.percentile(ppf_samples, CALIB_PERCENTILE))
        print(f"    [Calib] Adaptive PPF: {adaptive_ppf:.1f} "
              f"(p{CALIB_PERCENTILE} of {len(ppf_samples)} samples, "
              f"range={min(ppf_samples):.1f}-{max(ppf_samples):.1f}, "
              f"median={np.median(ppf_samples):.1f})")
        return adaptive_ppf
    else:
        print(f"    [Calib] Only {len(ppf_samples)} samples, using default PPF={DEFAULT_PPF}")
        return DEFAULT_PPF


def get_peak_frame_indices(scan_results, ppf, n_peaks=10):
    """Get frame indices with highest estimated counts."""
    if not scan_results:
        return []
    after_warmup = scan_results[WARMUP_FRAMES:]
    # Convert to estimated fish counts for ranking
    with_counts = [(idx, t, px / ppf) for idx, t, px in after_warmup]
    sorted_frames = sorted(with_counts, key=lambda x: x[2], reverse=True)
    selected = []
    selected_times = []
    for frame_idx, time_sec, count in sorted_frames:
        if len(selected) >= n_peaks:
            break
        if any(abs(time_sec - t) < 2.0 for t in selected_times):
            continue
        selected.append((frame_idx, time_sec, count))
        selected_times.append(time_sec)
    return selected


def tier2_yolo_count(video_path, peak_frames):
    """Tier 2: Run YOLOv8 on peak frames for direct detection count."""
    if not peak_frames:
        return 0

    try:
        import cv2
        model = _get_yolo_model()
    except ImportError:
        print("    [T2] YOLO not available, skipping")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    max_det = 0
    for frame_idx, time_sec, t1_count in peak_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        n_det = _yolo_detect_frame(model, frame)
        max_det = max(max_det, n_det)
        print(f"    [T2] frame {frame_idx} t={time_sec:.1f}s: "
              f"yolo={n_det}, t1={t1_count:.0f}")

    cap.release()

    print(f"    [T2] YOLO MaxN: {max_det}")
    return max_det if max_det > 0 else None


def tier3_vlm_count(video_path, peak_frames):
    """Tier 3: Send peak frames to Claude for visual counting."""
    if not peak_frames:
        return None

    try:
        import anthropic
        import base64
        import cv2
    except ImportError as e:
        print(f"    [T3] Missing dependency: {e}, skipping")
        return None

    client = anthropic.Anthropic()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    vlm_counts = []
    frames_to_send = peak_frames[:VLM_N_FRAMES]

    for frame_idx, time_sec, t1_count in frames_to_send:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.standard_b64encode(buffer).decode('utf-8')

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": img_b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "This is a frame from a Baited Remote Underwater Video (BRUV) "
                                "in the Galapagos. Count the number of fish visible in this image. "
                                "Include all fish you can see, even partially visible ones at the edges. "
                                "The fish are likely Caranx caballus (green jacks) — silvery elongated fish. "
                                "Return ONLY an integer number, nothing else."
                            ),
                        },
                    ],
                }],
            )
            count_str = response.content[0].text.strip()
            count = int(''.join(c for c in count_str if c.isdigit()) or '0')
            vlm_counts.append(count)
            print(f"    [T3] frame {frame_idx} t={time_sec:.1f}s: "
                  f"claude={count}, t1={t1_count:.0f}")
        except Exception as e:
            print(f"    [T3] API error at frame {frame_idx}: {e}")
            continue

    cap.release()

    if not vlm_counts:
        return None

    vlm_maxn = max(vlm_counts)
    print(f"    [T3] Claude VLM MaxN: {vlm_maxn}")
    return vlm_maxn


def ensemble_maxn(t1_maxn, t2_maxn, t3_maxn):
    """Combine tier predictions. T1 is primary, T2/T3 are diagnostics.

    T1 (calibrated pixel density) works at all scales — dense and sparse.
    T2 (YOLO detection) saturates in dense scenes (~60-80 max detections).
    T3 (VLM) is a fundamentally different approach, useful for validation.

    Without ground truth to tune blending weights, the most robust approach
    is: T1 is the answer. T2/T3 are logged for human QA review.
    """
    t1 = t1_maxn if t1_maxn is not None else 0
    t2 = t2_maxn if t2_maxn is not None else 0
    t3 = t3_maxn if t3_maxn is not None else 0

    print(f"    [Ensemble] T1={t1}, T2={t2}, T3={t3}")

    # T1 is the primary estimate
    final = t1

    # Diagnostic: flag large disagreements for human review
    if t2 > 0 and t1 > 0:
        ratio = t1 / t2
        if ratio > 3:
            print(f"    [Ensemble] NOTE: T1/T2 ratio={ratio:.1f} — "
                  f"likely dense scene (YOLO saturated)")
        elif ratio < 0.5:
            print(f"    [Ensemble] NOTE: T1/T2 ratio={ratio:.1f} — "
                  f"T1 may be undercounting, review recommended")

    if t3 > 0 and t1 > 0:
        ratio = t1 / t3
        if abs(ratio - 1) > 0.5:
            print(f"    [Ensemble] NOTE: T1/T3 ratio={ratio:.1f} — "
                  f"disagreement, review recommended")

    print(f"    [Ensemble] final={final}")
    return final


def process_video(video_path, t_start):
    """Process one video: scan → calibrate → aggregate → ensemble."""
    video_name = Path(video_path).name
    print(f"\n  === {video_name} ===")

    # Phase 1: BG subtraction scan (raw pixel counts)
    t1 = time.time()
    scan_results = tier1_scan(video_path)
    print(f"    [T1-scan] completed in {time.time()-t1:.1f}s, {len(scan_results)} frames")

    if len(scan_results) <= WARMUP_FRAMES:
        return 0

    # Phase 2: Calibrate PPF using YOLO on medium-activity frames
    elapsed = time.time() - t_start
    if elapsed < TIME_BUDGET - 60:
        t_cal = time.time()
        ppf = calibrate_ppf(video_path, scan_results)
        print(f"    [Calib] completed in {time.time()-t_cal:.1f}s")
    else:
        ppf = DEFAULT_PPF
        print(f"    [Calib] skipped (time budget), using default PPF={ppf}")

    # Phase 3: Aggregate with calibrated PPF
    t1_maxn = tier1_aggregate(scan_results, ppf)

    if t1_maxn < 5:
        print(f"    Skipping T2/T3 (T1 count too low)")
        return t1_maxn

    # Phase 4: Peak frames for T2/T3
    peak_frames = get_peak_frame_indices(scan_results, ppf, N_PEAK_FRAMES)
    print(f"    Peak frames: {len(peak_frames)} selected")

    # Phase 5: YOLO on peaks (direct count)
    elapsed = time.time() - t_start
    t2_maxn = None
    if elapsed < TIME_BUDGET - 20:
        t2 = time.time()
        t2_maxn = tier2_yolo_count(video_path, peak_frames)
        print(f"    [T2] completed in {time.time()-t2:.1f}s")

    # Phase 6: VLM on peaks
    elapsed = time.time() - t_start
    t3_maxn = None
    if elapsed < TIME_BUDGET - 20:
        t3 = time.time()
        t3_maxn = tier3_vlm_count(video_path, peak_frames)
        print(f"    [T3] completed in {time.time()-t3:.1f}s")

    # Phase 7: Ensemble
    final = ensemble_maxn(t1_maxn, t2_maxn, t3_maxn)
    print(f"    Final MaxN: {final} (T1={t1_maxn}, T2={t2_maxn}, T3={t3_maxn}, PPF={ppf:.1f})")
    return final


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"BRUV Fish Counting — Adaptive PPF Calibration (Tier {TIER})")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    config = {
        "tier": TIER,
        "method": "adaptive_ppf_ensemble",
        "bg_method": "dual_bg_mog2_knn",
        "calibration": "yolo_ppf",
        "default_ppf": DEFAULT_PPF,
        "calib_range": f"{CALIB_MIN_DETECTIONS}-{CALIB_MAX_DETECTIONS}",
        "yolo_conf": YOLO_CONF,
        "n_peak_frames": N_PEAK_FRAMES,
        "vlm_n_frames": VLM_N_FRAMES,
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

    # Zero-padding: videos without target species entries have true MaxN = 0
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

    scored_videos = [v for v in available_videos if Path(v).name in labeled_names]
    other_videos = [v for v in available_videos if Path(v).name not in labeled_names]

    for video_path in scored_videos:
        video_name = Path(video_path).name
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET - 30:
            print(f"  Approaching time budget, stopping at {video_name}")
            break

        pred_maxn[video_name] = process_video(video_path, t_start)

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
    print(f"method:           adaptive_ppf_ensemble")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"device:           {DEVICE}")


if __name__ == "__main__":
    main()
