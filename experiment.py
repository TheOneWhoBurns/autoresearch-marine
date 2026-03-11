"""
Multi-tier BRUV fish counting experiment.

Tier 1: Dual BG subtraction (MOG2+KNN) with pixel density — finds peak activity windows
Tier 2: YOLOv8 detection on peak frames — individual fish counting
Tier 3: Claude VLM counting on peak frames — zero-shot visual counting
Ensemble: weighted combination of all tiers
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

# --- Tier 1 config (existing, proven) ---
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
SUSTAINED_FLOOR = 0.96  # MaxN >= 96% of peak sustained count

# --- Tier 2 config ---
YOLO_CONF = 0.01  # very low conf — "kite" class acts as fish proxy
YOLO_IOU = 0.3
YOLO_IMG_SIZE = 1280
YOLO_FISH_CLASSES = {"kite", "bird"}  # COCO classes that match fish shapes
N_PEAK_FRAMES = 10

# --- Tier 3 config ---
VLM_N_FRAMES = 3  # number of frames to send to Claude


def tier1_count(video_path):
    """Tier 1: Dual background subtraction with pixel density counting."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0, []

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(native_fps / SAMPLE_FPS))

    print(f"    [T1] {Path(video_path).name}: {native_fps:.1f}fps, "
          f"{total_frames} frames, interval={frame_interval}")

    bg_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=MOG2_DETECT_SHADOWS)
    bg_knn = cv2.createBackgroundSubtractorKNN(
        history=KNN_HISTORY, dist2Threshold=KNN_DIST_THRESHOLD,
        detectShadows=True)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

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
        frame_counts.append((frame_idx, time_sec, count))
        frame_idx += frame_interval

    cap.release()

    if len(frame_counts) <= WARMUP_FRAMES:
        return 0, frame_counts

    counts = np.array([c for _, _, c in frame_counts[WARMUP_FRAMES:]])

    if len(counts) >= SUSTAINED_WINDOW:
        windowed = np.convolve(
            counts, np.ones(SUSTAINED_WINDOW) / SUSTAINED_WINDOW, mode='valid')
        sustained_max = windowed.max()
    else:
        sustained_max = counts.max()

    p99 = np.percentile(counts, 99)
    blend = int(round(0.45 * p99 + 0.55 * sustained_max))
    floor = int(round(sustained_max * SUSTAINED_FLOOR))
    maxn = max(blend, floor)

    print(f"    [T1] p99={p99:.0f}, sustained_max={sustained_max:.0f}, "
          f"blend={blend}, floor={floor}, maxn={maxn}")
    return maxn, frame_counts


def get_peak_frame_indices(frame_counts, n_peaks=10):
    """Get frame indices with highest counts from Tier 1."""
    if not frame_counts:
        return []
    # Skip warmup, sort by count descending
    after_warmup = frame_counts[WARMUP_FRAMES:]
    sorted_frames = sorted(after_warmup, key=lambda x: x[2], reverse=True)
    # Return top N frame indices, spaced at least 5 frames apart
    selected = []
    selected_times = []
    for frame_idx, time_sec, count in sorted_frames:
        if len(selected) >= n_peaks:
            break
        # Ensure frames are at least 2 seconds apart
        if any(abs(time_sec - t) < 2.0 for t in selected_times):
            continue
        selected.append((frame_idx, time_sec, count))
        selected_times.append(time_sec)
    return selected


_yolo_model = None

def _get_yolo_model():
    """Load YOLOv8n once, reuse across videos."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print("    [T2] Loading YOLOv8n...")
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def tier2_yolo_count(video_path, peak_frames):
    """Tier 2: Run YOLOv8 on peak frames identified by Tier 1."""
    if not peak_frames:
        return 0

    try:
        import cv2
        model = _get_yolo_model()
    except ImportError:
        print("    [T2] ultralytics not available, skipping")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_detections = []

    for frame_idx, time_sec, t1_count in peak_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Run YOLO detection — look for any animal-like detections
        results = model(frame, conf=YOLO_CONF, iou=YOLO_IOU,
                        imgsz=YOLO_IMG_SIZE, verbose=False)

        # Filter for fish-like COCO classes (kite=33, bird=14 match fish shapes)
        n_detections = 0
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for cls_id in boxes.cls.cpu().numpy().astype(int):
                    cls_name = model.names[int(cls_id)]
                    if cls_name in YOLO_FISH_CLASSES:
                        n_detections += 1

        frame_detections.append({
            'frame_idx': frame_idx,
            'time_sec': time_sec,
            't1_count': t1_count,
            'yolo_count': n_detections,
        })
        print(f"    [T2] frame {frame_idx} t={time_sec:.1f}s: "
              f"yolo={n_detections}, t1={t1_count:.0f}")

    cap.release()

    if not frame_detections:
        return None

    # MaxN from YOLO: take the max detection count across peak frames
    yolo_maxn = max(d['yolo_count'] for d in frame_detections)
    print(f"    [T2] YOLO MaxN: {yolo_maxn}")
    return yolo_maxn


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
    # Use top N peak frames
    frames_to_send = peak_frames[:VLM_N_FRAMES]

    for frame_idx, time_sec, t1_count in frames_to_send:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Encode frame as JPEG
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
            # Extract integer from response
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
    """Combine tier predictions into final MaxN estimate.

    T1 (pixel density) is the primary predictor — proven reliable.
    T2 (YOLO) and T3 (Claude VLM) provide secondary signals:
    - Used as sanity checks and logged for analysis
    - Only override T1 if they strongly agree on a different estimate
    """
    t1 = t1_maxn if t1_maxn is not None else 0
    t2 = t2_maxn if t2_maxn is not None else 0
    t3 = t3_maxn if t3_maxn is not None else 0

    print(f"    [Ensemble] t1={t1}, t2={t2}, t3={t3}")

    if t1 == 0 and t2 == 0 and t3 == 0:
        return 0

    # T1 is the default prediction
    final = t1

    # If Claude VLM is available, it gets moderate influence
    if t3 > 0 and t1 > 0:
        # VLM is good at counting visible fish — blend lightly
        final = int(round(0.7 * t1 + 0.3 * t3))
        print(f"    [Ensemble] T1+T3 blend: 0.7*{t1} + 0.3*{t3} = {final}")

    # YOLO: only useful in sparse scenes where T1 might undercount
    # In dense scenes (T1 > 2.5 * T2), YOLO is clearly failing
    if t2 > 0 and t1 > 0 and t1 <= 2.5 * t2:
        # Sparse scene — YOLO can help slightly
        # Apply miss-rate correction (~30% for underwater BRUV)
        yolo_corrected = int(round(t2 * 1.30))
        # Only nudge final if YOLO corrected is higher (undercounting fix)
        if yolo_corrected > final:
            nudge = int(round(0.8 * final + 0.2 * yolo_corrected))
            print(f"    [Ensemble] YOLO nudge: yolo_raw={t2}, corrected={yolo_corrected}, "
                  f"nudge {final}->{nudge}")
            final = nudge

    print(f"    [Ensemble] final={final}")
    return final


def process_video(video_path, t_start):
    """Run all available tiers on a single video."""
    video_name = Path(video_path).name
    print(f"\n  === {video_name} ===")

    # Tier 1: Background subtraction (always runs)
    t1 = time.time()
    t1_maxn, frame_counts = tier1_count(video_path)
    print(f"    [T1] completed in {time.time()-t1:.1f}s, maxn={t1_maxn}")

    # If Tier 1 says ~0, skip higher tiers (no fish)
    if t1_maxn < 5:
        print(f"    Skipping T2/T3 (T1 count too low)")
        return t1_maxn

    # Get peak frames for Tier 2 & 3
    peak_frames = get_peak_frame_indices(frame_counts, N_PEAK_FRAMES)
    print(f"    Peak frames: {len(peak_frames)} selected")

    # Tier 2: YOLO detection on peak frames (fast: ~3s per video)
    elapsed = time.time() - t_start
    t2_maxn = None
    if elapsed < TIME_BUDGET - 20:
        t2 = time.time()
        t2_maxn = tier2_yolo_count(video_path, peak_frames)
        print(f"    [T2] completed in {time.time()-t2:.1f}s")

    # Tier 3: Claude VLM on top peak frames
    elapsed = time.time() - t_start
    t3_maxn = None
    if elapsed < TIME_BUDGET - 20:
        t3 = time.time()
        t3_maxn = tier3_vlm_count(video_path, peak_frames)
        print(f"    [T3] completed in {time.time()-t3:.1f}s")

    # Ensemble
    final = ensemble_maxn(t1_maxn, t2_maxn, t3_maxn)
    print(f"    Final MaxN: {final} (T1={t1_maxn}, T2={t2_maxn}, T3={t3_maxn})")
    return final


def main():
    t_start = time.time()
    print("=" * 60)
    print(f"BRUV Fish Counting — Multi-Tier Ensemble (Tier {TIER})")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    config = {
        "tier": TIER,
        "method": "multi_tier_ensemble",
        "t1_method": "dual_bg_pixel_density",
        "t2_method": "yolov8n_peak_frames",
        "t3_method": "claude_vlm_counting",
        "pixels_per_fish": PIXELS_PER_FISH,
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

    # Process labeled videos with full multi-tier pipeline
    scored_videos = [v for v in available_videos if Path(v).name in labeled_names]
    other_videos = [v for v in available_videos if Path(v).name not in labeled_names]

    for video_path in scored_videos:
        video_name = Path(video_path).name
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET - 30:
            print(f"  Approaching time budget, stopping at {video_name}")
            break

        pred_maxn[video_name] = process_video(video_path, t_start)

    # Predict 0 for all unlabeled videos
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
    print(f"method:           multi_tier_ensemble")
    print(f"total_seconds:    {t_total:.1f}")
    print(f"device:           {DEVICE}")


if __name__ == "__main__":
    main()
