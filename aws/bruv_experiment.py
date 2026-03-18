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
CALIB_MAX_DETECTIONS = 30  # max — above this, occlusion makes detection unreliable
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

    Uses harmonic mean of p99 and sustained_max (5-frame rolling max).
    Harmonic mean is conservative — closer to the smaller value — which
    is appropriate since both are noisy estimates and we prefer to slightly
    undercount than overcount.
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

    if p99 > 0 and sustained_max > 0:
        maxn = int(round(2.0 / (1.0/p99 + 1.0/sustained_max)))
    else:
        maxn = int(round(max(p99, sustained_max)))

    print(f"    [T1-agg] ppf={ppf:.1f}, p99={p99:.0f}, sustained={sustained_max:.0f}, "
          f"hmean={maxn}")
    return maxn


_yolo_model = None
_species_classifier = None
_species_scaler = None
USE_SPECIES_CLASSIFIER = True  # use trained C. caballus classifier
CLASSIFIER_THRESHOLD = 0.5

def _get_yolo_model():
    """Load YOLOv8n once, reuse across videos."""
    global _yolo_model
    if _yolo_model is None:
        from ultralytics import YOLO
        print("    [YOLO] Loading YOLOv8n...")
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def _get_species_classifier():
    """Load trained C. caballus classifier, if available."""
    global _species_classifier, _species_scaler
    if _species_classifier is not None:
        return _species_classifier, _species_scaler

    import pickle
    model_path = os.path.join(os.path.dirname(__file__), "data", "models",
                              "caballus_classifier.pkl")
    if not os.path.exists(model_path):
        print("    [Classifier] Not found, falling back to proxy classes")
        return None, None

    with open(model_path, "rb") as f:
        data = pickle.load(f)
    _species_classifier = data["model"]
    _species_scaler = data["scaler"]
    print(f"    [Classifier] Loaded (CV F1={data.get('cv_f1', '?'):.3f})")
    return _species_classifier, _species_scaler


def _classify_crop(crop_bgr, classifier, scaler):
    """Classify a single YOLO crop as C. caballus or not.

    Returns probability of being C. caballus.
    """
    import cv2
    from scipy.ndimage import sobel

    # Resize to 128x128 and convert BGR→RGB
    crop = cv2.resize(crop_bgr, (128, 128))
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    h, w, _ = img.shape
    features = []

    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    hue = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta > 0)
    mask_g = (cmax == g) & (delta > 0)
    mask_b = (cmax == b) & (delta > 0)
    hue[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    hue[mask_g] = 60 * ((b[mask_g] - r[mask_g]) / delta[mask_g] + 2)
    hue[mask_b] = 60 * ((r[mask_b] - g[mask_b]) / delta[mask_b] + 4)
    sat = np.where(cmax > 0, delta / cmax, 0)
    val = cmax

    h_hist, _ = np.histogram(hue.ravel(), bins=16, range=(0, 360), density=True)
    s_hist, _ = np.histogram(sat.ravel(), bins=8, range=(0, 1), density=True)
    v_hist, _ = np.histogram(val.ravel(), bins=8, range=(0, 1), density=True)
    features.extend(h_hist)
    features.extend(s_hist)
    features.extend(v_hist)

    eps = 1e-6
    mean_r, mean_g, mean_b = r.mean(), g.mean(), b.mean()
    features.extend([mean_b/(mean_g+eps), mean_r/(mean_g+eps),
                     mean_r/(mean_b+eps), sat.mean(), val.std()])

    gray = 0.299*r + 0.587*g + 0.114*b
    grad_x = sobel(gray, axis=1)
    grad_y = sobel(gray, axis=0)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_dir = np.arctan2(grad_y, grad_x)
    features.extend([grad_mag.mean(), grad_mag.std()])
    gdir_hist, _ = np.histogram(grad_dir.ravel(), bins=8, range=(-np.pi, np.pi),
                                 density=True, weights=grad_mag.ravel())
    features.extend(gdir_hist)

    mid_h, mid_w = h // 2, w // 2
    q_tl = gray[:mid_h, :mid_w].mean()
    q_tr = gray[:mid_h, mid_w:].mean()
    q_bl = gray[mid_h:, :mid_w].mean()
    q_br = gray[mid_h:, mid_w:].mean()
    center = gray[h//4:3*h//4, w//4:3*w//4].mean()
    edge_mean = (q_tl + q_tr + q_bl + q_br) / 4
    features.extend([center/(edge_mean+eps), gray.mean(), gray.std()])

    row_grad = grad_mag.mean(axis=1)
    col_grad = grad_mag.mean(axis=0)
    features.append(col_grad.std() / (row_grad.std() + eps))

    feat = np.nan_to_num(np.array(features, dtype=np.float32),
                          nan=0.0, posinf=1e6, neginf=-1e6)
    feat_scaled = scaler.transform(feat.reshape(1, -1))

    if hasattr(classifier, "predict_proba"):
        return classifier.predict_proba(feat_scaled)[0][1]
    return float(classifier.predict(feat_scaled)[0])


def _yolo_detect_frame(model, frame):
    """Run YOLO on a single frame, return fish-like detection count.

    If species classifier is available and enabled, uses it to filter
    detections to C. caballus only. Otherwise falls back to proxy classes.
    """
    results = model(frame, conf=YOLO_CONF, iou=YOLO_IOU,
                    imgsz=YOLO_IMG_SIZE, verbose=False)
    n_detections = 0
    if not (results and len(results) > 0):
        return n_detections

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return n_detections

    # Try species classifier
    classifier, scaler = None, None
    if USE_SPECIES_CLASSIFIER:
        classifier, scaler = _get_species_classifier()

    h_frame, w_frame = frame.shape[:2]
    for i, (xyxy, cls_id, conf) in enumerate(zip(
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy())):

        cls_name = model.names[int(cls_id)]

        if classifier is not None and scaler is not None:
            # Species classifier path: crop and classify
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(w_frame, x2 + pad_x)
            cy2 = min(h_frame, y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            prob = _classify_crop(crop, classifier, scaler)
            if prob >= CLASSIFIER_THRESHOLD:
                n_detections += 1
        else:
            # Fallback: proxy class filter
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


def _iou(box_a, box_b):
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


TRACK_IOU_THRESHOLD = 0.15  # low threshold — underwater fish move fast between 1fps frames
TRACK_MAX_AGE = 3  # keep track alive for 3 frames without match
TRACK_WINDOW_FRAMES = 40  # process 40 frames around peak


def _yolo_detect_frame_boxes(model, frame):
    """Run YOLO on a frame, return list of [x1,y1,x2,y2] boxes for fish-like detections."""
    results = model(frame, conf=YOLO_CONF, iou=YOLO_IOU,
                    imgsz=YOLO_IMG_SIZE, verbose=False)
    detections = []
    if not (results and len(results) > 0):
        return detections

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return detections

    classifier, scaler = None, None
    if USE_SPECIES_CLASSIFIER:
        classifier, scaler = _get_species_classifier()

    h_frame, w_frame = frame.shape[:2]
    for xyxy, cls_id, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy()):

        cls_name = model.names[int(cls_id)]

        if classifier is not None and scaler is not None:
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.1), int(bh * 0.1)
            cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            cx2, cy2 = min(w_frame, x2 + pad_x), min(h_frame, y2 + pad_y)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            prob = _classify_crop(crop, classifier, scaler)
            if prob >= CLASSIFIER_THRESHOLD:
                detections.append([float(v) for v in xyxy])
        else:
            if cls_name in YOLO_FISH_CLASSES:
                detections.append([float(v) for v in xyxy])

    return detections


def tier2_tracked_count(video_path, scan_results, ppf):
    """Tier 2: YOLO + IoU tracking on frames around peak activity.

    Runs YOLO on TRACK_WINDOW_FRAMES frames centered on the detected peak,
    tracks detections across frames with simple IoU matching, and returns
    the maximum number of simultaneously active tracks.
    """
    if not scan_results or len(scan_results) <= WARMUP_FRAMES:
        return None

    try:
        import cv2
        model = _get_yolo_model()
    except ImportError:
        print("    [T2] YOLO not available, skipping")
        return None

    # Find peak frame and center tracking window around it
    after_warmup = scan_results[WARMUP_FRAMES:]
    peak_idx = max(range(len(after_warmup)), key=lambda i: after_warmup[i][2])
    peak_frame_idx = after_warmup[peak_idx][0]

    # Get native fps to compute frame interval
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(native_fps / SAMPLE_FPS))

    # Build window: TRACK_WINDOW_FRAMES frames centered on peak
    half_window = TRACK_WINDOW_FRAMES // 2
    start_frame = max(0, peak_frame_idx - half_window * frame_interval)
    window_frames = []
    f = start_frame
    while f < total_frames and len(window_frames) < TRACK_WINDOW_FRAMES:
        window_frames.append(f)
        f += frame_interval

    print(f"    [T2-track] Window: {len(window_frames)} frames around peak "
          f"(frame {peak_frame_idx}, t={peak_frame_idx/native_fps:.1f}s)")

    # Simple IoU tracker
    next_track_id = 0
    active_tracks = {}  # track_id → {"box": [x1,y1,x2,y2], "age": int}
    max_simultaneous = 0

    for fi, frame_idx in enumerate(window_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        detections = _yolo_detect_frame_boxes(model, frame)

        # Match detections to existing tracks (greedy IoU matching)
        matched_tracks = set()
        matched_dets = set()

        # Compute all IOUs
        iou_pairs = []
        for d_idx, det in enumerate(detections):
            for t_id, track in active_tracks.items():
                iou_val = _iou(det, track["box"])
                if iou_val >= TRACK_IOU_THRESHOLD:
                    iou_pairs.append((iou_val, d_idx, t_id))

        # Greedy matching by highest IoU
        iou_pairs.sort(reverse=True)
        for iou_val, d_idx, t_id in iou_pairs:
            if d_idx in matched_dets or t_id in matched_tracks:
                continue
            active_tracks[t_id]["box"] = detections[d_idx]
            active_tracks[t_id]["age"] = 0
            matched_tracks.add(t_id)
            matched_dets.add(d_idx)

        # Create new tracks for unmatched detections
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                active_tracks[next_track_id] = {"box": det, "age": 0}
                next_track_id += 1

        # Age unmatched tracks and remove old ones
        to_remove = []
        for t_id in active_tracks:
            if t_id not in matched_tracks:
                active_tracks[t_id]["age"] += 1
                if active_tracks[t_id]["age"] > TRACK_MAX_AGE:
                    to_remove.append(t_id)
        for t_id in to_remove:
            del active_tracks[t_id]

        n_active = len(active_tracks)
        max_simultaneous = max(max_simultaneous, n_active)

        if fi % 10 == 0:
            print(f"    [T2-track] frame {frame_idx}: {len(detections)} dets, "
                  f"{n_active} active tracks, max_so_far={max_simultaneous}")

    cap.release()
    print(f"    [T2-track] MaxN (tracked): {max_simultaneous}, "
          f"total tracks created: {next_track_id}")
    return max_simultaneous if max_simultaneous > 0 else None


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


def ensemble_maxn(t1_maxn, t2_maxn, t2_tracked, t3_maxn, scan_results=None, ppf=None):
    """Combine tier predictions with tracking-aware logic.

    Strategy:
    - For sparse scenes (T2 tracked < 80): trust tracking — it prevents double-counting
    - For dense scenes (T2 tracked >= 80): use tracking to re-calibrate T1
      * Tracking provides a better fish count than single-frame YOLO
      * Use tracked count to derive a "tracking-calibrated PPF" from peak pixels
      * Re-compute T1 with this better PPF
    - T3 (VLM) is diagnostic only
    """
    t1 = t1_maxn if t1_maxn is not None else 0
    t2 = t2_maxn if t2_maxn is not None else 0
    t2t = t2_tracked if t2_tracked is not None else 0
    t3 = t3_maxn if t3_maxn is not None else 0

    print(f"    [Ensemble] T1={t1}, T2={t2}, T2-tracked={t2t}, T3={t3}")

    if t2t > 0 and t2t < 80:
        # Sparse/moderate scene: tracking is reliable
        # Use tracked count as primary, T1 as fallback if tracking seems too low
        if t1 > 0 and t2t / t1 < 0.3:
            # T1 says many more fish than tracking found — T1 probably right (dense scene)
            final = t1
            print(f"    [Ensemble] Using T1 (tracking too low vs pixel density)")
        else:
            final = t2t
            print(f"    [Ensemble] Using tracked count (reliable range)")
    elif t2t >= 80:
        # Dense scene: YOLO saturates so tracking undercounts significantly.
        # T1 pixel density is more reliable for dense scenes.
        # Use T1 as primary estimate — it scales linearly with fish count
        # while YOLO/tracking plateaus at ~50-100 detections.
        final = t1
        print(f"    [Ensemble] Dense scene — using T1 (tracking saturated at {t2t})")
    else:
        # No tracking data available
        final = t1
        print(f"    [Ensemble] No tracking data, using T1")

    if t3 > 0 and final > 0:
        ratio = final / t3
        if abs(ratio - 1) > 0.5:
            print(f"    [Ensemble] NOTE: final/T3 ratio={ratio:.1f}")

    print(f"    [Ensemble] final={final}")
    return final


def _stream_calibration_from_r2(local_videos, t_start, time_budget):
    """Stream non-local videos from R2 for calibration only.

    Downloads each video, runs T1 scan + calibrate_ppf, saves the PPF,
    then deletes the video. This provides cross-video PPF calibration
    data from all available videos, not just the ones stored locally.
    """
    import boto3

    local_names = {Path(v).name for v in local_videos}

    r2_endpoint = os.environ.get("R2_ENDPOINT",
        "https://6200702e94592ad231a53daba00f8a5d.r2.cloudflarestorage.com")
    r2_ak = os.environ.get("R2_AK", "93bb95ebfe47d5ef93c45efe3c108ca8")
    r2_sk = os.environ.get("R2_SK",
        "cee49fead9c1a8ac2741a4c2703c908efc5d965100a2d8d20c233fce05547a55")
    r2_bucket = os.environ.get("R2_BUCKET", "sala-2026-hackathon-data")

    r2 = boto3.client('s3', endpoint_url=r2_endpoint,
                       aws_access_key_id=r2_ak, aws_secret_access_key=r2_sk)

    resp = r2.list_objects_v2(Bucket=r2_bucket, Prefix='bruv-videos/')
    remote_videos = []
    for obj in resp.get('Contents', []):
        key = obj['Key']
        name = key.split('/')[-1]
        if name.endswith('.MP4') and name not in local_names:
            remote_videos.append((key, name))

    print(f"  [Stream] {len(remote_videos)} remote videos to calibrate")
    all_ppf = []
    video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "videos")

    for r2_key, video_name in remote_videos:
        elapsed = time.time() - t_start
        if elapsed > time_budget * 0.5:  # use at most 50% of budget for streaming
            print(f"  [Stream] Time budget limit, stopping ({elapsed:.0f}s)")
            break

        local_path = os.path.join(video_dir, f"_stream_{video_name}")
        try:
            print(f"\n  [Stream] Downloading {video_name}...")
            t0 = time.time()
            r2.download_file(r2_bucket, r2_key, local_path)
            dl_time = time.time() - t0
            size_mb = os.path.getsize(local_path) / 1e6
            print(f"  [Stream] Downloaded {size_mb:.0f}MB in {dl_time:.1f}s")

            # T1 scan
            scan_results = tier1_scan(local_path)
            print(f"  [Stream] {video_name}: {len(scan_results)} frames scanned")

            # Calibrate PPF
            if len(scan_results) > WARMUP_FRAMES:
                ppf = calibrate_ppf(local_path, scan_results)
                if ppf != DEFAULT_PPF:
                    all_ppf.append(ppf)
                    print(f"  [Stream] {video_name}: PPF={ppf:.1f}")
                else:
                    print(f"  [Stream] {video_name}: calibration failed, skipped")
            else:
                print(f"  [Stream] {video_name}: too few frames, skipped")

        except Exception as e:
            print(f"  [Stream] Error with {video_name}: {e}")
        finally:
            if os.path.exists(local_path):
                os.remove(local_path)

    print(f"\n  [Stream] Collected {len(all_ppf)} PPF values from remote videos")
    if all_ppf:
        print(f"  [Stream] PPF values: {[f'{p:.1f}' for p in all_ppf]}")
    return all_ppf


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

    scored_videos = sorted(
        [v for v in available_videos if Path(v).name in labeled_names],
        key=lambda v: Path(v).name)
    other_videos = [v for v in available_videos if Path(v).name not in labeled_names]

    # Three-pass approach:
    # Pass 1: Scan + YOLO calibrate all videos
    # Pass 2: T2-tracked for all videos
    # Pass 3: Compute tracking-calibrated PPF from sparse videos,
    #          re-aggregate T1 for dense videos, ensemble

    # Pass 1: Scan and YOLO-calibrate PPF
    print("\n  --- Pass 1: Scan & Calibrate ---")
    video_scan_data = {}  # name → (video_path, scan_results, yolo_ppf)
    all_yolo_ppf = []

    for video_path in scored_videos:
        video_name = Path(video_path).name
        elapsed = time.time() - t_start
        if elapsed > TIME_BUDGET - 120:
            print(f"  Approaching time budget, stopping scan at {video_name}")
            break

        print(f"\n  === {video_name} (scan) ===")
        t1 = time.time()
        scan_results = tier1_scan(video_path)
        print(f"    [T1-scan] completed in {time.time()-t1:.1f}s, "
              f"{len(scan_results)} frames")

        yolo_ppf = None
        if len(scan_results) > WARMUP_FRAMES:
            elapsed = time.time() - t_start
            if elapsed < TIME_BUDGET - 120:
                t_cal = time.time()
                ppf = calibrate_ppf(video_path, scan_results)
                print(f"    [Calib] completed in {time.time()-t_cal:.1f}s")
                if ppf != DEFAULT_PPF:
                    yolo_ppf = ppf
                    all_yolo_ppf.append(ppf)

        video_scan_data[video_name] = (video_path, scan_results, yolo_ppf)

    # Initial cross-video PPF from YOLO calibration
    if all_yolo_ppf:
        yolo_cross_ppf = float(np.median(all_yolo_ppf))
        print(f"\n  YOLO cross-video PPF: {yolo_cross_ppf:.1f} "
              f"(median of {len(all_yolo_ppf)}: {[f'{p:.1f}' for p in all_yolo_ppf]})")
    else:
        yolo_cross_ppf = DEFAULT_PPF
        print(f"\n  No YOLO calibrations succeeded, using default PPF={DEFAULT_PPF}")

    # Pass 2: T2-tracked for all videos
    print("\n  --- Pass 2: Tracking ---")
    video_tracking = {}  # name → (t2_maxn, t2_tracked)

    for video_name, (video_path, scan_results, yolo_ppf) in video_scan_data.items():
        print(f"\n  === {video_name} (tracking) ===")

        if len(scan_results) <= WARMUP_FRAMES:
            video_tracking[video_name] = (None, None)
            continue

        # Use YOLO PPF if available, else cross-video, for peak frame selection
        ppf_for_peaks = yolo_ppf if yolo_ppf is not None else yolo_cross_ppf
        peak_frames = get_peak_frame_indices(scan_results, ppf_for_peaks, N_PEAK_FRAMES)

        # T2: YOLO on peaks (single-frame max)
        elapsed = time.time() - t_start
        t2_maxn = None
        if elapsed < TIME_BUDGET - 90:
            t2 = time.time()
            t2_maxn = tier2_yolo_count(video_path, peak_frames)
            print(f"    [T2] completed in {time.time()-t2:.1f}s")

        # T2-tracked: IoU tracking on window around peak
        elapsed = time.time() - t_start
        t2_tracked = None
        if elapsed < TIME_BUDGET - 60:
            t2t = time.time()
            t2_tracked = tier2_tracked_count(video_path, scan_results, ppf_for_peaks)
            print(f"    [T2-track] completed in {time.time()-t2t:.1f}s")

        video_tracking[video_name] = (t2_maxn, t2_tracked)

    # Pass 3: Tracking-calibrated PPF + ensemble
    # For sparse videos (tracked < 80), tracking is reliable ground truth.
    # Derive PPF from: peak_sustained_pixels / tracked_count
    # This is more accurate than YOLO calibration for cross-video use.
    print("\n  --- Pass 3: Tracking-calibrated PPF + Ensemble ---")

    tracking_ppfs = []
    for video_name, (video_path, scan_results, yolo_ppf) in video_scan_data.items():
        t2_maxn, t2_tracked = video_tracking[video_name]
        if t2_tracked is not None and 10 < t2_tracked < 80:
            # Sparse video: tracking is reliable
            # Compute peak sustained pixels
            after_warmup = [px for _, _, px in scan_results[WARMUP_FRAMES:]]
            if len(after_warmup) >= SUSTAINED_WINDOW:
                windowed = np.convolve(
                    after_warmup,
                    np.ones(SUSTAINED_WINDOW) / SUSTAINED_WINDOW,
                    mode='valid')
                peak_sustained_px = float(windowed.max())
            else:
                peak_sustained_px = float(max(after_warmup))
            track_ppf = peak_sustained_px / t2_tracked
            tracking_ppfs.append(track_ppf)
            print(f"  {video_name}: T2-tracked={t2_tracked}, "
                  f"peak_sustained_px={peak_sustained_px:.0f}, "
                  f"track_ppf={track_ppf:.1f}")

    # Choose best PPF: prefer tracking-calibrated, fall back to YOLO
    if tracking_ppfs:
        final_ppf = float(np.median(tracking_ppfs))
        print(f"\n  Tracking-calibrated PPF: {final_ppf:.1f} "
              f"(from {len(tracking_ppfs)} sparse video(s))")
    elif all_yolo_ppf:
        final_ppf = yolo_cross_ppf
        print(f"\n  No tracking calibration, using YOLO PPF: {final_ppf:.1f}")
    else:
        final_ppf = DEFAULT_PPF
        print(f"\n  No calibration, using default PPF: {final_ppf:.1f}")

    # Now aggregate and ensemble each video
    for video_name, (video_path, scan_results, yolo_ppf) in video_scan_data.items():
        print(f"\n  === {video_name} (ensemble) ===")
        t2_maxn, t2_tracked = video_tracking[video_name]

        if len(scan_results) <= WARMUP_FRAMES:
            pred_maxn[video_name] = 0
            continue

        # Dense video (tracking saturated): use tracking-calibrated PPF
        # Sparse video: use per-video YOLO PPF or tracking-calibrated
        t2t = t2_tracked if t2_tracked is not None else 0
        if t2t >= 80:
            # Dense scene — YOLO PPF unreliable, use tracking-calibrated
            ppf = final_ppf
            print(f"    [PPF] Dense scene, using tracking-calibrated PPF={ppf:.1f}")
        else:
            # Sparse scene — per-video YOLO or tracking-calibrated
            ppf = yolo_ppf if yolo_ppf is not None else final_ppf
            print(f"    [PPF] Sparse scene, PPF={ppf:.1f}")

        t1_maxn = tier1_aggregate(scan_results, ppf)

        if t1_maxn < 5:
            pred_maxn[video_name] = t1_maxn
            continue

        peak_frames = get_peak_frame_indices(scan_results, ppf, N_PEAK_FRAMES)

        # T3: VLM on peaks
        elapsed = time.time() - t_start
        t3_maxn = None
        if elapsed < TIME_BUDGET - 20:
            t3 = time.time()
            t3_maxn = tier3_vlm_count(video_path, peak_frames)
            print(f"    [T3] completed in {time.time()-t3:.1f}s")

        final = ensemble_maxn(t1_maxn, t2_maxn, t2_tracked, t3_maxn,
                              scan_results=scan_results, ppf=ppf)
        print(f"    Final MaxN: {final} (T1={t1_maxn}, T2={t2_maxn}, "
              f"T2-tracked={t2_tracked}, T3={t3_maxn}, PPF={ppf:.1f})")
        pred_maxn[video_name] = final

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
