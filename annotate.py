"""
Generate annotated videos with YOLO bounding boxes and BG subtraction
contours drawn on every frame. Outputs one .mp4 per input video.

Runs on AWS where videos are available. Uploads results to S3.

Usage: python3 annotate.py [--videos LGH020002.MP4 LGH040001.MP4]
       python3 annotate.py --all
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# --- Config (matches experiment.py) ---
SCALE_FACTOR = 0.5
MOG2_HISTORY = 300
MOG2_VAR_THRESHOLD = 30
KNN_HISTORY = 200
KNN_DIST_THRESHOLD = 400.0
MORPH_KERNEL_SIZE = 3
BLUR_SIZE = 5
YOLO_CONF = 0.01
YOLO_IOU = 0.3
YOLO_IMG_SIZE = 640  # smaller for faster CPU inference
YOLO_FISH_CLASSES = {"kite", "bird"}

# Annotated video output FPS — process every Nth frame from original
# 30 = 1fps from 30fps source (matches experiment scan rate, fast on CPU)
FRAME_STEP = 30
OUTPUT_FPS = 1


def annotate_video(video_path, output_dir, model):
    """Process one video: run YOLO + BG subtraction, draw boxes, write .mp4."""
    import cv2

    video_name = Path(video_path).stem
    out_path = os.path.join(output_dir, f"{video_name}_annotated.mp4")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return None

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / native_fps

    print(f"\n  === {video_name} ===")
    print(f"  {width}x{height} @ {native_fps:.0f}fps, "
          f"{total_frames} frames, {duration:.0f}s")

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, OUTPUT_FPS, (width, height))

    # BG subtractors (run on downscaled for speed, but draw on full res)
    bg_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD,
        detectShadows=True)
    bg_knn = cv2.createBackgroundSubtractorKNN(
        history=KNN_HISTORY, dist2Threshold=KNN_DIST_THRESHOLD,
        detectShadows=True)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))

    frame_idx = 0
    n_written = 0
    t_start = time.time()

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # --- BG subtraction at half scale ---
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

        # Scale contours back to full resolution
        fg_full = cv2.resize(fg_clean * 255, (width, height),
                             interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(
            fg_full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        fg_pixels = np.count_nonzero(fg_clean)

        # --- YOLO detection ---
        results = model(frame, conf=YOLO_CONF, iou=YOLO_IOU,
                        imgsz=YOLO_IMG_SIZE, verbose=False)

        yolo_count = 0
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for i, cls_id in enumerate(boxes.cls.cpu().numpy().astype(int)):
                    cls_name = model.names[int(cls_id)]
                    if cls_name in YOLO_FISH_CLASSES:
                        yolo_count += 1
                        # Draw bounding box
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                        conf = float(boxes.conf[i])
                        cv2.rectangle(frame, (x1, y1), (x2, y2),
                                      (0, 255, 0), 2)
                        label = f"{conf:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    (0, 255, 0), 1, cv2.LINE_AA)

        # --- Draw BG subtraction contours (cyan, thin) ---
        # Only draw contours with area > 20px to skip noise
        for cnt in contours:
            if cv2.contourArea(cnt) > 20:
                cv2.drawContours(frame, [cnt], -1, (255, 255, 0), 1)

        # --- HUD overlay ---
        time_sec = frame_idx / native_fps
        mmss = f"{int(time_sec//60):02d}:{time_sec%60:05.2f}"

        # Semi-transparent bar at top
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        hud = (f"{video_name} | {mmss} | "
               f"YOLO: {yolo_count} fish | "
               f"BG: {fg_pixels}px | "
               f"f{frame_idx}/{total_frames}")
        cv2.putText(frame, hud, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame)
        n_written += 1
        frame_idx += FRAME_STEP

        # Progress every 500 output frames
        if n_written % 500 == 0:
            elapsed = time.time() - t_start
            pct = 100 * frame_idx / total_frames
            fps = n_written / elapsed if elapsed > 0 else 0
            eta = (total_frames - frame_idx) / FRAME_STEP / fps if fps > 0 else 0
            print(f"  [{pct:.0f}%] {n_written} frames written, "
                  f"{fps:.1f} fps, ETA {eta:.0f}s")

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"  Done: {n_written} frames in {elapsed:.1f}s "
          f"({size_mb:.1f} MB) → {out_path}")
    return out_path


def main():
    import cv2
    from ultralytics import YOLO

    video_dir = "data/videos"
    output_dir = "data/results/annotated"
    os.makedirs(output_dir, exist_ok=True)

    # Parse args
    if "--all" in sys.argv:
        videos = sorted(Path(video_dir).glob("*.MP4"))
    elif len(sys.argv) > 1 and sys.argv[1] != "--all":
        # Specific videos
        names = [a for a in sys.argv[1:] if not a.startswith("--")]
        videos = [Path(video_dir) / n for n in names]
    else:
        # Default: all videos
        videos = sorted(Path(video_dir).glob("*.MP4"))

    if not videos:
        print("No videos found")
        return

    print(f"=== BRUV Video Annotator ===")
    print(f"Videos: {len(videos)}")
    print(f"Output: {output_dir}")
    print(f"Frame step: {FRAME_STEP} (output {OUTPUT_FPS}fps)")

    # Load YOLO once
    print("\nLoading YOLOv8n...")
    model = YOLO("yolov8n.pt")

    t_start = time.time()
    outputs = []

    for vpath in videos:
        if not vpath.exists():
            print(f"  SKIP: {vpath} not found")
            continue
        result = annotate_video(str(vpath), output_dir, model)
        if result:
            outputs.append(result)

    elapsed = time.time() - t_start
    print(f"\n=== Complete ===")
    print(f"Processed {len(outputs)} videos in {elapsed:.0f}s")
    print(f"Output directory: {output_dir}")

    # Upload to S3 if AWS credentials available
    try:
        import boto3
        bucket = os.environ.get("BUCKET", "autoresearch-marine-data")
        track = os.environ.get("TRACK", "bruv")
        print(f"\nUploading to s3://{bucket}/{track}/results/annotated/...")
        for fpath in outputs:
            key = f"{track}/results/annotated/{Path(fpath).name}"
            boto3.client('s3').upload_file(fpath, bucket, key)
            print(f"  Uploaded: {Path(fpath).name}")
        print("Upload complete.")
    except Exception as e:
        print(f"S3 upload skipped: {e}")


if __name__ == "__main__":
    main()
