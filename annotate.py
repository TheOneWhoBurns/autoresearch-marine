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


def download_video(video_name, video_dir):
    """Download a single video from R2 if not already present."""
    import boto3

    dest = os.path.join(video_dir, video_name)
    if os.path.exists(dest):
        print(f"  {video_name} already on disk")
        return dest

    r2_endpoint = os.environ.get("R2_ENDPOINT")
    r2_ak = os.environ.get("R2_AK")
    r2_sk = os.environ.get("R2_SK")
    r2_bucket = os.environ.get("R2_BUCKET")

    if not all([r2_endpoint, r2_ak, r2_sk, r2_bucket]):
        print(f"  R2 credentials not set, skipping {video_name}")
        return None

    client = boto3.client('s3',
        endpoint_url=r2_endpoint,
        aws_access_key_id=r2_ak,
        aws_secret_access_key=r2_sk,
    )
    key = f"bruv-videos/{video_name}"
    print(f"  Downloading {video_name} from R2...")
    t = time.time()
    try:
        client.download_file(r2_bucket, key, dest)
        sz = os.path.getsize(dest) / 1e9
        print(f"  Downloaded {video_name}: {sz:.2f} GB in {time.time()-t:.0f}s")
        return dest
    except Exception as e:
        print(f"  Download error: {e}")
        return None


def list_r2_videos():
    """List all video filenames on R2."""
    import boto3

    r2_endpoint = os.environ.get("R2_ENDPOINT")
    r2_ak = os.environ.get("R2_AK")
    r2_sk = os.environ.get("R2_SK")
    r2_bucket = os.environ.get("R2_BUCKET")

    if not all([r2_endpoint, r2_ak, r2_sk, r2_bucket]):
        return []

    client = boto3.client('s3',
        endpoint_url=r2_endpoint,
        aws_access_key_id=r2_ak,
        aws_secret_access_key=r2_sk,
    )
    videos = []
    paginator = client.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=r2_bucket, Prefix='bruv-videos/'):
        for obj in page.get('Contents', []):
            videos.append(obj['Key'].split('/')[-1])
    return sorted(videos)


def upload_to_s3(fpath):
    """Upload a single file to S3."""
    import boto3
    bucket = os.environ.get("BUCKET", "autoresearch-marine-data")
    track = os.environ.get("TRACK", "bruv")
    key = f"{track}/results/annotated/{Path(fpath).name}"
    try:
        boto3.client('s3').upload_file(fpath, bucket, key)
        print(f"  Uploaded to s3://{bucket}/{key}")
    except Exception as e:
        print(f"  Upload failed: {e}")


def main():
    from ultralytics import YOLO

    video_dir = "data/videos"
    output_dir = "data/results/annotated"
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Determine which videos to process
    if "--all" in sys.argv:
        # List from R2, or from local dir
        video_names = list_r2_videos()
        if not video_names:
            video_names = sorted(f.name for f in Path(video_dir).glob("*.MP4"))
    elif len(sys.argv) > 1:
        video_names = [a for a in sys.argv[1:] if not a.startswith("--")]
    else:
        video_names = sorted(f.name for f in Path(video_dir).glob("*.MP4"))

    if not video_names:
        print("No videos found")
        return

    print(f"=== BRUV Video Annotator ===")
    print(f"Videos to process: {len(video_names)}")
    print(f"Output: {output_dir}")
    print(f"Frame step: {FRAME_STEP} (output {OUTPUT_FPS}fps)")

    # Load YOLO once
    print("\nLoading YOLOv8n...")
    model = YOLO("yolov8n.pt")

    t_start = time.time()
    n_done = 0

    # Process one video at a time: download → annotate → upload → delete source
    for video_name in video_names:
        print(f"\n--- [{n_done+1}/{len(video_names)}] {video_name} ---")

        # Download this video
        video_path = download_video(video_name, video_dir)
        if video_path is None:
            continue

        # Annotate
        out_path = annotate_video(video_path, output_dir, model)
        if out_path is None:
            continue

        # Upload annotated video immediately
        upload_to_s3(out_path)
        n_done += 1

        # Delete source video to save disk space (keep annotated)
        try:
            os.remove(video_path)
            print(f"  Removed source: {video_name}")
        except OSError:
            pass

    elapsed = time.time() - t_start
    print(f"\n=== Complete ===")
    print(f"Processed {n_done}/{len(video_names)} videos in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
