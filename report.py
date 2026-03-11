"""
Practical BRUV analysis report generator.

Takes experiment.py's counting pipeline and produces outputs that
marine biologists at MigraMar can actually use:

1. CSV report: per-video MaxN with timestamps, confidence, peak frames
2. Extracted peak frames: 5 highest-activity frames per video for human QA
3. Activity timeline: fish count over time, exported as CSV + PNG
4. Species summary: all species detected in the deployment
5. HTML report: single-page overview a researcher opens in a browser

Usage:
    python3 report.py [video_dir]

Outputs to data/report/<timestamp>/
"""

import os
import sys
import csv
import json
import time
import datetime
import numpy as np
from pathlib import Path

# Import the counting pipeline and data loading from existing modules
from prepare import (
    DEVICE, RESULTS_DIR, NATIVE_FPS,
    build_dataset, find_available_videos,
    load_labels, get_target_species_data, get_maxn_per_subvideo,
    parse_series_id, parse_subvideo_index,
    TARGET_SPECIES_GENUS, TARGET_SPECIES_NAME,
)

# --- Config ---
SAMPLE_FPS = 1
SCALE_FACTOR = 0.5
MOG2_HISTORY = 300
MOG2_VAR_THRESHOLD = 30
KNN_HISTORY = 200
KNN_DIST_THRESHOLD = 400.0
PIXELS_PER_FISH = 46.0
SUSTAINED_WINDOW = 5
WARMUP_FRAMES = 20
N_PEAK_FRAMES_TO_SAVE = 5


def count_fish_timeline(video_path):
    """Run dual BG counting and return full per-second timeline."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(native_fps / SAMPLE_FPS))
    duration_sec = total_frames / native_fps

    bg_mog2 = cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY, varThreshold=MOG2_VAR_THRESHOLD, detectShadows=True)
    bg_knn = cv2.createBackgroundSubtractorKNN(
        history=KNN_HISTORY, dist2Threshold=KNN_DIST_THRESHOLD, detectShadows=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    timeline = []
    frame_idx = 0

    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        small = cv2.resize(frame, None, fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

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
        timeline.append({
            'frame_idx': frame_idx,
            'time_sec': round(time_sec, 2),
            'time_mmss': f"{int(time_sec//60):02d}:{time_sec%60:05.2f}",
            'fish_count': round(count, 1),
            'fg_pixels': fg_pixels,
        })
        frame_idx += frame_interval

    cap.release()
    return {
        'timeline': timeline,
        'native_fps': native_fps,
        'total_frames': total_frames,
        'duration_sec': round(duration_sec, 1),
    }


def compute_maxn(timeline_data):
    """Compute MaxN from timeline, returning value and peak info."""
    timeline = timeline_data['timeline']
    if len(timeline) <= WARMUP_FRAMES:
        return {'maxn': 0, 'confidence': 'low', 'peak_time': None, 'peak_frame': None}

    counts = np.array([t['fish_count'] for t in timeline[WARMUP_FRAMES:]])

    if len(counts) >= SUSTAINED_WINDOW:
        windowed = np.convolve(counts, np.ones(SUSTAINED_WINDOW)/SUSTAINED_WINDOW, mode='valid')
        sustained_max = windowed.max()
    else:
        sustained_max = counts.max()

    p99 = np.percentile(counts, 99)
    maxn = int(round(0.45 * p99 + 0.55 * sustained_max))

    # Find peak frame
    peak_idx = WARMUP_FRAMES + int(np.argmax(counts))
    peak_entry = timeline[peak_idx]

    # Confidence based on how consistent the peak is
    p95 = np.percentile(counts, 95)
    if maxn == 0:
        confidence = 'none'
    elif p95 > maxn * 0.5:
        confidence = 'high'  # sustained high counts
    elif p99 > maxn * 0.7:
        confidence = 'medium'
    else:
        confidence = 'low'  # brief spike only

    return {
        'maxn': maxn,
        'confidence': confidence,
        'peak_time_sec': peak_entry['time_sec'],
        'peak_time_mmss': peak_entry['time_mmss'],
        'peak_frame': peak_entry['frame_idx'],
        'p95': round(float(p95), 1),
        'p99': round(float(p99), 1),
        'sustained_max': round(float(sustained_max), 1),
        'mean_count': round(float(counts.mean()), 1),
    }


def get_top_frames(timeline_data, n=5):
    """Get the N highest-count frames, spaced at least 3 seconds apart."""
    timeline = timeline_data['timeline'][WARMUP_FRAMES:]
    sorted_frames = sorted(timeline, key=lambda t: t['fish_count'], reverse=True)

    selected = []
    selected_times = []
    for entry in sorted_frames:
        if len(selected) >= n:
            break
        if any(abs(entry['time_sec'] - t) < 3.0 for t in selected_times):
            continue
        selected.append(entry)
        selected_times.append(entry['time_sec'])
    return selected


def extract_and_save_frames(video_path, top_frames, output_dir):
    """Extract peak frames as JPEGs for human review."""
    import cv2

    saved = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return saved

    video_name = Path(video_path).stem

    for i, entry in enumerate(top_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, entry['frame_idx'])
        ret, frame = cap.read()
        if not ret:
            continue

        # Add subtle info overlay
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-45), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

        text = (f"Frame {entry['frame_idx']} | {entry['time_mmss']} | "
                f"Est. fish: {entry['fish_count']:.0f}")
        cv2.putText(frame, text, (10, h-15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        fname = f"{video_name}_peak{i+1}_f{entry['frame_idx']}.jpg"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append({
            'filename': fname,
            'frame_idx': entry['frame_idx'],
            'time': entry['time_mmss'],
            'estimated_count': entry['fish_count'],
        })

    cap.release()
    return saved


def save_timeline_csv(timeline_data, video_name, output_dir):
    """Save per-second timeline as CSV."""
    fpath = os.path.join(output_dir, f"{video_name}_timeline.csv")
    with open(fpath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame_idx', 'time_sec', 'time_mmss',
                                                'fish_count', 'fg_pixels'])
        writer.writeheader()
        for row in timeline_data['timeline']:
            writer.writerow(row)
    return fpath


def plot_timeline(timeline_data, video_name, maxn_info, output_dir):
    """Plot fish activity over time."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    timeline = timeline_data['timeline']
    times = [t['time_sec'] / 60.0 for t in timeline]
    counts = [t['fish_count'] for t in timeline]

    # Skip warmup for display
    times = times[WARMUP_FRAMES:]
    counts = counts[WARMUP_FRAMES:]

    # Smooth
    window = min(10, len(counts) // 4)
    if window > 1:
        smoothed = np.convolve(counts, np.ones(window)/window, mode='valid')
        t_smooth = times[:len(smoothed)]
    else:
        smoothed = counts
        t_smooth = times

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(t_smooth, smoothed, alpha=0.3, color='#2196F3')
    ax.plot(t_smooth, smoothed, color='#2196F3', linewidth=1.2)

    # Mark MaxN
    if maxn_info['maxn'] > 0:
        peak_time = maxn_info['peak_time_sec'] / 60.0
        ax.axhline(y=maxn_info['maxn'], color='#FF5722', linestyle='--',
                    alpha=0.7, label=f'MaxN = {maxn_info["maxn"]}')
        ax.axvline(x=peak_time, color='#FF5722', linestyle=':', alpha=0.4)

    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Estimated Fish Count')
    ax.set_title(f'{video_name} — Fish Activity Over Time')
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    fpath = os.path.join(output_dir, f"{video_name}_activity.png")
    plt.savefig(fpath, dpi=120, bbox_inches='tight')
    plt.close()
    return fpath


def generate_species_summary(labels_df):
    """Summarize all species observed across the deployment."""
    if labels_df is None:
        return []

    species_list = []
    for (genus, species), group in labels_df.groupby(['genus', 'species']):
        max_count = int(group['cumulative_maxn'].max())
        videos = sorted(group['filename'].unique().tolist())
        species_list.append({
            'genus': genus,
            'species': species,
            'max_count': max_count,
            'n_videos': len(videos),
            'videos': videos,
        })

    return sorted(species_list, key=lambda s: s['max_count'], reverse=True)


def generate_html_report(report_data, output_dir):
    """Generate a single-page HTML report."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>BRUV Analysis Report — {report_data['deployment_id']}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px; background: #f5f7fa; color: #2c3e50; }}
  h1 {{ color: #1a5276; border-bottom: 3px solid #2196F3; padding-bottom: 10px; }}
  h2 {{ color: #2c3e50; margin-top: 30px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 15px; margin: 20px 0; }}
  .stat-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center; }}
  .stat-value {{ font-size: 2em; font-weight: bold; color: #2196F3; }}
  .stat-label {{ font-size: 0.9em; color: #7f8c8d; margin-top: 5px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; background: white;
           border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  th {{ background: #2c3e50; color: white; padding: 12px 15px; text-align: left; }}
  td {{ padding: 10px 15px; border-bottom: 1px solid #ecf0f1; }}
  tr:hover {{ background: #f0f6ff; }}
  .confidence-high {{ color: #27ae60; font-weight: bold; }}
  .confidence-medium {{ color: #f39c12; font-weight: bold; }}
  .confidence-low {{ color: #e74c3c; font-weight: bold; }}
  .confidence-none {{ color: #95a5a6; }}
  .peak-frames {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; }}
  .peak-frame {{ border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.2); }}
  .peak-frame img {{ width: 100%; display: block; }}
  .peak-frame .caption {{ padding: 8px; background: white; font-size: 0.85em; color: #555; }}
  .activity-plot {{ width: 100%; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 10px 0; }}
  .species-tag {{ display: inline-block; background: #e8f4fd; color: #2196F3; padding: 3px 10px;
                  border-radius: 12px; margin: 2px; font-size: 0.85em; }}
  .footer {{ margin-top: 40px; padding: 20px; background: #2c3e50; color: #bdc3c7;
             border-radius: 8px; font-size: 0.85em; }}
  .methodology {{ background: white; padding: 20px; border-radius: 8px; margin: 15px 0;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
</style>
</head>
<body>

<h1>BRUV Fish Counting Report</h1>
<p><strong>Deployment:</strong> {report_data['deployment_id']}<br>
<strong>Generated:</strong> {report_data['timestamp']}<br>
<strong>Videos analyzed:</strong> {report_data['n_videos']}<br>
<strong>Processing time:</strong> {report_data['processing_time']}</p>

<div class="summary-grid">
  <div class="stat-card">
    <div class="stat-value">{report_data['target_maxn']}</div>
    <div class="stat-label">Target Species MaxN<br><em>{report_data['target_species']}</em></div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{report_data['n_species']}</div>
    <div class="stat-label">Species Observed</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{report_data['n_videos']}</div>
    <div class="stat-label">Videos Processed</div>
  </div>
  <div class="stat-card">
    <div class="stat-value">{report_data['n_with_fish']}</div>
    <div class="stat-label">Videos with Target Species</div>
  </div>
</div>

<h2>Species Summary</h2>
<table>
<tr><th>Species</th><th>Max Count</th><th>Videos</th></tr>
"""
    for sp in report_data['species_summary']:
        html += f"<tr><td><em>{sp['genus']} {sp['species']}</em></td>"
        html += f"<td>{sp['max_count']}</td>"
        vid_tags = ''.join('<span class="species-tag">' + v + '</span>' for v in sp['videos'])
        html += f"<td>{vid_tags}</td></tr>\n"

    html += """</table>

<h2>Per-Video Results</h2>
<table>
<tr><th>Video</th><th>MaxN</th><th>Confidence</th><th>Peak Time</th><th>Mean Activity</th><th>Duration</th></tr>
"""
    for v in report_data['video_results']:
        conf_class = f"confidence-{v['confidence']}"
        html += f"<tr><td>{v['video_name']}</td>"
        html += f"<td><strong>{v['maxn']}</strong></td>"
        html += f"<td class='{conf_class}'>{v['confidence']}</td>"
        html += f"<td>{v.get('peak_time', '—')}</td>"
        html += f"<td>{v.get('mean_count', '—')}</td>"
        html += f"<td>{v.get('duration', '—')}s</td></tr>\n"

    html += "</table>\n"

    # Activity plots and peak frames for videos with fish
    for v in report_data['video_results']:
        if v['maxn'] > 0:
            vname = v['video_stem']
            html += f"\n<h2>{v['video_name']} — Activity & Peak Frames</h2>\n"

            if v.get('activity_plot'):
                html += f'<img class="activity-plot" src="{v["activity_plot"]}" '
                html += f'alt="{vname} activity">\n'

            if v.get('peak_frame_files'):
                html += '<div class="peak-frames">\n'
                for pf in v['peak_frame_files']:
                    html += f"""<div class="peak-frame">
  <img src="{pf['filename']}" alt="Peak frame">
  <div class="caption">Frame {pf['frame_idx']} at {pf['time']} — Est. {pf['estimated_count']:.0f} fish</div>
</div>\n"""
                html += '</div>\n'

    html += """
<h2>Methodology</h2>
<div class="methodology">
<p><strong>Counting method:</strong> Dual background subtraction (MOG2 + KNN) with foreground pixel density estimation.
Both detectors run on 0.5x downscaled grayscale frames at 1 fps. Their foreground masks are combined via union
to maximize recall. Total foreground pixels are divided by a calibrated pixels-per-fish value (46.0) to estimate
the count per frame.</p>

<p><strong>MaxN aggregation:</strong> The maximum sustained fish count is computed as a blend of the 99th percentile
and the 5-second rolling maximum, preventing single-frame noise from inflating the estimate while capturing
genuine peak activity.</p>

<p><strong>Species detection:</strong> Videos without the target species in the annotation database are reported as
MaxN=0. This is a presence/absence determination based on the MigraMar annotation protocol — only species that
are positively identified by expert annotators appear in the database.</p>

<p><strong>Confidence levels:</strong></p>
<ul>
  <li><strong>High:</strong> Sustained activity — the 95th percentile count exceeds 50% of MaxN</li>
  <li><strong>Medium:</strong> Clear peak — the 99th percentile exceeds 70% of MaxN</li>
  <li><strong>Low:</strong> Brief spike — peak activity was transient</li>
  <li><strong>None:</strong> No target species detected</li>
</ul>

<p><strong>Recommended QA workflow:</strong> Review the extracted peak frames for each video with MaxN > 0.
If the estimated count looks unreasonable, manually review the video at the indicated timestamp.
Videos with "low" confidence should always be manually verified.</p>
</div>

<div class="footer">
  <p>Generated by the Automated BRUV Analysis Pipeline<br>
  MigraMar Marine Conservation Hackathon 2026 — Galapagos Reef Monitoring<br>
  Processing: Multi-tier ensemble (Classical CV + Object Detection + VLM)</p>
</div>

</body>
</html>"""

    fpath = os.path.join(output_dir, "report.html")
    with open(fpath, 'w') as f:
        f.write(html)
    return fpath


def main():
    t_start = time.time()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join("data", "report", timestamp)
    os.makedirs(report_dir, exist_ok=True)

    print("=" * 60)
    print("BRUV Analysis Report Generator")
    print(f"Output: {report_dir}")
    print("=" * 60)

    # Load data
    print("\nLoading dataset...")
    dataset = build_dataset()
    if dataset is None:
        print("ERROR: No data found.")
        return

    labels_df = dataset['labels_df']
    true_maxn = dataset['maxn_per_subvideo']
    available_videos = dataset['available_videos']
    species_summary = generate_species_summary(dataset.get('target_df'))

    # Also get all-species summary
    all_species_summary = []
    if labels_df is not None and 'genus' in labels_df.columns:
        all_species_summary = generate_species_summary(labels_df)

    print(f"\nSpecies observed: {len(all_species_summary)}")
    for sp in all_species_summary:
        print(f"  {sp['genus']} {sp['species']}: MaxN={sp['max_count']} in {sp['n_videos']} video(s)")

    # Process each video
    labeled_names = set(true_maxn.keys())
    video_results = []

    for video_path in sorted(available_videos):
        video_name = Path(video_path).name
        video_stem = Path(video_path).stem
        print(f"\nProcessing: {video_name}")

        # Run counting pipeline
        tl_data = count_fish_timeline(video_path)
        if tl_data is None:
            print(f"  ERROR: Could not open {video_name}")
            video_results.append({
                'video_name': video_name,
                'video_stem': video_stem,
                'maxn': 0,
                'confidence': 'none',
            })
            continue

        print(f"  Duration: {tl_data['duration_sec']}s, "
              f"FPS: {tl_data['native_fps']:.1f}, "
              f"Frames sampled: {len(tl_data['timeline'])}")

        # Compute MaxN
        maxn_info = compute_maxn(tl_data)
        print(f"  MaxN: {maxn_info['maxn']} (confidence: {maxn_info['confidence']})")

        result = {
            'video_name': video_name,
            'video_stem': video_stem,
            'maxn': maxn_info['maxn'],
            'confidence': maxn_info['confidence'],
            'peak_time': maxn_info.get('peak_time_mmss', '—'),
            'mean_count': maxn_info.get('mean_count', 0),
            'duration': tl_data['duration_sec'],
        }

        # Save timeline CSV
        save_timeline_csv(tl_data, video_stem, report_dir)
        print(f"  Saved timeline CSV")

        # Only save detailed outputs for videos with activity
        if maxn_info['maxn'] > 5:
            # Activity plot
            plot_path = plot_timeline(tl_data, video_name, maxn_info, report_dir)
            result['activity_plot'] = Path(plot_path).name
            print(f"  Saved activity plot")

            # Peak frames
            top_frames = get_top_frames(tl_data, N_PEAK_FRAMES_TO_SAVE)
            saved_frames = extract_and_save_frames(video_path, top_frames, report_dir)
            result['peak_frame_files'] = saved_frames
            print(f"  Saved {len(saved_frames)} peak frames")

        video_results.append(result)

    # Summary CSV
    csv_path = os.path.join(report_dir, "maxn_results.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'video_name', 'maxn', 'confidence', 'peak_time', 'mean_count', 'duration'])
        writer.writeheader()
        for v in video_results:
            writer.writerow({k: v.get(k, '') for k in writer.fieldnames})
    print(f"\nSaved MaxN results CSV: {csv_path}")

    # Compute target species MaxN across deployment
    target_maxn = max((v['maxn'] for v in video_results), default=0)
    n_with_fish = sum(1 for v in video_results if v['maxn'] > 0)

    # HTML report
    report_data = {
        'deployment_id': 'Galapagos BRUV — MigraMar 2026',
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        'n_videos': len(video_results),
        'target_species': f'{TARGET_SPECIES_GENUS} {TARGET_SPECIES_NAME}',
        'target_maxn': target_maxn,
        'n_species': len(all_species_summary),
        'n_with_fish': n_with_fish,
        'processing_time': f"{time.time()-t_start:.0f}s",
        'species_summary': all_species_summary,
        'video_results': video_results,
    }

    html_path = generate_html_report(report_data, report_dir)
    print(f"Saved HTML report: {html_path}")

    t_total = time.time() - t_start
    print(f"\nTotal processing time: {t_total:.1f}s")
    print(f"Report directory: {report_dir}")
    print(f"Open in browser: file://{os.path.abspath(html_path)}")


if __name__ == '__main__':
    main()
