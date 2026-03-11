# autoresearch-marine: BRUV Fish Counting

Autonomous BRUV (Baited Remote Underwater Video) fish counting research. The
agent iterates on `experiment.py` to count *Caranx caballus* (green jack) in
underwater video frames, progressing through hackathon tiers naturally.

## Setup

1. **Run tag**: propose a tag (e.g. `mar11`). Branch: `autoresearch/<tag>`.
2. **Create branch**: `git checkout -b autoresearch/<tag>`.
3. **Read files**: `prepare.py` (fixed), `experiment.py` (you edit), this file.
4. **Verify data**: `data/videos/` must contain at least one MP4 sub-video and
   `data/labels/` must contain `CumulativeMaxN.csv`. If not, tell human.
5. **Init results.tsv**: header row only. Baseline recorded after first run.
6. **Go**.

## Context

Underwater BRUV footage from MigraMar deployments. Two camera stations with
18 sub-videos total (~65 GB). Each sub-video is ~11:47 long at 30fps.

### Target Species

| Species | Common Name | Video 1 MaxN | Video 2 MaxN |
|---------|------------|-------------|-------------|
| Caranx caballus | Green jack | 251 | 52 |

Other species present: almaco jack, rainbow runner, pilotfish, silky shark,
unicorn filefish, mahi-mahi. Primary task: count green jack MaxN per frame.

### Data Sources

**Labels (Kaggle CSV):**
- `CumulativeMaxN.csv` — frame-level species counts with timestamps
- `TimeFirstSeen.csv` — species first appearance times
- Columns: filename, frame_number, time_mins, taxonomic classification, count

**Videos (Cloudflare R2):**
- 18 sub-videos across 2 deployment series
- Each ~11:47 at 30fps (~4 GB per file)
- Timing: `sub_video_index = floor(time_mins / 11.783) + 1`

### BRUV Apparatus

The bait arm is stationary and visible in all frames. It must be masked or
excluded from detection pipelines to avoid false positives.

## Platform

Linux with GPU if available (CUDA preferred), CPU fallback. Available packages:
numpy, scipy, scikit-learn, opencv-python, matplotlib, torch, torchvision,
ultralytics (YOLOv8), pandas, Pillow.

No video download during experiments — work with pre-downloaded frames or
sub-videos in `data/videos/`.

## The Hackathon Tiers

### Tier 1: Frame Sampling + Background Subtraction (starting point)
- Sample frames at 1-2 fps from 30fps source video
- MOG2 background subtraction to detect moving objects
- Contour detection and counting on foreground masks
- Simple blob counting as MaxN baseline
- **Discovery goal**: establish motion-based count, identify high-activity frames

### Tier 2: Pre-trained Object Detection
- YOLOv8 pre-trained on COCO (fish ~ "bird" or general animal class)
- Fine-tune YOLOv8 on manually annotated BRUV frames
- Multi-object tracking (ByteTrack / BoT-SORT) across frames
- Temporal aggregation: max detections in sliding window
- **Discovery goal**: species-specific detection, MaxN estimation

### Tier 3: Crowd Counting + Advanced Methods
- Density map regression (CSRNet, CAN, or similar)
- Vision-language model zero-shot counting (e.g. CLIP + counting head)
- Intelligent frame selection: predict which frames have peak abundance
- Ensemble methods combining Tier 1+2+3 signals
- **Discovery goal**: handle dense schools (251 fish), surpass expert counts

### Cross-Tier Combinations
- Tier 1 motion signal identifies "interesting" frames → Tier 2 detects on those
- Tier 2 detections bootstrap pseudo-labels → Tier 3 trains density estimator
- Motion peaks from Tier 1 validate Tier 2/3 temporal MaxN estimates
- Background subtraction masks improve Tier 2 detection precision

## Hackathon Judging (for context)
- **Originality & Innovation (30%)**: unique approaches, surprising discoveries
- **Technical Execution (30%)**: code quality, methodology complexity
- **Impact & Relevance (25%)**: practical applicability for conservation
- **Presentation (15%)**: (we handle this separately)

## Experimentation Rules

**What you CAN do:**
- Modify `experiment.py` — the ONLY file you edit. Everything is fair game:
  detection method, counting strategy, model architecture, anything.
- Use PyTorch with CUDA/CPU for neural network experiments.
- Import any installed package (numpy, scipy, cv2, sklearn, torch, ultralytics, etc).

**What you CANNOT do:**
- Modify `prepare.py` (fixed evaluation + data loading).
- Install new packages (use what's available).
- Skip the evaluation — every run must output `maxn_score`.

**Primary metric**: `maxn_score` from `evaluate_counting()` — higher is better.
This is what drives keep/discard decisions. It measures how close your predicted
MaxN is to the ground truth MaxN for each video.

**Discovery metric**: `evaluate_discovery()` output is logged but doesn't drive
keep/discard. Experiments that reveal temporal patterns (when do fish arrive?),
spatial patterns (where in frame?), or species co-occurrence are valuable for
the hackathon presentation.

## Output format

```
---
maxn_score:       0.823456
mae:              12.5
predicted_maxn:   238
ground_truth:     251
n_frames_processed: 1200
method:           mog2_contour
tier:             1
total_seconds:    45.3
device:           cuda
```

Extract metric: `grep "^maxn_score:" run.log`

## Logging results

`results.tsv` (tab-separated, 6 columns):

```
commit	maxn_score	predicted_maxn	tier	status	description
```

Example:
```
commit	maxn_score	predicted_maxn	tier	status	description
a1b2c3d	0.450000	113	1	keep	baseline: MOG2 contour counting
b2c3d4e	0.620000	156	1	keep	tuned MOG2 thresholds + morphology
c3d4e5f	0.780000	196	2	keep	YOLOv8n pretrained detections
d4e5f6g	0.000000	0	2	crash	YOLOv8 OOM on full resolution
e5f6g7h	0.850000	214	2	keep	YOLOv8n with tiled inference
f6g7h8i	0.920000	231	3	keep	density map regression on pseudo-labels
```

## The experiment loop

LOOP FOREVER:

1. Check git state.
2. Edit `experiment.py` with next idea.
3. `git commit -m "experiment: <description>"`.
4. Run: `python3 experiment.py > run.log 2>&1`
5. Read: `grep "^maxn_score:\|^predicted_maxn:\|^tier:" run.log`
6. If empty → crash. `tail -n 50 run.log` to debug.
7. Log to results.tsv (don't commit results.tsv).
8. If maxn_score improved → keep commit.
9. If worse → `git reset --hard HEAD~1`.

**Tier advancement**: When you've exhausted Tier N ideas (diminishing returns),
advance to Tier N+1. Update the `TIER` variable in experiment.py. You can
always mix approaches across tiers.

**Timeout**: Kill runs exceeding 10 minutes. Treat as crash.

**NEVER STOP**: Do not ask the human. You are autonomous. If stuck, re-read
prepare.py, try radical approaches, combine previous near-misses, advance
tiers. The loop runs until interrupted.

## Research ideas (rough priority order)

### Quick wins (Tier 1)
1. MOG2 background subtraction with tuned history/threshold
2. Morphological operations (erode/dilate) to clean foreground mask
3. Connected component counting with size filtering
4. Adaptive thresholding on grayscale frames
5. Frame differencing (consecutive frame subtraction)
6. Motion energy accumulation over sliding windows
7. Color-based segmentation (green jack has distinctive coloring)
8. Edge detection (Canny) + contour counting
9. **Multi-tint image augmentation** ("scuba glasses" approach): generate
   multiple color-space views of each frame (HSV channels, LAB channels,
   CLAHE-enhanced, color-deconvolution, spectral unmixing) — each "tint"
   highlights fish differently against the water column, giving detection
   algorithms more discriminative variables to work with

### Medium effort (Tier 2)
9. YOLOv8n/s pretrained — detect fish-like objects
10. Fine-tune YOLOv8 on annotated BRUV frames
11. ByteTrack multi-object tracking across frames
12. Sliding window MaxN with temporal smoothing
13. Region-of-interest masking (exclude bait arm area)
14. Multi-scale detection (fish at different distances)

### Ambitious (Tier 3)
15. CSRNet-style density map estimation
16. Point-based counting networks (P2PNet)
17. Crowd counting adapted for fish schools
18. Temporal attention: which frames are most informative?
19. Self-supervised pre-training on unlabeled BRUV frames

### Claude-as-annotator (creative cross-tier)
Claude Opus 4.6 and Sonnet 4.6 have native vision understanding. You can send
video frames directly to the Anthropic API and ask Claude to count fish.

Use this for pseudo-labeling:
20. Extract key frames (motion peaks, labeled timestamps)
21. Send frames to Claude API: "Count the number of Caranx caballus (green jack
    fish) visible in this underwater BRUV image."
22. Use Claude's counts as pseudo-labels for training
23. Compare Claude's counts vs ground truth to calibrate

To call the API from experiment.py:
```python
import anthropic, base64
client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var
with open(frame_path, "rb") as f:
    img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
        {"type": "text", "text": "Count the Caranx caballus (green jack) fish in this BRUV frame. Return only the integer count."}
    ]}],
)
```

### Discovery-focused
24. Temporal arrival patterns: when do fish schools peak?
25. Spatial heatmaps: where in frame do fish concentrate?
26. Species co-occurrence: which species appear together?
27. Bait response curves: fish count over time since deployment
