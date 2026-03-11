# autoresearch-bruv

Autonomous BRUV fish counting research on Apple Silicon. The agent iterates
on `experiment.py` to improve Caranx caballus MaxN prediction from underwater
BRUV video, progressing through hackathon tiers.

## Setup

1. **Run tag**: propose a tag (e.g. `mar10`). Branch: `autoresearch/<tag>`.
2. **Create branch**: `git checkout -b autoresearch/<tag>`.
3. **Read files**: `prepare.py` (fixed), `experiment.py` (you edit), this file.
4. **Verify data**:
   - `data/labels/CumulativeMaxN.csv` must exist. If not, tell human.
   - `data/videos/` should contain at least one MP4 sub-video.
5. **Init results.tsv**: header row only. Baseline recorded after first run.
6. **Go**.

## Context

Baited Remote Underwater Video stations (BRUVs) from MigraMar reef monitoring
in Galapagos. Two video series, 18 sub-videos total (~65 GB).

**Task**: Count the maximum number of Caranx caballus (green jack) visible in
a single frame per video. This is the "MaxN" metric used in marine biology.

**Kaggle competition**: marine-conservation-with-migra-mar

**Species data**:
| Common Name | Scientific Name | Max Count (vid1) | Max Count (vid2) |
|-------------|----------------|-------------------|-------------------|
| Green jack | Caranx caballus | 251 | 52 |
| Almaco jack | Seriola rivoliana | 11 | — |
| Rainbow runner | Elagatis bipinnulata | 4 | — |
| Pilotfish | Naucrates ductor | 3 | — |
| Silky shark | Carcharhinus falciformis | 2 | — |

Primary target is Caranx caballus — appears in large schools up to 251 fish.

**Video structure**:
- 2 series x 9 sub-videos each, ~11.783 min (707 sec) per sub-video
- Vid 1: LGH010001.MP4 through LGH090001.MP4
- Vid 2: LGH010002.MP4 through LGH090002.MP4
- Each ~4 GB except final segments (shorter)
- Native 30fps, ~20,000 frames per sub-video

**Sub-video timing** (CSV timestamps to sub-video mapping):
```
sub_video_index = floor(time_mins / 11.783) + 1
local_time = time_mins - (sub_video_index - 1) * 11.783
```

**Key sub-video**: LGH020002.MP4 has the most Caranx caballus action.

## Platform

Apple Silicon (MPS). Available packages: numpy, scipy, pandas, scikit-learn,
opencv-python, matplotlib, torch (MPS backend), ultralytics, Pillow.

No CUDA. No `torch.compile`. If using PyTorch, use `torch.device("mps")`.

## The Hackathon Tiers

### Tier 1: Classical CV Baselines (starting point)
- Background subtraction (MOG2): stationary BRUV means moving objects = fish
- Frame differencing for motion detection
- Contour detection + area/shape filtering
- Color-based segmentation (fish vs blue water)
- BRUV bait arm masking (always visible, must ignore)
- **Goal**: establish baseline count, understand the visual challenges

### Tier 2: Pretrained Object Detection
- YOLOv8 / RT-DETR out of the box for "fish-shaped" objects
- Fine-tune with semi-automated annotations (SAM + Grounding DINO)
- Tracking-based counting (ByteTrack, BoT-SORT) to avoid double-counting
- Temporal aggregation across frame windows
- **Goal**: reliable fish detection per frame

### Tier 3: Dense Counting & Advanced Methods
- Crowd counting adaptation (CSRNet, CAN) for dense fish schools
- VLM zero-shot counting (send frames to Claude API)
- Active frame selection: lightweight model picks peak-count frames
- Regression on density maps instead of individual detection
- Ensemble: combine detection + density estimation
- **Goal**: handle 251-fish frames accurately

### Cross-Tier Combinations
- Tier 1 motion detection finds active frames -> Tier 2 runs detection on those
- Tier 2 detections -> pseudo-labels for Tier 3 density model training
- Temporal patterns from Tier 1 -> inform frame sampling strategy
- Tier 2 tracking counts -> calibrate Tier 3 density regression

## Hackathon Judging
- **Originality & Innovation (30%)**: unique approaches, surprising discoveries
- **Technical Execution (30%)**: code quality, methodology complexity
- **Impact & Relevance (25%)**: practical applicability for conservation
- **Presentation (15%)**: (handled separately)

## Experimentation Rules

**What you CAN do:**
- Modify `experiment.py` — the ONLY file you edit. Everything is fair game:
  architecture, features, models, counting strategies, anything.
- Use PyTorch with MPS device for neural network experiments.
- Use OpenCV for all image/video processing.
- Import any installed package (numpy, scipy, sklearn, torch, cv2, ultralytics, etc).

**What you CANNOT do:**
- Modify `prepare.py` (fixed evaluation + data loading).
- Install new packages (use what's available).
- Skip the evaluation — every run must output `composite_score`.

**Primary metric**: `composite_score` from `evaluate_maxn_predictions()` — higher
is better. Combines log-scale MAE, mean relative error, and correlation.

## Output format

```
---
composite_score:  0.456789
mae:              23.50
mre:              0.3456
correlation:      0.8901
n_videos:         5
tier:             1
method:           background_subtraction
total_seconds:    45.3
device:           mps
```

Extract metric: `grep "^composite_score:" run.log`

## Logging results

`results.tsv` (tab-separated, 6 columns):

```
commit	composite_score	n_videos	tier	status	description
```

Example:
```
commit	composite_score	n_videos	tier	status	description
a1b2c3d	0.234567	3	1	keep	baseline: MOG2 background subtraction
b2c3d4e	0.312345	3	1	keep	tune contour area thresholds
c3d4e5f	0.423456	5	2	keep	YOLOv8 pretrained fish detection
d4e5f6g	0.000000	0	2	crash	YOLO OOM on full resolution
e5f6g7h	0.534567	5	2	keep	YOLOv8 with frame downscaling
f6g7h8i	0.612345	5	3	keep	density regression + YOLO ensemble
```

## The experiment loop

LOOP FOREVER:

1. Check git state.
2. Edit `experiment.py` with next idea.
3. `git commit -m "experiment: <description>"`.
4. Run: `python3 experiment.py > run.log 2>&1`
5. Read: `grep "^composite_score:\|^n_videos:\|^tier:" run.log`
6. If empty -> crash. `tail -n 50 run.log` to debug.
7. Log to results.tsv (don't commit results.tsv).
8. If composite_score improved -> keep commit.
9. If worse -> `git reset --hard HEAD~1`.

**Tier advancement**: When diminishing returns at Tier N, advance to N+1.
Update `TIER` in experiment.py. Mix approaches across tiers freely.

**Timeout**: Kill runs exceeding 5 minutes. Treat as crash.

**NEVER STOP**: Do not ask the human. You are autonomous. If stuck, re-read
prepare.py, try radical approaches, combine previous near-misses, advance
tiers. The loop runs until interrupted.

## Research ideas (rough priority order)

### Quick wins (Tier 1)
1. Tune MOG2 parameters (history, varThreshold, learning rate)
2. Add morphological operations (open/close) to clean foreground mask
3. BRUV arm masking: detect and mask the bait arm to avoid false positives
4. Color-based fish segmentation in HSV space (fish are silvery/green)
5. Adaptive thresholding + connected components instead of MOG2
6. Frame differencing (current - previous) for simple motion detection
7. Optical flow magnitude as fish activity indicator
8. Watershed segmentation on foreground mask for touching fish

### Medium effort (Tier 2)
9. YOLOv8n (nano) pretrained on COCO — detects "fish" class
10. RT-DETR pretrained for fish-like object detection
11. Fine-tune YOLOv8 on manually labeled frames from this data
12. SAM (Segment Anything) for zero-shot fish segmentation
13. ByteTrack on YOLO detections for unique fish count per window
14. Sliding window temporal aggregation for robust MaxN
15. Multi-scale detection (fish appear at various sizes/distances)
16. Non-max suppression tuning for dense fish clusters

### Ambitious (Tier 3)
17. CSRNet / CAN density map regression for crowd counting adaptation
18. Regression CNN: frame -> fish count (train on pseudo-labeled data)
19. Claude VLM counting: send key frames to Claude API for zero-shot counting
20. Active frame selection: predict frame "interestingness" then count only peaks
21. Temporal transformer: sequence of frames -> MaxN prediction
22. Self-supervised pretraining on unlabeled frames

### Claude-as-counter (creative cross-tier)
Claude has vision capabilities. Send extracted frames to the API:
```python
import anthropic, base64
client = anthropic.Anthropic()
with open(frame_path, "rb") as f:
    img_b64 = base64.standard_b64encode(f.read()).decode("utf-8")
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
        {"type": "text", "text": "Count the number of fish visible in this underwater BRUV image. Return only the integer count."}
    ]}],
)
```
Use this for:
23. Zero-shot counting on peak frames identified by Tier 1/2
24. Validate detection results: compare model counts to Claude's counts
25. Generate training labels for Tier 3 regression models

### Practical tips
- Start with 1 sub-video. LGH020002.MP4 has the most action.
- Sample at 1-2 fps, not 30fps. Most frames are redundant.
- The BRUV bait arm is always visible — learn to ignore it.
- 251 fish in one frame is a dense counting problem — detection may fail.
- Background subtraction works because the BRUV is stationary.
