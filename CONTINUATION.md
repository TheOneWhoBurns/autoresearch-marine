# BRUV Fish Counting — Continuation Guide

## Current State

**Branch**: `bruv-fish-counting` on `TheOneWhoBurns/autoresearch-marine`
**Best composite_score**: 0.785 (2 labeled videos, MAE=1.0)
**Method**: Dual background subtractor (MOG2 + KNN) with foreground pixel density counting

### Latest Results (commit c07ca09)

| Video | True MaxN | Predicted | Error |
|-------|-----------|-----------|-------|
| LGH020002.MP4 | 251 | 251 | 0 |
| LGH040001.MP4 | 52 | 50 | 2 |

## EC2 Instance (RUNNING — terminate when done)

```
Instance: i-0e3c519f0c0027460
Type: c5.4xlarge (16 vCPU, 32GB RAM, ~$0.27/hr spot)
IP: 98.91.235.176
SSH: ssh ubuntu@98.91.235.176
Region: us-east-1
```

**IMPORTANT**: Terminate when done to avoid charges:
```bash
aws ec2 terminate-instances --instance-ids i-0e3c519f0c0027460 --region us-east-1
```

### Instance Layout
- `/opt/autoresearch/` — working directory (chmod 777)
- `/opt/autoresearch/data/videos/` — 15/18 BRUV videos (~60GB)
- `/opt/autoresearch/data/labels/CumulativeMaxN.csv` — ground truth
- `/opt/autoresearch/experiment.py` — current experiment
- `/opt/autoresearch/prepare.py` — fixed evaluation harness (DO NOT MODIFY)
- `/opt/venv/` — Python venv with cv2, numpy, pandas, torch, ultralytics, boto3

### Fast Iteration Loop
```bash
# 1. Edit experiment.py locally
# 2. Deploy and run (~2.5 min cycle):
scp experiment.py ubuntu@98.91.235.176:/opt/autoresearch/experiment.py
ssh ubuntu@98.91.235.176 "cd /opt/autoresearch && source /opt/venv/bin/activate && timeout 200 python3 -u experiment.py 2>&1"
```

### Launch New Instance (if terminated)
```bash
INSTANCE_TYPE=c5.4xlarge ./aws/run_remote.sh bruv
```
This uploads code to S3, launches spot instance, installs deps, downloads all videos from R2, runs experiment.

## AWS Resources

| Resource | Value |
|----------|-------|
| S3 Bucket | `autoresearch-marine-data` |
| Security Group | `sg-0d0de2713308d5e70` |
| SSH Key | `autoresearch-key` (~/.ssh/id_ed25519) |
| AMI | `ami-0b47cd94844ed56a7` (DL AMI PyTorch 2.7) |
| Region | `us-east-1` |

### S3 Layout
```
s3://autoresearch-marine-data/bruv/
  code/experiment.py, prepare.py
  labels/CumulativeMaxN.csv
  results/
```

### R2 Video Source
- Endpoint: `https://6200702e94592ad231a53daba00f8a5d.r2.cloudflarestorage.com`
- Bucket: `sala-2026-hackathon-data`
- Prefix: `bruv-videos/` (18 videos, ~4GB each, 69.5GB total)
- AK: `93bb95ebfe47d5ef93c45efe3c108ca8`
- SK: `cee49fead9c1a8ac2741a4c2703c908efc5d965100a2d8d20c233fce05547a55`

## How the Method Works

1. **Dual background subtraction**: MOG2 (history=300, var=30) + KNN (history=200, dist=400) run in parallel on 0.5x scaled grayscale frames at 1fps
2. **Union mask**: Foreground from either detector is counted (catches fish that one misses)
3. **Pixel density**: Total foreground pixels / PIXELS_PER_FISH (46.0) = fish count per frame
4. **Aggregation**: Blend of 0.45 * p99 + 0.55 * sustained_5s_max (rolling 5-frame average peak)

## Score Breakdown

composite = 0.4 * log_mae_score + 0.4 * mre_score + 0.2 * correlation
- With only 2 labeled videos, correlation=0 (needs >= 3), so max possible = 0.80
- Current 0.785 means log_mae and mre components are nearly maxed out

## What to Try Next

### Quick Wins (stay Tier 1)
- **Lower PIXELS_PER_FISH slightly** (try 44-48 range) to fine-tune
- **Adjust sustained window** (try 3, 7, 10 frames)
- **Adjust blend ratio** (0.40-0.50 range for p99 weight)
- **Lower MOG2_VAR_THRESHOLD to 25** for more sensitivity
- **Try frame differencing** as a third detector in the union

### Bigger Improvements (Tier 2 — needs GPU)
- **YOLO fish detection**: Use YOLOv8 pretrained on COCO (has "fish" class) or fine-tune
- Switch to g4dn.xlarge on-demand (~$0.53/hr, T4 GPU)
- `INSTANCE_TYPE=g4dn.xlarge ./aws/run_remote.sh bruv` (use on-demand, spot quota is 0)
- For on-demand, remove the `--instance-market-options` line from run_remote.sh

### Creative Ideas from program.md
- Use color filtering for Caranx caballus (silvery/green jack)
- Temporal tracking with Hungarian algorithm
- Use key_frames from labels to validate timing (not for training)
- Claude-as-annotator: send peak frames to Claude API for visual counting

## Experiment History

| Commit | Score | Method | Notes |
|--------|-------|--------|-------|
| d911f91 | 0.208 | MOG2+optflow area | Over-counted 6x |
| 84f1486 | 0.311 | contour count / capped density | Under-counted |
| 1711f87 | 0.748 | FG pixel density (ppf=46) | p99 only |
| c07ca09 | 0.785 | Dual BG + sustained blend | Current best |

## Budget

- $112.70 total across 3 tracks for 2 days
- ~$18.78/track/day
- c5.4xlarge spot: ~$6.50/day (well within budget)
- g4dn.xlarge on-demand: ~$12.70/day (still fits)
- EBS: 100GB gp3 = negligible

## Files

- `program.md` — Full problem description, 25 research ideas, rules
- `prepare.py` — Fixed harness (DO NOT MODIFY)
- `experiment.py` — Your code to iterate on
- `aws/run_remote.sh` — EC2 launch script
- `aws/README.md` — AWS setup documentation
- `results.tsv` — Manual tracking (gitignored)
