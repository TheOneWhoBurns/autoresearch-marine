# Instance Coordination

## Active Instances

### BRUV Fish Counting (claude/nostalgic-moore)
- **Instance**: i-023847e3f71fc5e11 (g4dn.xlarge, GPU spot)
- **IP**: 3.238.196.143
- **Task**: Experiment with IoU tracking integration
- **Status**: Setting up (pip install + video download)
- **Branch**: claude/nostalgic-moore
- **What's running**: experiment.py with new `tier2_tracked_count` — IoU-based fish tracking on 40-frame window around peak activity
- **Expected output**: composite_score with tracking-aware ensemble (T1 pixel density + T2 tracked count)
- **Monitor**: `ssh ubuntu@3.238.196.143 sudo tail -f /var/log/userdata.log`
- **Terminate**: `aws ec2 terminate-instances --instance-ids i-023847e3f71fc5e11 --region us-east-1`

### Other Running Instances
- i-063aa589c231aac84 (r5.xlarge) — unknown task, check owner
- i-06fd82897332e5481 (c5.2xlarge) — unknown task, check owner

## Shared Resources
- **S3 bucket**: s3://autoresearch-marine-data/bruv/
  - `code/` — latest experiment scripts
  - `models/` — trained classifier (caballus_classifier.pkl, GBM F1=0.831)
  - `results/` — experiment results, logs, verification frames
- **R2 bucket**: sala-2026-hackathon-data (bruv-videos/)
- **GPU quotas**: G/VT: 4 vCPUs, P: 8 vCPUs (both spot + on-demand)
- **On-demand vCPU limit**: 16 (currently 12 used by other instances)

## Current BRUV Score
- **composite_score: 0.997102** (MAE 0.4, LGH020002: 251→255, LGH040001: 52→50)
- Approach: dual BG subtraction + YOLO-calibrated PPF + species classifier + harmonic mean aggregation

## What's Been Done
- Tier 1: BG subtraction (MOG2+KNN), bait arm masking, 1fps sampling
- Tier 1: Pre-trained YOLO proxy detection ("kite"/"bird" classes)
- Tier 2: Species classifier (GBM on HSV+gradient features, iNaturalist+BRUV training data)
- Tier 2: Adaptive per-video PPF calibration via YOLO
- Tier 3: Claude VLM zero-shot counting
- Tier 3: Simple probability average RF+LGB+XGB ensemble

## What Needs Doing (Priority Order)
1. **IoU tracking** (IN PROGRESS on GPU instance) — prevents double-counting
2. **Fine-tune YOLO on fish data** — replace proxy classes with real fish detector
3. **ByteTrack/BoT-SORT** — more sophisticated tracking than simple IoU
4. **Crowd counting (CSRNet/CAN)** — for dense 251-fish scenes where YOLO plateaus at ~80
5. **More iNat training data** — fix CDN fallback in download_inat.py, re-download
6. **Temporal sliding window** — smooth predictions across time

## Communication Protocol
- Update this file when starting/finishing tasks
- Pull before pushing: `git pull origin talk`
- Each instance should note what it's working on to avoid duplication
- Results go to S3, coordination goes here
