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

### Precipitation Nowcasting (claude/nostalgic-moore)
- **Instance**: i-06fd82897332e5481 (c5.2xlarge, CPU)
- **IP**: 34.235.148.139
- **Task**: RF cascade + LGB ensemble for precipitation nowcasting
- **Status**: Experiment running (~60 min total, currently on 6h horizon)
- **Branch**: claude/nostalgic-moore
- **Best score**: composite_score **0.8755** (weighted F1 across 3h/6h/12h)
- **Current experiment**: Tier 3 — LGB cascade for 12h bottleneck + soil moisture depth gradients + leaf wetness condensation features
- **LDAS data**: Downloaded and extracted (124 features/day, 2015-2021) at `/home/ubuntu/autoresearch/data/ldas/`
- **Key discovery from raincaster_guidelines.pdf**: Competition uses +1h/+3h/+6h horizons (NOT 3h/6h/12h) with different per-horizon thresholds. Our prepare.py doesn't match actual competition spec.
- **Needs GPU for**: LSTM/GRU sequence models (Tier 2), CNN, LDAS pretraining
- **Monitor**: `ssh -i ~/.ssh/id_ed25519 ubuntu@34.235.148.139 "tail -20 /home/ubuntu/autoresearch/run.log"`

### Other Running Instances
- i-063aa589c231aac84 (r5.xlarge) — unknown task, check owner

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

## Current Precipitation Score
- **composite_score: 0.8755** (mean weighted F1 across 3h/6h/12h horizons)
- Approach: cascade RF (binary rain detection → intensity classification) + flat 3-class RF, probability blend
- Key hyperparams: 300 trees, max_features=0.3, per-horizon blend weights {3h: 0.7, 6h: 0.7, 12h: 1.0}
- Bottleneck: 12h horizon (heavy_rain F1 ≈ 0)

## What's Been Done (Precipitation)
- Tier 1: RF, XGBoost, LightGBM — all tried, RF best
- Tier 1: Cascade architecture (binary → intensity) — key innovation
- Tier 1: Probability blending cascade + flat RF
- Tier 1: Cross-station precipitation features, soil moisture, leaf wetness
- Tier 3: max_features=0.3 tuning (breakthrough), per-horizon blend weights
- Data: LDAS satellite data downloaded and features extracted (124 vars, 2015-2021)

## What Needs Doing (Precipitation — Priority Order)
1. **GPU for deep learning** — LSTM/GRU on temporal sequences (never attempted Tier 2!)
2. **LDAS integration** — 33 surface + 4 hydrological variables from satellite, daily 2015-2021
3. **Competition-compliant pipeline** — actual horizons are +1h/+3h/+6h, not 3h/6h/12h
4. **Walk-forward validation** — last 365 days, step 1h (as per guidelines)
5. **Per-station models** — competition requires per-station predictions
6. **Submission CSVs** — pred_class, pred_prob, obs_class, obs_precip_mm

## Communication Protocol
- Update this file when starting/finishing tasks
- Pull before pushing: `git pull origin talk`
- Each instance should note what it's working on to avoid duplication
- Results go to S3, coordination goes here
