# AWS Remote Execution for Autoresearch

Run experiments on EC2 spot instances instead of locally. The agent iterates
on `experiment.py` locally, but the actual `python3 experiment.py` runs on AWS.

## Architecture

```
LOCAL (agent loop)                    AWS (compute)
─────────────────                    ──────────────
experiment.py ──upload──> S3 bucket
                                     EC2 spot instance:
                                       pull code from S3
                                       pull videos from R2
                                       run experiment.py
                                       push results to S3
results <──────poll────── S3 bucket
```

## Setup (one-time, already done)

```bash
# S3 bucket for code + results
aws s3 mb s3://autoresearch-marine-data --region us-east-1

# Security group for SSH
aws ec2 create-security-group --group-name autoresearch-ssh ...
aws ec2 authorize-security-group-ingress ... --port 22

# SSH key (imported existing ed25519)
aws ec2 import-key-pair --key-name autoresearch-key --public-key-material fileb://~/.ssh/id_ed25519.pub

# Labels uploaded to S3
aws s3 cp data/labels/CumulativeMaxN.csv s3://autoresearch-marine-data/bruv/labels/
```

## Resources

| Resource | Value |
|----------|-------|
| S3 Bucket | `autoresearch-marine-data` |
| Security Group | `sg-0d0de2713308d5e70` (autoresearch-ssh) |
| SSH Key | `autoresearch-key` (uses ~/.ssh/id_ed25519) |
| AMI | `ami-0b47cd94844ed56a7` (DL AMI PyTorch 2.7, Ubuntu 22.04) |
| Region | `us-east-1` |

## Usage

### Run an experiment remotely

```bash
# BRUV fish counting (default)
./aws/run_remote.sh bruv

# With different instance type
INSTANCE_TYPE=c5.2xlarge ./aws/run_remote.sh bruv
```

### Monitor

```bash
# SSH and watch logs
ssh ubuntu@$(cat /tmp/autoresearch-bruv-ip) tail -f /var/log/userdata.log

# Check metrics from S3
aws s3 cp s3://autoresearch-marine-data/bruv/results/latest_metrics.txt - --region us-east-1
```

### Terminate

```bash
aws ec2 terminate-instances --instance-ids $(cat /tmp/autoresearch-bruv-instance) --region us-east-1
```

## S3 Layout

```
s3://autoresearch-marine-data/
├── bruv/
│   ├── labels/          # CumulativeMaxN.csv, TimeFirstSeen.csv
│   ├── code/            # prepare.py, experiment.py (uploaded each run)
│   └── results/         # run logs, metrics, result JSONs
├── acoustic/            # (same structure for marine acoustic track)
└── precip/              # (same structure for precipitation track)
```

## Adapting for Other Tracks

1. Upload your track's labels/data to `s3://autoresearch-marine-data/<track>/labels/`
2. If your data is on R2, add the R2 keys to the video download section in run_remote.sh
3. Run: `./aws/run_remote.sh <track>`

The userdata script auto-detects the DL AMI (conda pytorch) vs plain Ubuntu
and installs accordingly. Videos pull directly from R2 for speed.

## Cost

- **g4dn.xlarge** (T4 GPU): ~$0.22/hr spot → ~$5/day
- **c5.2xlarge** (CPU only): ~$0.14/hr spot → ~$3/day
- S3 storage: negligible for code+results, ~$0.08/mo for 3.7GB video
- Budget: $112.70 / 3 tracks / 2 days ≈ $18/track/day (plenty)

Instances are spot (one-time) — they auto-terminate if interrupted.
Always terminate manually when done to avoid waste.
