#!/bin/bash
set -euo pipefail

BUCKET="autoresearch-marine-data"
TRACK="${TRACK:-bruv}"

exec > /var/log/userdata.log 2>&1
echo "=== autoresearch cloud setup: $TRACK ==="

apt-get update -qq
apt-get install -y -qq python3-pip python3-venv awscli git ffmpeg libgl1 libglib2.0-0 > /dev/null

mkdir -p /opt/autoresearch
cd /opt/autoresearch

python3 -m venv .venv
source .venv/bin/activate
pip install -q numpy pandas scipy scikit-learn opencv-python-headless torch torchvision ultralytics matplotlib boto3 Pillow

aws s3 sync "s3://$BUCKET/$TRACK/" data/ --region us-east-1

echo "=== data synced ==="
ls -lhR data/

echo "=== ready ==="
touch /opt/autoresearch/.ready
