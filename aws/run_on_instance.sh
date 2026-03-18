#!/bin/bash
# Upload experiment.py to running instance and run it
# Usage: ./aws/run_on_instance.sh [IP]
set -euo pipefail

IP="${1:-$(cat /tmp/autoresearch-precip-ip)}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AWS="/Users/sol/Library/Python/3.9/bin/aws"
BUCKET="autoresearch-marine-data"
REGION="us-east-1"

echo "--- uploading experiment.py to $IP ---"
scp -o StrictHostKeyChecking=no "$REPO_DIR/experiment.py" ubuntu@$IP:/opt/autoresearch/experiment.py

echo "--- running experiment remotely ---"
ssh -o StrictHostKeyChecking=no ubuntu@$IP "cd /opt/autoresearch && source /opt/venv/bin/activate && python3 -u experiment.py 2>&1 | tee run.log; grep '^composite_score:\|^f1_3h:\|^f1_6h:\|^f1_12h:\|^tier:\|^model:' run.log > metrics.txt"

echo ""
echo "--- results ---"
ssh -o StrictHostKeyChecking=no ubuntu@$IP "cat /opt/autoresearch/metrics.txt"

echo ""
echo "--- uploading results to S3 ---"
ssh -o StrictHostKeyChecking=no ubuntu@$IP "
export AWS_ACCESS_KEY_ID=\$(grep aws_access_key_id ~/.aws/credentials 2>/dev/null | awk '{print \$3}' || echo '')
cd /opt/autoresearch
STAMP=\$(date +%Y%m%d_%H%M%S)
aws s3 cp run.log s3://$BUCKET/precip/results/run_\$STAMP.log --region $REGION 2>/dev/null || true
aws s3 cp metrics.txt s3://$BUCKET/precip/results/latest_metrics.txt --region $REGION 2>/dev/null || true
" 2>/dev/null || true
