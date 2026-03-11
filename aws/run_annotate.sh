#!/bin/bash
set -euo pipefail

REGION="us-east-1"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.4xlarge}"
KEY_NAME="autoresearch-key"
SG_ID="sg-0d0de2713308d5e70"
BUCKET="autoresearch-marine-data"
TRACK="bruv"
AMI="ami-0b47cd94844ed56a7"

AWS_AK=$(aws configure get aws_access_key_id)
AWS_SK=$(aws configure get aws_secret_access_key)

R2_ENDPOINT="https://6200702e94592ad231a53daba00f8a5d.r2.cloudflarestorage.com"
R2_AK="93bb95ebfe47d5ef93c45efe3c108ca8"
R2_SK="cee49fead9c1a8ac2741a4c2703c908efc5d965100a2d8d20c233fce05547a55"
R2_BUCKET="sala-2026-hackathon-data"

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Which videos to annotate (default: all)
VIDEOS="${1:---all}"

echo "=== BRUV video annotator ==="
echo "Instance: $INSTANCE_TYPE"
echo "Videos: $VIDEOS"

echo "--- uploading code to S3 ---"
aws s3 cp "$REPO_DIR/annotate.py" "s3://$BUCKET/$TRACK/code/annotate.py" --region $REGION

echo "--- launching spot instance ---"
USERDATA=$(cat <<UDEOF
#!/bin/bash
set -euo pipefail
exec > /var/log/userdata.log 2>&1

export AWS_ACCESS_KEY_ID="$AWS_AK"
export AWS_SECRET_ACCESS_KEY="$AWS_SK"
export AWS_DEFAULT_REGION="$REGION"
export BUCKET="$BUCKET"
export TRACK="$TRACK"

R2_ENDPOINT="$R2_ENDPOINT"
R2_AK="$R2_AK"
R2_SK="$R2_SK"
R2_BUCKET="$R2_BUCKET"

echo "=== BRUV video annotator ==="
date

if command -v conda &>/dev/null; then
    eval "\$(conda shell.bash hook)"
    conda activate pytorch 2>/dev/null || true
    pip install -q opencv-python-headless ultralytics boto3 2>/dev/null || true
else
    apt-get update -qq
    apt-get install -y -qq python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 > /dev/null 2>&1
    python3 -m venv /opt/venv
    source /opt/venv/bin/activate
    pip install -q numpy opencv-python-headless ultralytics boto3
fi

mkdir -p /opt/autoresearch/data/videos /opt/autoresearch/data/results/annotated
cd /opt/autoresearch

echo "--- downloading annotate.py ---"
aws s3 cp "s3://\$BUCKET/\$TRACK/code/annotate.py" .

echo "--- downloading videos from R2 ---"
python3 -u -c "
import boto3, os
client = boto3.client('s3',
    endpoint_url='\$R2_ENDPOINT',
    aws_access_key_id='\$R2_AK',
    aws_secret_access_key='\$R2_SK',
)
paginator = client.get_paginator('list_objects_v2')
videos = []
for page in paginator.paginate(Bucket='\$R2_BUCKET', Prefix='bruv-videos/'):
    for obj in page.get('Contents', []):
        videos.append(obj['Key'].split('/')[-1])
print(f'  Found {len(videos)} videos on R2')
for v in sorted(videos):
    dest = f'data/videos/{v}'
    if os.path.exists(dest):
        print(f'  {v} already exists, skipping')
        continue
    key = f'bruv-videos/{v}'
    print(f'  Downloading {key}...')
    try:
        client.download_file('\$R2_BUCKET', key, dest)
        sz = os.path.getsize(dest) / 1e9
        print(f'  Done: {sz:.2f} GB')
    except Exception as e:
        print(f'  Error: {e}')
"

echo "--- running annotator ---"
python3 -u annotate.py $VIDEOS 2>&1 | tee annotate.log

echo "--- uploading log ---"
aws s3 cp annotate.log "s3://\$BUCKET/\$TRACK/results/annotate.log"

echo "=== done ==="
date
UDEOF
)

USERDATA_B64=$(echo "$USERDATA" | base64)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time","InstanceInterruptionBehavior":"terminate"}}' \
    --user-data "$USERDATA_B64" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=autoresearch-annotate},{Key=Project,Value=autoresearch}]" \
    --region $REGION \
    --query "Instances[0].InstanceId" \
    --output text)

echo ""
echo "Instance: $INSTANCE_ID"
echo "Waiting for running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
    --query "Reservations[0].Instances[0].PublicIpAddress" --output text --region $REGION)

echo "IP: $PUBLIC_IP"
echo ""
echo "=== monitor ==="
echo "  ssh ubuntu@$PUBLIC_IP tail -f /var/log/userdata.log"
echo ""
echo "=== download annotated videos when done ==="
echo "  aws s3 sync s3://$BUCKET/$TRACK/results/annotated/ ./annotated/ --region $REGION"
echo ""
echo "=== terminate when done ==="
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "$INSTANCE_ID" > "/tmp/autoresearch-annotate-instance"
