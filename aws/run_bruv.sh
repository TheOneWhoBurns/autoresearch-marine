#!/bin/bash
set -euo pipefail

REGION="us-east-1"
INSTANCE_TYPE="${INSTANCE_TYPE:-c5.xlarge}"
KEY_NAME="autoresearch-acoustic-key"
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

echo "=== BRUV Experiment: ByteTrack IoU Tracking ==="
echo "Instance: $INSTANCE_TYPE"

USERDATA=$(cat <<UDEOF
#!/bin/bash
set -euo pipefail
exec > /var/log/userdata.log 2>&1

export AWS_ACCESS_KEY_ID="$AWS_AK"
export AWS_SECRET_ACCESS_KEY="$AWS_SK"
export AWS_DEFAULT_REGION="$REGION"
export BUCKET="$BUCKET"
export TRACK="$TRACK"

export R2_ENDPOINT="$R2_ENDPOINT"
export R2_AK="$R2_AK"
export R2_SK="$R2_SK"
export R2_BUCKET="$R2_BUCKET"

echo "=== BRUV Experiment with IoU Tracking ==="
date

# Setup Python env — use conda on DL AMI (has PyTorch+CUDA pre-installed)
if command -v conda &>/dev/null; then
    eval "\$(conda shell.bash hook)"
    conda activate pytorch 2>/dev/null || conda activate base
    pip install -q opencv-python-headless ultralytics boto3 scikit-learn scipy pillow pandas anthropic 2>/dev/null || true
else
    apt-get update -qq
    apt-get install -y -qq python3-pip python3-venv ffmpeg libgl1 libglib2.0-0 > /dev/null 2>&1
    python3 -m venv /opt/venv
    source /opt/venv/bin/activate
    pip install -q numpy opencv-python-headless ultralytics boto3 scikit-learn scipy pillow pandas anthropic
fi

mkdir -p /opt/autoresearch/data/{videos,labels,models,results}
cd /opt/autoresearch

echo "--- downloading code ---"
for f in experiment.py prepare.py; do
    aws s3 cp "s3://\$BUCKET/\$TRACK/code/\$f" . --region $REGION
done

echo "--- downloading labels ---"
aws s3 cp "s3://\$BUCKET/\$TRACK/code/labels/CumulativeMaxN.csv" data/labels/ --region $REGION 2>/dev/null || true

echo "--- downloading trained model ---"
aws s3 cp "s3://\$BUCKET/\$TRACK/models/caballus_classifier.pkl" data/models/ --region $REGION

echo "--- downloading BRUV videos from R2 ---"
python3 -c "
import boto3, os
r2 = boto3.client('s3',
    endpoint_url=os.environ['R2_ENDPOINT'],
    aws_access_key_id=os.environ['R2_AK'],
    aws_secret_access_key=os.environ['R2_SK'])
bucket = os.environ['R2_BUCKET']
resp = r2.list_objects_v2(Bucket=bucket, Prefix='bruv-videos/')
for obj in resp.get('Contents', []):
    key = obj['Key']
    name = key.split('/')[-1]
    if not name.endswith('.MP4'):
        continue
    dest = f'data/videos/{name}'
    if not os.path.exists(dest):
        print(f'Downloading {name}...')
        r2.download_file(bucket, key, dest)
        print(f'  Done: {os.path.getsize(dest)/1e6:.0f}MB')
"

echo ""
echo "=== Running experiment ==="
date
python3 -u experiment.py 2>&1 | tee /tmp/experiment.log

echo ""
echo "--- uploading results ---"
aws s3 cp /tmp/experiment.log "s3://\$BUCKET/\$TRACK/results/experiment_tracking.log" --region $REGION

# Upload all result JSONs
for f in data/results/result_*.json; do
    aws s3 cp "\$f" "s3://\$BUCKET/\$TRACK/results/\$(basename \$f)" --region $REGION 2>/dev/null || true
done

# Upload latest metrics
tail -10 /tmp/experiment.log | grep -E "^(composite|mae|mre|n_videos|tier|method)" > /tmp/latest_metrics.txt 2>/dev/null || true
aws s3 cp /tmp/latest_metrics.txt "s3://\$BUCKET/\$TRACK/results/latest_metrics.txt" --region $REGION 2>/dev/null || true

echo ""
echo "=== DONE ==="
date
echo "DONE \$(date)" | aws s3 cp - "s3://\$BUCKET/\$TRACK/results/experiment_done.txt" --region $REGION
UDEOF
)

USERDATA_B64=$(echo "$USERDATA" | base64)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --user-data "$USERDATA_B64" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=autoresearch-experiment-gpu},{Key=Project,Value=autoresearch}]" \
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
echo "  ssh -i ~/.ssh/autoresearch-acoustic-key.pem ubuntu@$PUBLIC_IP sudo tail -f /var/log/userdata.log"
echo ""
echo "=== terminate when done ==="
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "$INSTANCE_ID" > "/tmp/autoresearch-experiment-instance"
