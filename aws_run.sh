#!/bin/bash
set -euo pipefail

REGION="us-east-1"
INSTANCE_TYPE="${INSTANCE_TYPE:-r5.xlarge}"
KEY_NAME="autoresearch-acoustic-key"
SG_ID="sg-0d0de2713308d5e70"
BUCKET="autoresearch-marine-data"
TRACK="marine-acoustic"
AMI="ami-0b47cd94844ed56a7"

AWS_AK=$(aws configure get aws_access_key_id)
AWS_SK=$(aws configure get aws_secret_access_key)

R2_ENDPOINT="https://6200702e94592ad231a53daba00f8a5d.r2.cloudflarestorage.com"
R2_AK="93bb95ebfe47d5ef93c45efe3c108ca8"
R2_SK="cee49fead9c1a8ac2741a4c2703c908efc5d965100a2d8d20c233fce05547a55"
R2_BUCKET="sala-2026-hackathon-data"

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== marine acoustic remote runner ==="
echo "Instance: $INSTANCE_TYPE"
echo "Region: $REGION"

echo "--- uploading code to S3 ---"
aws s3 cp "$REPO_DIR/prepare.py" "s3://$BUCKET/$TRACK/code/prepare.py" --region $REGION
aws s3 cp "$REPO_DIR/experiment.py" "s3://$BUCKET/$TRACK/code/experiment.py" --region $REGION

echo "--- launching instance ---"
USERDATA=$(cat <<'UDEOF'
#!/bin/bash
set -euo pipefail
exec > /var/log/userdata.log 2>&1

echo "=== marine acoustic autoresearch ==="
date

# Install dependencies
apt-get update -qq
apt-get install -y -qq python3-pip ffmpeg libsndfile1 > /dev/null 2>&1
pip3 install --break-system-packages -q numpy scipy scikit-learn librosa soundfile umap-learn hdbscan matplotlib boto3
pip3 install --break-system-packages -q torch --index-url https://download.pytorch.org/whl/cpu
pip3 install --break-system-packages -q panns-inference

mkdir -p /opt/autoresearch/data/raw/5783 /opt/autoresearch/data/raw/6478 /opt/autoresearch/data/raw/Music_Soundtrap_Pilot
mkdir -p /opt/autoresearch/data/cache /opt/autoresearch/data/results
cd /opt/autoresearch

echo "--- syncing code from S3 ---"
UDEOF
)

# Now append the parts that need variable expansion
USERDATA+="
export AWS_ACCESS_KEY_ID=\"$AWS_AK\"
export AWS_SECRET_ACCESS_KEY=\"$AWS_SK\"
export AWS_DEFAULT_REGION=\"$REGION\"
aws s3 sync \"s3://$BUCKET/$TRACK/code/\" /opt/autoresearch/

echo '--- downloading WAV files from R2 ---'
python3 -u -c \"
import boto3, os
client = boto3.client('s3',
    endpoint_url='$R2_ENDPOINT',
    aws_access_key_id='$R2_AK',
    aws_secret_access_key='$R2_SK',
)
paginator = client.get_paginator('list_objects_v2')
for page in paginator.paginate(Bucket='$R2_BUCKET', Prefix='marine-acoustic/'):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if not key.endswith('.wav'): continue
        parts = key.split('/')
        unit = parts[1]
        fname = parts[-1]
        dest = f'data/raw/{unit}/{fname}'
        if os.path.exists(dest):
            print(f'  {fname} exists, skipping')
            continue
        print(f'  Downloading {key} ({obj[\\\"Size\\\"]/1e6:.0f} MB)...')
        client.download_file('$R2_BUCKET', key, dest)
print('Done downloading WAV files')
\"

echo '--- data ready ---'
find data/raw -name '*.wav' | wc -l
du -sh data/raw/

echo '=== running experiment ==='
cd /opt/autoresearch
python3 -u experiment.py 2>&1 | tee run.log

echo '=== uploading results ==='
STAMP=\$(date +%Y%m%d_%H%M%S)
aws s3 cp run.log \"s3://$BUCKET/$TRACK/results/run_\${STAMP}.log\"
aws s3 sync data/results/ \"s3://$BUCKET/$TRACK/results/\"
grep '^composite_score:\|^silhouette:\|^n_clusters:\|^tier:\|^total_seconds:' run.log > metrics.txt 2>/dev/null || true
aws s3 cp metrics.txt \"s3://$BUCKET/$TRACK/results/latest_metrics.txt\"

echo '=== done ==='
date
"

USERDATA_B64=$(echo "$USERDATA" | base64)

INSTANCE_ID=$(aws ec2 run-instances \
    --image-id $AMI \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --user-data "$USERDATA_B64" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=autoresearch-marine},{Key=Project,Value=autoresearch}]" \
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
echo "=== access ==="
echo "  ssh ubuntu@$PUBLIC_IP"
echo "  ssh ubuntu@$PUBLIC_IP tail -f /var/log/userdata.log"
echo ""
echo "=== monitor results ==="
echo "  aws s3 cp s3://$BUCKET/$TRACK/results/latest_metrics.txt - --region $REGION"
echo ""
echo "=== terminate when done ==="
echo "  aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION"
echo ""
echo "$INSTANCE_ID" > "/tmp/autoresearch-marine-instance"
echo "$PUBLIC_IP" > "/tmp/autoresearch-marine-ip"
