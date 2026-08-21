#!/usr/bin/env bash
# open-agent-robot — provision a secured GPU dev instance in ap-southeast-1.
# Security posture: SSH-only ingress restricted to the operator's current IP,
# IMDSv2 required, encrypted gp3 root, instance role limited to Bedrock invoke.
# Idempotent: safe to re-run; reuses existing resources by name/tag.
set -euo pipefail

REGION="${REGION:-ap-southeast-1}"
NAME="open-agent-robot"
INSTANCE_TYPE="${INSTANCE_TYPE:-g4dn.xlarge}"
AMI_ID="${AMI_ID:-ami-0ed6a65b84536f6ce}"   # ubuntu noble 24.04 amd64 20260714
VOL_GB="${VOL_GB:-200}"
KEY_FILE="$HOME/.ssh/${NAME}.pem"
MY_IP="$(curl -s --max-time 5 https://checkip.amazonaws.com)"
AWS=(aws --region "$REGION")

echo "== key pair =="
if ! "${AWS[@]}" ec2 describe-key-pairs --key-names "$NAME" >/dev/null 2>&1; then
  "${AWS[@]}" ec2 create-key-pair --key-name "$NAME" \
    --query 'KeyMaterial' --output text > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  echo "created $KEY_FILE"
else
  echo "exists (private key must already be at $KEY_FILE)"
fi

echo "== security group =="
VPC_ID=$("${AWS[@]}" ec2 describe-vpcs --filters Name=is-default,Values=true \
  --query 'Vpcs[0].VpcId' --output text)
SG_ID=$("${AWS[@]}" ec2 describe-security-groups \
  --filters Name=group-name,Values="$NAME" Name=vpc-id,Values="$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
  SG_ID=$("${AWS[@]}" ec2 create-security-group --group-name "$NAME" \
    --description "open-agent-robot: SSH only, operator IP only" \
    --vpc-id "$VPC_ID" --query 'GroupId' --output text)
fi
# Replace any stale SSH rules with the current operator IP.
EXISTING=$("${AWS[@]}" ec2 describe-security-groups --group-ids "$SG_ID" \
  --query 'SecurityGroups[0].IpPermissions' --output json)
if [ "$EXISTING" != "[]" ]; then
  "${AWS[@]}" ec2 revoke-security-group-ingress --group-id "$SG_ID" \
    --ip-permissions "$EXISTING" >/dev/null
fi
"${AWS[@]}" ec2 authorize-security-group-ingress --group-id "$SG_ID" \
  --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${MY_IP}/32,Description=operator}]" >/dev/null
echo "$SG_ID (22/tcp from ${MY_IP}/32 only)"

echo "== IAM role (Bedrock invoke only) =="
ROLE="${NAME}-ec2"
if ! "${AWS[@]}" iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  "${AWS[@]}" iam create-role --role-name "$ROLE" --assume-role-policy-document '{
    "Version":"2012-10-17",
    "Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
fi
"${AWS[@]}" iam put-role-policy --role-name "$ROLE" --policy-name bedrock-invoke \
  --policy-document '{
  "Version":"2012-10-17",
  "Statement":[{"Effect":"Allow",
    "Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],
    "Resource":"*"}]}'
if ! "${AWS[@]}" iam get-instance-profile --instance-profile-name "$ROLE" >/dev/null 2>&1; then
  "${AWS[@]}" iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  "${AWS[@]}" iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  sleep 10  # instance-profile propagation
fi
echo "$ROLE"

echo "== instance =="
IID=$("${AWS[@]}" ec2 describe-instances \
  --filters "Name=tag:Name,Values=$NAME" "Name=instance-state-name,Values=pending,running,stopped,stopping" \
  --query 'Reservations[0].Instances[0].InstanceId' --output text 2>/dev/null || true)
if [ -z "$IID" ] || [ "$IID" = "None" ]; then
  IID=$("${AWS[@]}" ec2 run-instances \
    --image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE" \
    --key-name "$NAME" --security-group-ids "$SG_ID" \
    --iam-instance-profile "Name=$ROLE" \
    --metadata-options 'HttpTokens=required,HttpEndpoint=enabled' \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":${VOL_GB},\"VolumeType\":\"gp3\",\"Encrypted\":true,\"DeleteOnTermination\":true}}]" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=Project,Value=$NAME}]" \
    --query 'Instances[0].InstanceId' --output text)
  echo "launched $IID"
else
  STATE=$("${AWS[@]}" ec2 describe-instances --instance-ids "$IID" \
    --query 'Reservations[0].Instances[0].State.Name' --output text)
  [ "$STATE" = "stopped" ] && "${AWS[@]}" ec2 start-instances --instance-ids "$IID" >/dev/null
  echo "reusing $IID ($STATE)"
fi
"${AWS[@]}" ec2 wait instance-running --instance-ids "$IID"
IP=$("${AWS[@]}" ec2 describe-instances --instance-ids "$IID" \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "READY  instance=$IID  ip=$IP"

# Write the operator's local, git-ignored config from the live values. The
# instance id and public IP are account-specific (and the IP rotates on every
# stop/start), so they live here and never in a tracked file.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sed -e "s|<EC2_PUBLIC_IP>|$IP|" -e "s|<PATH_TO_KEY>|$KEY_FILE|" \
  "$HERE/ssh_config.example" > "$HERE/ssh_config"
cat > "$HERE/local.env" <<EOF
# generated by provision.sh — git-ignored, account-specific
export OAR_INSTANCE_ID=$IID
export OAR_REGION=$REGION
export OAR_HOST=$IP
EOF
echo "wrote $HERE/ssh_config and $HERE/local.env"
echo "ssh -F $HERE/ssh_config robot"
