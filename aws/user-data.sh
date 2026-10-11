#!/bin/bash
# First-boot setup for the FraudLens EC2 instance (Amazon Linux 2023, runs as root once).
# Needs no edits: region and account come from the instance itself, and the bucket name follows
# the convention fraudlens-<account id>-<region> used in docs/aws_deployment.md.
# Log: /var/log/fraudlens-setup.log (view with `sudo cat` over SSH).
set -euo pipefail
exec > >(tee -a /var/log/fraudlens-setup.log) 2>&1
echo "== FraudLens setup started $(date -u)"

# 1. 2 GiB swap: a t3.micro has 1 GiB RAM; the API peaks at ~410 MB while loading.
if [ ! -f /swapfile ]; then
  dd if=/dev/zero of=/swapfile bs=1M count=2048 status=none
  chmod 600 /swapfile
  mkswap /swapfile
  echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi
swapon -a

# 2. Docker (from the AL2023 repository) and the Compose plugin (not packaged for AL2023).
dnf install -y docker git
systemctl enable --now docker
usermod -aG docker ec2-user
COMPOSE_VERSION=v5.6.0
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
  "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-x86_64"
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
docker compose version

# 3. Where am I? (IMDSv2: a session token is required.)
TOKEN=$(curl -fsS -X PUT http://169.254.169.254/latest/api/token \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
REGION=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/placement/region)
ACCOUNT_ID=$(curl -fsS -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/dynamic/instance-identity/document | grep -oP '"accountId"\s*:\s*"\K[0-9]+')
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"

# 4. The repository (public) for the compose file and the Prometheus/Grafana configuration.
if [ ! -d /opt/fraudlens ]; then
  git clone --depth 1 https://github.com/YASHR2002/fraudlens.git /opt/fraudlens
fi
cd /opt/fraudlens

# 5. Secrets from SSM Parameter Store (instance role; nothing stored in the image or in Git).
param() {
  aws ssm get-parameter --region "$REGION" --name "$1" --with-decryption \
    --query Parameter.Value --output text
}
umask 077
# The Gemini key is optional: without it the API writes template notes instead of LLM ones.
printf 'GOOGLE_API_KEY=%s\n' "$(param /fraudlens/google-api-key || true)" > secrets.env
cat > .env <<ENV
API_IMAGE=${REGISTRY}/fraudlens-api:latest
AWS_REGION=${REGION}
S3_BUCKET=fraudlens-${ACCOUNT_ID}-${REGION}
GRAFANA_ADMIN_PASSWORD=$(param /fraudlens/grafana-admin-password)
ENV

# 6. Pull from ECR and start. restart: unless-stopped brings the stack back after a reboot.
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker compose -f docker-compose.aws.yml pull
docker compose -f docker-compose.aws.yml up -d
echo "== FraudLens setup finished $(date -u)"
