#!/usr/bin/env bash
# Push code (laptop -> AWS). Excludes outputs, env, and the input catalog.
# Set HOST env var. Optional: USER (default ec2-user), DEST (default roman_l2_job/).
# Usage: HOST=ec2-1-2-3-4.compute-1.amazonaws.com bash scripts/_push.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HOST:?set HOST=<dns_or_ip>}"
USER="${USER:-ec2-user}"
DEST="${DEST:-roman_l2_job/}"

rsync -avz --delete \
    --exclude='output/' \
    --exclude='data/' \
    --exclude='catalogs/sources.parquet' \
    --exclude='.pixi/' \
    --exclude='__pycache__/' \
    --exclude='.git/' \
    ./ "${USER}@${HOST}:${DEST}"
