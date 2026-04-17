#!/usr/bin/env bash
# Pull results (AWS -> laptop). Skips L2 cal files (~65 GB) by default.
# Set NO_MOSAICS=1 to also skip mosaic ASDFs (~17 GB).
# Set HOST env var. Optional: USER (default ec2-user), SRC (default roman_l2_job/).
# Usage: HOST=ec2-1-2-3-4.compute-1.amazonaws.com bash scripts/_pull.sh
set -euo pipefail
cd "$(dirname "$0")/.."

: "${HOST:?set HOST=<dns_or_ip>}"
USER="${USER:-ec2-user}"
SRC="${SRC:-roman_l2_job/}"

INCLUDES=(
    --include='output/'
    --include='output/logs/***'
    --include='output/crds_context.log'
    --include='output/*/'
    --include='output/*/asn/***'
    --include='output/*/catalog/***'
    --include='output/*/sims.script'
    --include='output/*/sims_raw.script'
)
if [ -z "${NO_MOSAICS:-}" ]; then
    INCLUDES+=(--include='output/*/mosaic/***')
fi
INCLUDES+=(--exclude='output/cal/***' --exclude='*')

rsync -avz "${INCLUDES[@]}" "${USER}@${HOST}:${SRC}" ./
