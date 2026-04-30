#!/usr/bin/env bash
# Stage 04: run romancal MosaicPipeline on each association in this tag,
# in parallel.
#
# Each strun invocation is independent; we drive them via xargs -P the same
# way stage 02 runs romanisim. Memory: each worker holds one input image at
# a time (in_memory=False on resample) plus a 5000x5000 coadd buffer, so
# 4-way fits comfortably on a 32 GB box. On bigger RAM we can go wider.
# Skip-if-exists guards each mosaic; safe to re-run.
#
# Usage: 04_run_mosaic.sh configs/<tag>.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: $0 configs/<tag>.yaml"; exit 1; }
eval "$(pixi run python scripts/_config.py "$CONFIG")"

ASN_DIR="${OUTPUT_BASE}/${TAG}/asn"
MOSAIC_DIR="${OUTPUT_BASE}/${TAG}/mosaic"
LOG_DIR="${OUTPUT_BASE}/logs"
PARALLELISM="${PARALLELISM:-$RUN_PARALLELISM}"

[ -d "$ASN_DIR" ] || { echo "missing $ASN_DIR (run 03_build_asn.sh $CONFIG first)"; exit 1; }
mkdir -p "$MOSAIC_DIR" "$LOG_DIR"

ASN_FILES=("$ASN_DIR"/*_asn.json)
[ -e "${ASN_FILES[0]}" ] || { echo "no asn files in $ASN_DIR"; exit 1; }

echo "Running MosaicPipeline on ${#ASN_FILES[@]} associations with ${PARALLELISM} workers..."
START=$(date +%s)

# Emit one self-contained skip-guarded strun line per asn, then run them in
# parallel. strun is on PATH because we're inside `pixi run bash`, so no
# per-worker pixi activation cost.
for asn in "${ASN_FILES[@]}"; do
    base=$(basename "$asn" .json)
    out="${MOSAIC_DIR}/${base%_asn}_coadd.asdf"
    log="${LOG_DIR}/${base%_asn}_mosaic.log"
    printf '[ -f %q ] || strun romancal.pipeline.MosaicPipeline %q --output_dir %q --steps.resample.in_memory=False > %q 2>&1\n' \
        "$out" "$asn" "$MOSAIC_DIR" "$log"
done | xargs -P "$PARALLELISM" -I{} bash -c '{}'

END=$(date +%s)
echo "Done in $((END - START)) seconds."
