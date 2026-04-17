#!/usr/bin/env bash
# Stage 01: build sims.script for a given pointings file.
# - Calls the patched make_stack.py with --make_script
# - Post-processes each line to:
#     * prepend output/cal/ to the output filename
#     * replace the constant --rng_seed with a deterministic per-line seed
#     * wrap with `[ -f <out> ] || ... > log 2>&1` for skip-if-exists + per-sim log
# Usage: 01_build_script.sh <smoke|full>
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:-smoke}"
POINTINGS="pointings_${TAG}.ecsv"
RAW="output/${TAG}/sims_raw.script"
SCRIPT="output/${TAG}/sims.script"
CATALOG="catalogs/sources.parquet"

[ -f "$POINTINGS" ] || { echo "missing $POINTINGS"; exit 1; }
[ -f "$CATALOG" ]   || { echo "missing $CATALOG (run 00_prepare_catalog.py first)"; exit 1; }

mkdir -p output/cal output/logs "output/${TAG}"

# Generate raw script with constant seed; we rewrite per line below.
pixi run python scripts/make_stack.py "$POINTINGS" "$CATALOG" \
    --make_script "output/${TAG}/sims_raw" \
    --level 2 --usecrds --psftype stpsf --rng_seed 1

pixi run python scripts/_postprocess_sims.py \
    --input "$RAW" --output "$SCRIPT" \
    --cal-dir output/cal --log-dir output/logs

N=$(wc -l < "$SCRIPT")
echo "Wrote $SCRIPT (${N} lines)"
