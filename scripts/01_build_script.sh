#!/usr/bin/env bash
# Stage 01: build sims.script for a given config.
# - Regenerates pointings_<tag>.ecsv from the config if missing.
# - Calls the patched make_stack.py with --make_script.
# - Post-processes each line to:
#     * prepend output/cal/ to the output filename
#     * replace the constant --rng_seed with a deterministic per-line seed
#     * wrap with `[ -f <out> ] || ... > log 2>&1` for skip-if-exists + per-sim log
# Usage: 01_build_script.sh configs/<tag>.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: $0 configs/<tag>.yaml"; exit 1; }
eval "$(pixi run python scripts/_config.py "$CONFIG")"

POINTINGS="pointings_${TAG}.ecsv"
RAW="${OUTPUT_BASE}/${TAG}/sims_raw.script"
SCRIPT="${OUTPUT_BASE}/${TAG}/sims.script"
CATALOG="catalogs/sources.parquet"

[ -f "$CATALOG" ] || { echo "missing $CATALOG (run 00_prepare_catalog.py $CONFIG first)"; exit 1; }

# Regenerate pointings file from config if missing — the config is the
# source of truth for region + bandpass + visit restrictions.
if [ ! -f "$POINTINGS" ]; then
    echo "pointings file $POINTINGS missing; regenerating from $CONFIG..."
    pixi run python scripts/filter_pointings.py "$CONFIG"
fi

mkdir -p "${OUTPUT_BASE}/cal" "${OUTPUT_BASE}/logs" "${OUTPUT_BASE}/${TAG}"

# Generate raw script with constant seed; we rewrite per line below.
pixi run python scripts/make_stack.py "$POINTINGS" "$CATALOG" \
    --make_script "${OUTPUT_BASE}/${TAG}/sims_raw" \
    --level 2 --usecrds --psftype stpsf --rng_seed 1

pixi run python scripts/_postprocess_sims.py \
    --input "$RAW" --output "$SCRIPT" \
    --cal-dir "${OUTPUT_BASE}/cal" --log-dir "${OUTPUT_BASE}/logs"

N=$(wc -l < "$SCRIPT")
echo "Wrote $SCRIPT (${N} lines)"
