#!/usr/bin/env bash
# Phase 4c: run MosaicPipeline on a skycell's selected asn.
#
# Usage: scripts/detection/04c_run_mosaic.sh <SKYCELL_ID> [skycell_name]
set -euo pipefail
cd "$(dirname "$0")/../.."

CELL="${1:-}"
SKYCELL="${2:-}"
[ -n "$CELL" ] || { echo "usage: $0 <SKYCELL_ID> [skycell_name]"; exit 1; }

ASN_DIR="output/detection/asn"
L3_DIR="output/detection/l3"
LOG_DIR="output/detection/logs"
mkdir -p "$L3_DIR" "$LOG_DIR"

if [ -z "$SKYCELL" ]; then
    ASN=$(for f in "$ASN_DIR/sky_${CELL}"_*_asn.json; do
        n=$(grep -c '"expname"' "$f"); echo "$n $f"
    done | sort -rn | head -1 | awk '{print $2}')
    echo "no skycell specified — picking best-coverage asn: $ASN"
else
    ASN=$(ls "$ASN_DIR/sky_${CELL}"_*"_${SKYCELL}_"*_asn.json 2>/dev/null | head -1)
fi
[ -n "$ASN" ] || { echo "no asn found in $ASN_DIR/ for skycell $CELL (run 04b first)"; exit 1; }

BASE=$(basename "$ASN" _asn.json)
OUT="${L3_DIR}/${BASE}_coadd.asdf"
LOG="${LOG_DIR}/${BASE}_mosaic.log"

if [ -f "$OUT" ]; then
    echo "already exists: $OUT"
    exit 0
fi

echo "Running MosaicPipeline on $ASN ..."
echo "  -> $OUT"

pixi run --manifest-path pixi.toml strun romancal.pipeline.MosaicPipeline \
    "$ASN" \
    --output_dir "$L3_DIR" \
    --steps.resample.in_memory=False \
    > "$LOG" 2>&1

echo "Done. Output: $OUT"
ls -la "$OUT"
