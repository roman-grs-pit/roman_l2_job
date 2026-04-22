#!/usr/bin/env bash
# Phase 4: run MosaicPipeline on the pair's selected skycell association,
# SKIPPING source_catalog (we run that separately in Phase 5).
#
# Usage: scripts/detection/04c_run_mosaic.sh 11119
set -euo pipefail
cd "$(dirname "$0")/../.."

PAIR="${1:-}"
SKYCELL="${2:-}"
[ -n "$PAIR" ] || { echo "usage: $0 <PAIR_ID> [skycell_name]"; exit 1; }

ASN_DIR="output/detection/asn"
L3_DIR="output/detection/l3"
LOG_DIR="output/detection/logs"
mkdir -p "$L3_DIR" "$LOG_DIR"

if [ -z "$SKYCELL" ]; then
    # Pick the asn with the most L2 members (best coverage)
    ASN=$(for f in "$ASN_DIR/pair_${PAIR}"_*_asn.json; do
        n=$(grep -c '"expname"' "$f"); echo "$n $f"
    done | sort -rn | head -1 | awk '{print $2}')
    echo "no skycell specified — picking best-coverage asn: $ASN"
else
    ASN=$(ls "$ASN_DIR/pair_${PAIR}"_*"_${SKYCELL}_"*_asn.json 2>/dev/null | head -1)
fi
[ -n "$ASN" ] || { echo "no asn found in $ASN_DIR/ for pair $PAIR (run 04b first)"; exit 1; }

BASE=$(basename "$ASN" _asn.json)
OUT="${L3_DIR}/${BASE}_coadd.asdf"
LOG="${LOG_DIR}/${BASE}_mosaic.log"

if [ -f "$OUT" ]; then
    echo "already exists: $OUT"
    exit 0
fi

echo "Running MosaicPipeline on $ASN ..."
echo "  -> $OUT"

# MosaicPipeline includes an internal source_catalog step that writes
# side-effect _cat.parquet/_segm.asdf next to the coadd. We keep it
# on (matching the full-run convention) and then run SourceCatalogStep
# with custom parameters in Phase 5 anyway.
pixi run --manifest-path pixi.toml strun romancal.pipeline.MosaicPipeline \
    "$ASN" \
    --output_dir "$L3_DIR" \
    --steps.resample.in_memory=False \
    > "$LOG" 2>&1

echo "Done. Output: $OUT"
ls -la "$OUT"
