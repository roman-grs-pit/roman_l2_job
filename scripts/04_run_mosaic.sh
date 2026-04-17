#!/usr/bin/env bash
# Stage 04: run romancal MosaicPipeline on each association in this tag.
# low_memory=True is set so tight-memory instances don't OOM.
# Skip-if-exists guards each mosaic.
# Usage: 04_run_mosaic.sh configs/<tag>.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: $0 configs/<tag>.yaml"; exit 1; }
eval "$(pixi run python scripts/_config.py "$CONFIG")"

ASN_DIR="output/${TAG}/asn"
MOSAIC_DIR="output/${TAG}/mosaic"
[ -d "$ASN_DIR" ] || { echo "missing $ASN_DIR (run 03_build_asn.sh $CONFIG first)"; exit 1; }
mkdir -p "$MOSAIC_DIR"

ASN_FILES=("$ASN_DIR"/*_asn.json)
[ -e "${ASN_FILES[0]}" ] || { echo "no asn files in $ASN_DIR"; exit 1; }

echo "Running MosaicPipeline on ${#ASN_FILES[@]} associations..."

for asn in "${ASN_FILES[@]}"; do
    base=$(basename "$asn" .json)
    out="${MOSAIC_DIR}/${base%_asn}_coadd.asdf"
    if [ -f "$out" ]; then
        echo "  skip (exists): $out"
        continue
    fi
    echo "  $asn -> $MOSAIC_DIR"
    pixi run strun romancal.pipeline.MosaicPipeline "$asn" \
        --output_dir "$MOSAIC_DIR" \
        --steps.resample.in_memory=False
done

echo "Done."
