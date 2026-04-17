#!/usr/bin/env bash
# Stage 05: run SourceCatalogStep on each mosaic in this tag.
# Outputs per-mosaic: <root>_cat.parquet (catalog), <root>_segm.asdf (segmap).
# Skip-if-exists by checking the catalog parquet.
# Usage: 05_run_catalog.sh configs/<tag>.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: $0 configs/<tag>.yaml"; exit 1; }
eval "$(pixi run python scripts/_config.py "$CONFIG")"

MOSAIC_DIR="output/${TAG}/mosaic"
CAT_DIR="output/${TAG}/catalog"
[ -d "$MOSAIC_DIR" ] || { echo "missing $MOSAIC_DIR (run 04_run_mosaic.sh $CONFIG first)"; exit 1; }
mkdir -p "$CAT_DIR"

MOSAICS=("$MOSAIC_DIR"/*_coadd.asdf)
[ -e "${MOSAICS[0]}" ] || { echo "no mosaics in $MOSAIC_DIR"; exit 1; }

echo "Running SourceCatalogStep on ${#MOSAICS[@]} mosaics..."

for m in "${MOSAICS[@]}"; do
    base=$(basename "$m" .asdf)
    cat_out="${CAT_DIR}/${base%_coadd}_cat.parquet"
    if [ -f "$cat_out" ]; then
        echo "  skip (exists): $cat_out"
        continue
    fi
    echo "  $m -> $CAT_DIR"
    pixi run strun romancal.source_catalog.SourceCatalogStep "$m" \
        --output_dir "$CAT_DIR" --save_results=true
done

echo "Done."
