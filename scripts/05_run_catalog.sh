#!/usr/bin/env bash
# Stage 05: run SourceCatalogStep on each mosaic in this tag, in parallel.
# Outputs per mosaic: <root>_cat.parquet (catalog), <root>_segm.asdf (segmap).
# Lighter per-worker memory than stage 04 (no resample buffer), so 4-way
# or wider is fine. Skip-if-exists guards each catalog.
#
# Usage: 05_run_catalog.sh configs/<tag>.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: $0 configs/<tag>.yaml"; exit 1; }
eval "$(pixi run python scripts/_config.py "$CONFIG")"

MOSAIC_DIR="output/${TAG}/mosaic"
CAT_DIR="output/${TAG}/catalog"
LOG_DIR="output/logs"
PARALLELISM="${PARALLELISM:-$RUN_PARALLELISM}"

[ -d "$MOSAIC_DIR" ] || { echo "missing $MOSAIC_DIR (run 04_run_mosaic.sh $CONFIG first)"; exit 1; }
mkdir -p "$CAT_DIR" "$LOG_DIR"

MOSAICS=("$MOSAIC_DIR"/*_coadd.asdf)
[ -e "${MOSAICS[0]}" ] || { echo "no mosaics in $MOSAIC_DIR"; exit 1; }

echo "Running SourceCatalogStep on ${#MOSAICS[@]} mosaics with ${PARALLELISM} workers..."
START=$(date +%s)

for m in "${MOSAICS[@]}"; do
    base=$(basename "$m" .asdf)
    cat_out="${CAT_DIR}/${base%_coadd}_cat.parquet"
    log="${LOG_DIR}/${base%_coadd}_catalog.log"
    printf '[ -f %q ] || strun romancal.source_catalog.SourceCatalogStep %q --output_dir %q --save_results=true > %q 2>&1\n' \
        "$cat_out" "$m" "$CAT_DIR" "$log"
done | xargs -P "$PARALLELISM" -I{} bash -c '{}'

END=$(date +%s)
echo "Done in $((END - START)) seconds."
