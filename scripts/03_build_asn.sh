#!/usr/bin/env bash
# Stage 03: build skycell associations from the L2 _cal.asdf files for this tag.
# Selects only the cal files matching pointings_<tag>.ecsv (so smoke and full
# don't pollute each other), then runs skycell_asn --product-type full.
# Usage: 03_build_asn.sh <smoke|full>
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:-smoke}"
POINTINGS="pointings_${TAG}.ecsv"
ASN_DIR="output/${TAG}/asn"
[ -f "$POINTINGS" ] || { echo "missing $POINTINGS"; exit 1; }

mkdir -p "$ASN_DIR"

CAL_LIST=$(pixi run python scripts/_select_cal_files.py "$POINTINGS" \
    --cal-dir output/cal --require-exists)

if [ -z "$CAL_LIST" ]; then
    echo "no L2 cal files found for $TAG -- run 02_run_sims.sh $TAG first"
    exit 1
fi

N=$(echo "$CAL_LIST" | wc -l | tr -d ' ')
echo "Building associations from ${N} L2 files in $ASN_DIR ..."

cd "$ASN_DIR"
# skycell_asn produces one JSON per skycell, named <root>_<skycell>_asn.json.
# We feed absolute cal paths so it can find them from any cwd.
ABS_CAL_LIST=$(echo "$CAL_LIST" | awk -v p="$PWD/../../../" '{print p $0}')
echo "$ABS_CAL_LIST" | xargs pixi run --manifest-path ../../../pixi.toml \
    skycell_asn --product-type full --data-release-id "p" -o "${TAG}"

echo "Wrote $(ls *_asn.json 2>/dev/null | wc -l | tr -d ' ') association files."
