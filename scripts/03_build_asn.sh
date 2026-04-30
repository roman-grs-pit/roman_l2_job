#!/usr/bin/env bash
# Stage 03: build skycell associations from the L2 _cal.asdf files for this tag.
# Selects only the cal files matching pointings_<tag>.ecsv (so smoke and full
# don't pollute each other), then runs skycell_asn --product-type full.
# Usage: 03_build_asn.sh configs/<tag>.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: $0 configs/<tag>.yaml"; exit 1; }
eval "$(pixi run python scripts/_config.py "$CONFIG")"

POINTINGS="pointings_${TAG}.ecsv"
ASN_DIR="${OUTPUT_BASE}/${TAG}/asn"
CAL_DIR="${OUTPUT_BASE}/cal"
[ -f "$POINTINGS" ] || { echo "missing $POINTINGS"; exit 1; }

mkdir -p "$ASN_DIR"

# Repo root captured before we cd into ASN_DIR for skycell_asn output.
REPO_ROOT="$(pwd)"

# _select_cal_files emits paths joined onto --cal-dir. If CAL_DIR is absolute,
# results are absolute; if relative, we'll prepend REPO_ROOT below so skycell_asn
# can resolve them from any cwd.
CAL_LIST=$(pixi run python scripts/_select_cal_files.py "$POINTINGS" \
    --cal-dir "$CAL_DIR" --require-exists)

if [ -z "$CAL_LIST" ]; then
    echo "no L2 cal files found for tag=${TAG} -- run 02_run_sims.sh $CONFIG first"
    exit 1
fi

case "$CAL_DIR" in
    /*) ABS_CAL_LIST="$CAL_LIST" ;;
    *)  ABS_CAL_LIST=$(echo "$CAL_LIST" | awk -v p="$REPO_ROOT/" '{print p $0}') ;;
esac

N=$(echo "$ABS_CAL_LIST" | wc -l | tr -d ' ')
echo "Building associations from ${N} L2 files in $ASN_DIR ..."

cd "$ASN_DIR"
# skycell_asn produces one JSON per skycell, named <root>_<skycell>_asn.json.
echo "$ABS_CAL_LIST" | xargs pixi run --manifest-path "$REPO_ROOT/pixi.toml" \
    skycell_asn --product-type full --data-release-id "p" -o "${TAG}"

echo "Wrote $(ls *_asn.json 2>/dev/null | wc -l | tr -d ' ') association files."
