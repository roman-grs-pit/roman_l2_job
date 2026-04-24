#!/usr/bin/env bash
# Phase 4b: build a skycell association for one skycell's L2 files.
#
# skycell_asn emits one JSON per skycell each L2 image overlaps; we
# keep only the JSON for the selected target (Phase 1 pick).
#
# Usage:  scripts/detection/04b_build_asn.sh <SKYCELL_ID>
set -euo pipefail
cd "$(dirname "$0")/../.."

CELL="${1:-}"
[ -n "$CELL" ] || { echo "usage: $0 <SKYCELL_ID>"; exit 1; }

SKYCELL_NAME=$(pixi run python -c "
from astropy.table import Table
t = Table.read('catalogs/detection/selected_skycells.ecsv', format='ascii.ecsv').to_pandas()
row = t[t['SKYCELL_ID'] == $CELL]
if row.empty: raise SystemExit(f'no selected skycell for SKYCELL_ID=$CELL')
print(row['skycell_name'].iloc[0])
")

L2_DIR="output/detection/l2"
ASN_DIR="output/detection/asn"
mkdir -p "$ASN_DIR"

# Collect all L2 files that pointings_sca_to_simulate.ecsv enumerates
# for this skycell.
CAL_LIST=$(pixi run python -c "
from pathlib import Path
from astropy.table import Table
L2 = Path('$L2_DIR').resolve()
t = Table.read('catalogs/detection/pointings_sca_to_simulate.ecsv', format='ascii.ecsv').to_pandas()
r = t[t['SKYCELL_ID']==$CELL]
for _, row in r.iterrows():
    pass_ = int(row['PASS']); seg = int(row['SEGMENT'])
    obs = int(row['OBSERVATION']); vis = int(row['VISIT'])
    exp = int(row['EXPOSURE']); sca = int(row['SCA'])
    fname = (f'r00001010{pass_:02d}{seg:03d}{obs:03d}{vis:03d}_'
             f'{exp:04d}_wfi{sca:02d}_f158_cal.asdf')
    print(L2 / fname)
")
if [ -z "$CAL_LIST" ]; then
    echo "no L2 files matching skycell $CELL in $L2_DIR"; exit 1
fi
N=$(echo "$CAL_LIST" | wc -l | tr -d ' ')
echo "Running skycell_asn on $N L2 files for skycell $CELL (target $SKYCELL_NAME)..."

(
    cd "$ASN_DIR"
    echo "$CAL_LIST" | xargs pixi run --manifest-path ../../../pixi.toml \
        skycell_asn --product-type full --data-release-id "d" -o "sky_${CELL}"
)

# skycell_asn names its output <root>_<drid>_full_<skycellname>_<filter>_asn.json.
TARGET_ASN=$(ls "$ASN_DIR"/sky_${CELL}_*_"${SKYCELL_NAME}"_*_asn.json 2>/dev/null | head -1)
if [ -z "$TARGET_ASN" ]; then
    echo "WARN: no asn matching target skycell $SKYCELL_NAME in $ASN_DIR/"
    ls "$ASN_DIR/sky_${CELL}"_*_asn.json | head
    exit 1
fi
echo "target asn: $TARGET_ASN"
for a in "$ASN_DIR"/sky_${CELL}_*_asn.json; do
    if [ "$a" != "$TARGET_ASN" ]; then
        mv "$a" "$a.nontarget"
    fi
done
echo "kept: $(ls "$ASN_DIR"/sky_${CELL}_*_asn.json 2>/dev/null | wc -l) asn file(s)"
