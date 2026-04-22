#!/usr/bin/env bash
# Phase 4: build a skycell association for one pair's L2 files.
#
# skycell_asn emits one JSON per skycell each L2 image overlaps; we
# keep only the JSON for the pair's selected skycell (Phase 2 pick).
#
# Usage:  scripts/detection/04b_build_asn.sh 11119
set -euo pipefail
cd "$(dirname "$0")/../.."

PAIR="${1:-}"
[ -n "$PAIR" ] || { echo "usage: $0 <PAIR_ID>"; exit 1; }

# Find the pair's selected skycell name from the Phase-2 index
SKYCELL_NAME=$(pixi run python -c "
from astropy.table import Table
t = Table.read('catalogs/detection/selected_skycells.ecsv', format='ascii.ecsv').to_pandas()
row = t[t['PAIR_ID'] == $PAIR]
if row.empty: raise SystemExit(f'no selected skycell for pair $PAIR')
print(row['skycell_name'].iloc[0])
")

L2_DIR="output/detection/l2"
ASN_DIR="output/detection/asn"
mkdir -p "$ASN_DIR"

# Pick the L2 files matching this pair's (PASS, SEGMENT, VISIT).
SEGMENT=$(pixi run python -c "
from astropy.table import Table
t = Table.read('catalogs/detection/selected_pointings.ecsv', format='ascii.ecsv').to_pandas()
print(int(t[t['PAIR_ID']==$PAIR]['SEGMENT'].iloc[0]))
")
VISIT=$(pixi run python -c "
from astropy.table import Table
t = Table.read('catalogs/detection/selected_pointings.ecsv', format='ascii.ecsv').to_pandas()
print(int(t[t['PAIR_ID']==$PAIR]['VISIT'].iloc[0]))
")
printf -v PAT 'r000010101[56]%03d001%03d_*_wfi*_f158_cal.asdf' "$SEGMENT" "$VISIT"
# Collect matching L2 files into an absolute-path list
CAL_LIST=$(ls "$L2_DIR"/$PAT 2>/dev/null | xargs -I{} readlink -f {} | sort -u)
if [ -z "$CAL_LIST" ]; then
    echo "no L2 files matching pair $PAIR in $L2_DIR"; exit 1
fi
N=$(echo "$CAL_LIST" | wc -l | tr -d ' ')
echo "Running skycell_asn on $N L2 files for pair $PAIR (target skycell $SKYCELL_NAME)..."

# skycell_asn writes to cwd; change to ASN_DIR to collect outputs
(
    cd "$ASN_DIR"
    echo "$CAL_LIST" | xargs pixi run --manifest-path ../../../pixi.toml \
        skycell_asn --product-type full --data-release-id "d" -o "pair_${PAIR}"
)

# Retain only the association for the target skycell; drop others.
TARGET_ASN=$(ls "$ASN_DIR"/pair_${PAIR}_*_"${SKYCELL_NAME}"_asn.json 2>/dev/null | head -1)
if [ -z "$TARGET_ASN" ]; then
    echo "WARN: no asn matching target skycell $SKYCELL_NAME in $ASN_DIR/"
    ls "$ASN_DIR/pair_${PAIR}"_*_asn.json | head
    exit 1
fi
echo "target asn: $TARGET_ASN"
# Archive any non-target asns so future runs only see the one we want.
for a in "$ASN_DIR"/pair_${PAIR}_*_asn.json; do
    if [ "$a" != "$TARGET_ASN" ]; then
        mv "$a" "$a.nontarget"
    fi
done
echo "kept: $(ls "$ASN_DIR"/pair_${PAIR}_*_asn.json 2>/dev/null | wc -l) asn file(s)"
