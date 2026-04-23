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

# Collect all L2 files that pointings_sca_to_simulate.ecsv enumerates
# for this pair (Phase 2 expansion picks up every HLWAS pointing whose
# SCA overlaps the selected skycell — not just the pair's 6 dithers).
CAL_LIST=$(pixi run python -c "
from pathlib import Path
from astropy.table import Table
L2 = Path('$L2_DIR').resolve()
t = Table.read('catalogs/detection/pointings_sca_to_simulate.ecsv', format='ascii.ecsv').to_pandas()
r = t[t['PAIR_ID']==$PAIR]
for _, row in r.iterrows():
    fname = (f\"r00001010115{int(row['PASS']):01d}\"
             if int(row['PASS']) in (15, 16) else None)
    # proper: reconstruct the canonical L2 filename
    pass_ = int(row['PASS']); seg = int(row['SEGMENT'])
    obs = int(row['OBSERVATION']); vis = int(row['VISIT'])
    exp = int(row['EXPOSURE']); sca = int(row['SCA'])
    fname = (f'r00001010{pass_:02d}{seg:03d}{obs:03d}{vis:03d}_'
             f'{exp:04d}_wfi{sca:02d}_f158_cal.asdf')
    print(L2 / fname)
")
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
