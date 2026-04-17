#!/usr/bin/env bash
# Stage status: count expected vs actual outputs for a given tag.
# Usage: 00_status.sh <smoke|full>
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:-smoke}"
POINTINGS="pointings_${TAG}.ecsv"
[ -f "$POINTINGS" ] || { echo "no $POINTINGS"; exit 1; }

NEXP=$(awk '!/^#/ && NF { if (++n>1) print }' "$POINTINGS" | wc -l | tr -d ' ')
EXPECTED_CAL=$(( NEXP * 18 ))
count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l | tr -d ' '; }
ACTUAL_CAL=$(count output/cal '*_cal.asdf')

ASN_DIR="output/${TAG}/asn"
MOSAIC_DIR="output/${TAG}/mosaic"
CAT_DIR="output/${TAG}/catalog"
N_ASN=$(count "${ASN_DIR}" '*.json')
N_MOSAIC=$(count "${MOSAIC_DIR}" '*_coadd.asdf')
N_CAT=$(count "${CAT_DIR}" '*_cat.parquet')

cat <<INFO
== Status: tag=${TAG} ==
pointings:      ${NEXP} exposures
L2 cal files:   ${ACTUAL_CAL} / ${EXPECTED_CAL}  (output/cal/)
asn (skycells): ${N_ASN}                          (${ASN_DIR})
mosaics:        ${N_MOSAIC}                          (${MOSAIC_DIR})
catalogs:       ${N_CAT}                          (${CAT_DIR})
INFO
