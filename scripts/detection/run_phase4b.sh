#!/usr/bin/env bash
# Phase 4b driver: after Phase 1 (selected_skycells.ecsv) and Phase 2
# (pointings_sca_to_simulate.ecsv) are in place, this script:
#
#   1. Regenerates Phase-3 catalogs (cheap).
#   2. Builds one combined L2 sims.script for all skycells; runs 4-way.
#   3. Per skycell: build skycell asn + run MosaicPipeline.
#   4. Per mosaic: Phase-5a detection + Phase-5b kernel sweep
#      + Phase-5c npixels sweep.
#
# Parallelism over L2 sims is global (single xargs across all skycells'
# commands — better load balancing than per-skycell). Mosaic and
# SourceCatalogStep runs are serial (each uses ~15 GB RSS briefly and
# romancal isn't thread-safe across processes sharing CRDS cache).
#
# Usage:  bash scripts/detection/run_phase4b.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

PARALLELISM="${PARALLELISM:-4}"
# Pair IDs are read from pointings_sca_to_simulate.ecsv so that any
# skycell with 0 L2 overlaps (e.g., HLWAS-sparse ecliptic region) is
# automatically skipped — it'd produce no mosaic anyway.
CELL_IDS=$(pixi run --manifest-path pixi.toml python -c "
from astropy.table import Table
t = Table.read('catalogs/detection/pointings_sca_to_simulate.ecsv',
               format='ascii.ecsv').to_pandas()
print(' '.join(str(int(p)) for p in sorted(t['SKYCELL_ID'].unique())))
")
echo "Skycells to process: $CELL_IDS"

# 3. Catalogs
echo ""
echo "=== Phase 3: regenerate truth catalogs ==="
pixi run --manifest-path pixi.toml python scripts/detection/03_generate_catalogs.py

# 4a. Build combined sims script
echo ""
echo "=== Phase 4a: build combined sims.script ==="
ALL_SCRIPT=output/detection/sims_all.script
: > "$ALL_SCRIPT"
for CELL in $CELL_IDS; do
    pixi run --manifest-path pixi.toml python scripts/detection/04a_build_sims.py \
        --cell "$CELL" >/dev/null
    cat "output/detection/sims_${CELL}.script" >> "$ALL_SCRIPT"
done
N=$(wc -l < "$ALL_SCRIPT")
echo "Combined sims.script: $N commands"

# 4b. Run sims
echo ""
echo "=== Phase 4a: running $N L2 sims at ${PARALLELISM}-way parallelism ==="
START=$(date +%s)
pixi run --manifest-path pixi.toml bash -c "\
    xargs -P $PARALLELISM -I{} bash -c '{}' < $ALL_SCRIPT"
echo "L2 sims done in $(( $(date +%s) - START ))s"

# 4c/d: asn + mosaic per skycell
echo ""
echo "=== Phase 4: asn + MosaicPipeline per skycell ==="
for CELL in $CELL_IDS; do
    echo "--- skycell $CELL ---"
    SKY=$(pixi run --manifest-path pixi.toml python -c "
from astropy.table import Table
t = Table.read('catalogs/detection/selected_skycells.ecsv',
               format='ascii.ecsv').to_pandas()
print(t[t['SKYCELL_ID']==$CELL]['skycell_name'].iloc[0])
")
    bash scripts/detection/04b_build_asn.sh "$CELL"
    bash scripts/detection/04c_run_mosaic.sh "$CELL" "$SKY"
done

# 5. Detection — baseline + sweeps
echo ""
echo "=== Phase 5: detection + sweeps per mosaic ==="
for CELL in $CELL_IDS; do
    echo "--- skycell $CELL ---"
    pixi run --manifest-path pixi.toml python \
        scripts/detection/05a_detect_and_efficiency.py --cell "$CELL" || true
    pixi run --manifest-path pixi.toml python \
        scripts/detection/05b_kernel_sweep.py --cell "$CELL" || true
    pixi run --manifest-path pixi.toml python \
        scripts/detection/05c_npixels_sweep.py --cell "$CELL" || true
done

echo ""
echo "=== Phase 6: cross-mosaic summary ==="
pixi run --manifest-path pixi.toml python scripts/detection/06_summary.py || true

echo ""
echo "=== done ==="
