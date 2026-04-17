#!/usr/bin/env bash
# Stage 02: execute the prepared sims.script in parallel.
# Uses xargs -P PARALLELISM (default = nproc).
# BLAS threads are pinned via pixi activation.env (OMP/OPENBLAS/MKL = 1).
# Each line is wrapped with skip-if-exists, so re-runs cheaply pick up partial state.
# Usage: 02_run_sims.sh <smoke|full>
set -euo pipefail

cd "$(dirname "$0")/.."

TAG="${1:-smoke}"
SCRIPT="output/${TAG}/sims.script"
PARALLELISM="${PARALLELISM:-$(nproc)}"

[ -f "$SCRIPT" ] || { echo "missing $SCRIPT (run 01_build_script.sh $TAG first)"; exit 1; }

# Pre-flight: sanity-check the CRDS cache for truncated references left
# behind by a prior crashed run. Set SKIP_CRDS_VERIFY=1 to bypass.
if [ -z "${SKIP_CRDS_VERIFY:-}" ]; then
    if ! pixi run python scripts/00_verify_crds.py; then
        echo
        echo "CRDS cache check failed. Delete the flagged files and re-run,"
        echo "or set SKIP_CRDS_VERIFY=1 to bypass this check."
        exit 1
    fi
fi

N=$(wc -l < "$SCRIPT")
echo "Running ${N} sims with ${PARALLELISM} workers..."
START=$(date +%s)

# Each line is a self-contained shell snippet (with skip-if-exists), so we feed
# them straight to bash via xargs.
xargs -P "$PARALLELISM" -I{} bash -c '{}' < "$SCRIPT"

END=$(date +%s)
echo "Done in $((END - START)) seconds."
