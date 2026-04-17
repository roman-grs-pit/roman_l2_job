#!/usr/bin/env bash
# Stage 00 (optional, one-off per CRDS context): pre-hydrate the CRDS
# reference cache by calling crds.getreferences() once per SCA (1..18) for
# the config's bandpass. No simulation is run — we just exercise the CRDS
# lookup + download path directly.
#
# Why: parallel stage-02 workers share one CRDS cache. If a worker crashes
# mid-download (e.g. OOM), the partial reference file is left on disk, and
# CRDS won't re-fetch it on retry — subsequent sims crash with an opaque
# "buffer is too small for requested array" error. Hydrating serially up
# front avoids the race entirely.
#
# Scope: stage 02 (romanisim) and stages 04/05 (romancal MosaicPipeline +
# SourceCatalogStep) reference types. See scripts/_hydrate_crds.py for the
# full list and the recipe for extending it when romanisim / romancal
# change what they fetch.
#
# Safe to re-run: CRDS fast-skips files already on disk.
#
# Usage: 00_hydrate_crds.sh configs/<tag>.yaml
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-}"
[ -n "$CONFIG" ] || { echo "usage: $0 configs/<tag>.yaml"; exit 1; }
eval "$(pixi run python scripts/_config.py "$CONFIG")"

pixi run python scripts/_hydrate_crds.py --bandpass "$POINTINGS_BANDPASS"

echo
echo "Run 'pixi run python scripts/00_verify_crds.py' for a size-based sanity check."
