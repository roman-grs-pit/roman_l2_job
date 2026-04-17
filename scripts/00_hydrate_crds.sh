#!/usr/bin/env bash
# Stage 00 (optional, one-off per CRDS context): pre-hydrate the CRDS
# reference cache by calling crds.getreferences() once per SCA (1..18) for
# the given bandpass. No simulation is run — we just exercise the CRDS
# lookup + download path directly.
#
# Why: parallel stage-02 workers share one CRDS cache. If a worker crashes
# mid-download (e.g. OOM), the partial reference file is left on disk, and
# CRDS won't re-fetch it on retry — subsequent sims crash with an opaque
# "buffer is too small for requested array" error. Hydrating serially up
# front avoids the race entirely.
#
# Scope: stage 02 (romanisim) only. Reference types for stages 04/05
# (photom, area, apcorr, ...) are fetched on-demand by those stages; since
# they run serially today there is no multi-worker race there.
#
# Safe to re-run: CRDS fast-skips files already on disk.
#
# Usage: 00_hydrate_crds.sh [bandpass=F158]
set -euo pipefail
cd "$(dirname "$0")/.."

BANDPASS="${1:-F158}"

pixi run python scripts/_hydrate_crds.py --bandpass "$BANDPASS"

echo
echo "Run 'pixi run python scripts/00_verify_crds.py' for a size-based sanity check."
