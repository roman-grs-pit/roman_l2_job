#!/usr/bin/env bash
# Stage 0: one-time setup on a fresh machine.
# - Ensures CRDS + STPSF dirs exist
# - Triggers STPSF data check/download
# - Logs CRDS context for reproducibility
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data catalogs
mkdir -p output/cal output/logs output/smoke/{asn,mosaic,catalog} output/full/{asn,mosaic,catalog}
mkdir -p "${CRDS_PATH:-./crds_cache}"

echo "== STPSF data check =="
# Manually fetch the stpsf 2.2.0-pinned data tarball into the bundle so the
# bundle stays self-contained. The tarball untars to ./stpsf-data/, which is
# what STPSF_PATH (set in pixi activation env) points to.
# https://stpsf.readthedocs.io/en/latest/installation.html
STPSF_DATA_URL="https://stsci.box.com/shared/static/mjst9j056ibf2uph4gxy8qxmi89tjzwk.gz"
if [ -d stpsf-data ] && [ -n "$(ls -A stpsf-data 2>/dev/null)" ]; then
    echo "STPSF data already present at $(pwd)/stpsf-data — skipping download."
else
    echo "Downloading STPSF data (2.2.0) ..."
    curl -fL "$STPSF_DATA_URL" -o /tmp/stpsf-data.tar.gz
    tar -xzf /tmp/stpsf-data.tar.gz -C .
    rm -f /tmp/stpsf-data.tar.gz
    echo "STPSF data installed at $(pwd)/stpsf-data"
fi

echo "== CRDS context =="
pixi run crds list --status 2>&1 | tee output/crds_context.log || true

echo "Setup complete."
