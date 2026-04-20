#!/usr/bin/env bash
# Pre-render hook: bring stage-06 output assets into the report tree so
# Quarto bundles them into _site. The `full/_plots` directory is a stable
# symlink that Quarto dereferences correctly; the depth-distribution PNG
# is a file symlink that Quarto copies as a broken symlink, so we keep it
# as a real file refreshed here on every render.
set -euo pipefail
cd "$(dirname "$0")"

cp -f ../output/full/compare/depth_distribution.png full/_depth_distribution.png
