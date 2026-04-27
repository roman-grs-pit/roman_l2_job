#!/usr/bin/env bash
# Archive the current output/ tree + configs + input catalogs to S3 for
# long-term storage. Idempotent: `aws s3 sync` skips objects whose size
# and mtime already match on the remote side.
#
# Usage:
#     scripts/archive_to_s3.sh [label]
#
# `label` is the second-level S3 prefix under the bucket and defaults to
# the current UTC date (YYYY-MM-DD). Override with a specific string if
# you want to stamp a particular milestone (e.g. "2026-04-20-full-run").
#
# Environment overrides:
#     ARCHIVE_BUCKET   default: spinup-003131-romanisim-l3
#     AWS_PROFILE      default: spinup-003131-romanisim-l3
#     STORAGE_CLASS    default: STANDARD_IA
#
# Prerequisites:
#     - aws CLI available
#     - ~/.aws/credentials has an entry for $AWS_PROFILE
#     - that profile has s3:Put/Get/List on the bucket
#
# The asn/ directory (lots of tiny JSONs) is tarred before sync to avoid
# the STANDARD_IA 128-KB-per-object minimum-billable-size charge on 320
# ~3 KB files. Other small files (catalog parquets ~100-200 KB each) are
# left individual for easy per-skycell retrieval.

set -euo pipefail
cd "$(dirname "$0")/.."

BUCKET="${ARCHIVE_BUCKET:-spinup-003131-romanisim-l3}"
PROFILE="${AWS_PROFILE:-spinup-003131-romanisim-l3}"
LABEL="${1:-$(date -u +%Y-%m-%d)}"
PREFIX="roman_l2_job/${LABEL}/full-run"
STORAGE_CLASS="${STORAGE_CLASS:-STANDARD_IA}"

export AWS_PROFILE="$PROFILE"

cat <<INFO
archive_to_s3:
  bucket        : $BUCKET
  prefix        : $PREFIX
  aws profile   : $PROFILE
  storage class : $STORAGE_CLASS
INFO
echo

# Quick auth probe — fail fast if the profile can't reach the bucket.
aws s3api head-bucket --bucket "$BUCKET" >/dev/null || {
    echo "error: cannot access s3://$BUCKET with profile '$PROFILE'" >&2
    echo "       (check ~/.aws/credentials + bucket policy)" >&2
    exit 1
}

# Tar the asn directory if it's present. Always regenerate: tar + gzip over
# ~1.5 MB is instant, and it's easier than tracking staleness.
if [ -d output/full/asn ]; then
    echo "packing output/full/asn/ -> output/full/asn.tar.gz"
    tar -C output/full -czf output/full/asn.tar.gz asn
fi

# Sync data products. Exclude the individual asn/*.json — tarball takes
# their place. Also exclude pixi env, crds cache, stpsf data (all
# reproducible on any host and not worth archiving).
aws s3 sync output/ "s3://$BUCKET/$PREFIX/output/" \
    --storage-class "$STORAGE_CLASS" \
    --exclude 'full/asn/*.json' \
    --exclude 'smoke/asn/*.json' \
    --no-progress

aws s3 sync configs/ "s3://$BUCKET/$PREFIX/configs/" \
    --storage-class "$STORAGE_CLASS" \
    --no-progress

# Input artefacts that live outside output/ but matter for restore.
for f in pointings_full.ecsv pointings_smoke.ecsv \
         catalogs/sources.parquet catalogs/HLWAS.sim.ecsv \
         data/metadata.parquet; do
    [ -f "$f" ] || continue
    aws s3 cp "$f" "s3://$BUCKET/$PREFIX/$f" \
        --storage-class "$STORAGE_CLASS" --no-progress
done

# Provenance stamps at the top of the prefix.
git rev-parse HEAD | aws s3 cp - "s3://$BUCKET/$PREFIX/GIT_COMMIT" \
    --storage-class "$STORAGE_CLASS"
if [ -f output/crds_context.log ]; then
    aws s3 cp output/crds_context.log "s3://$BUCKET/$PREFIX/CRDS_CONTEXT" \
        --storage-class "$STORAGE_CLASS"
fi
date -u +"%Y-%m-%dT%H:%M:%SZ" | aws s3 cp - "s3://$BUCKET/$PREFIX/ARCHIVED_AT" \
    --storage-class "$STORAGE_CLASS"

echo
echo "done. archive lives at:"
echo "  s3://$BUCKET/$PREFIX/"
echo "to restore (example): aws s3 sync s3://$BUCKET/$PREFIX/ ."
