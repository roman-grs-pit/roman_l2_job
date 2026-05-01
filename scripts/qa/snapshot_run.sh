#!/usr/bin/env bash
# Snapshot a completed pipeline run for provenance.
#
# Copies scripts/ + the config into ${output_base}/scripts/, then writes
# ${output_base}/PROVENANCE.md summarizing:
#   - git commit / branch / dirty? at snapshot time
#   - SLURM JobIDs grouped by stage (from slurm-meta/<stage>-<JobID>.env)
#   - output counts per stage dir (cal/, asn/, mosaic/, catalog/)
#   - pixi env summary
#
# Idempotent: re-running overwrites the snapshot.
#
# Usage: scripts/qa/snapshot_run.sh configs/<tag>.yaml
set -euo pipefail
cd "$(dirname "$0")/../.."

CONFIG_IN="${1:?usage: $0 configs/<tag>.yaml}"
# Resolve to absolute so cp later doesn't get confused by relative-to-cwd.
CONFIG=$(realpath -e "$CONFIG_IN" 2>/dev/null || readlink -f "$CONFIG_IN")
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG_IN"; exit 1; }
eval "$(pixi run python scripts/_config.py "$CONFIG")"

PROJ_ABS=$(pwd)
SNAP_SCRIPTS="${OUTPUT_BASE}/scripts"
SNAP_PROV="${OUTPUT_BASE}/PROVENANCE.md"

echo "[snapshot] config: $CONFIG"
echo "[snapshot] tag: $TAG"
echo "[snapshot] output_base: $OUTPUT_BASE"

# 1. Copy scripts/ + the config into output_base
mkdir -p "$SNAP_SCRIPTS"
rsync -a --delete --exclude '__pycache__' "${PROJ_ABS}/scripts/" "$SNAP_SCRIPTS/"
cp "$CONFIG" "$SNAP_SCRIPTS/"
echo "[snapshot] wrote ${SNAP_SCRIPTS}/ ($(find $SNAP_SCRIPTS -type f | wc -l) files)"

# 2. Provenance markdown
GIT_COMMIT=$(git -C "$PROJ_ABS" rev-parse HEAD)
GIT_BRANCH=$(git -C "$PROJ_ABS" rev-parse --abbrev-ref HEAD)
GIT_REMOTE=$(git -C "$PROJ_ABS" config --get remote.origin.url || echo NA)
GIT_DIRTY=$(git -C "$PROJ_ABS" status --porcelain | head -c 1)
TS=$(date -Iseconds)

# Output dir counts. Use find -- ls + glob expansion blows up argv at scale
# (~2700 cal paths) and explodes bash -x output too.
_count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l | tr -d ' '; }
COUNT_CAL=$(_count "${OUTPUT_BASE}/cal" '*.asdf')
COUNT_ASN=$(_count "${OUTPUT_BASE}/${TAG}/asn" '*_asn.json')
COUNT_ASN_REJ=$(_count "${OUTPUT_BASE}/${TAG}/asn/rejected" '*_asn.json')
COUNT_MOSAIC=$(_count "${OUTPUT_BASE}/${TAG}/mosaic" '*_coadd.asdf')
COUNT_CAT=$(_count "${OUTPUT_BASE}/${TAG}/catalog" '*_cat.parquet')
COUNT_SEGM=$(_count "${OUTPUT_BASE}/${TAG}/catalog" '*_segm.asdf')
COUNT_COMPARE=$(_count "${OUTPUT_BASE}/${TAG}/compare/plots" '*.png')

META_DIR="${OUTPUT_BASE}/${TAG}/slurm-meta"

{
    echo "# Run provenance — ${TAG}"
    echo
    echo "_Snapshot written ${TS}_"
    echo
    echo "## Git"
    echo
    echo "- Commit: \`${GIT_COMMIT}\`"
    echo "- Branch: \`${GIT_BRANCH}\`"
    echo "- Remote: \`${GIT_REMOTE}\`"
    [ -n "${GIT_DIRTY}" ] && echo "- **WARNING: working tree dirty at snapshot time**"
    echo
    echo "## Output counts"
    echo
    echo "| Stage | Path | Count |"
    echo "|---|---|---|"
    echo "| 02 cal | \`cal/\` | ${COUNT_CAL} |"
    echo "| 03 asn (kept) | \`${TAG}/asn/\` | ${COUNT_ASN} |"
    echo "| 03 asn (rejected) | \`${TAG}/asn/rejected/\` | ${COUNT_ASN_REJ} |"
    echo "| 04 coadd | \`${TAG}/mosaic/\` | ${COUNT_MOSAIC} |"
    echo "| 05 catalog | \`${TAG}/catalog/\` (\\*_cat.parquet) | ${COUNT_CAT} |"
    echo "| 05 segm | \`${TAG}/catalog/\` (\\*_segm.asdf) | ${COUNT_SEGM} |"
    echo "| 06 compare plots | \`${TAG}/compare/plots/\` | ${COUNT_COMPARE} |"
    echo
    echo "## SLURM submissions"
    echo
    if [ -d "$META_DIR" ]; then
        echo "Per-submission audit files in \`${TAG}/slurm-meta/\`. Aggregate:"
        echo
        echo '```'
        # awk (rather than grep|cut) so missing keys don't abort under set -e.
        # Older env files (pre-LINE_OFFSET patch) lack that key; treat as 0.
        _f() { awk -F= -v k="$1" '$1==k {print $2; exit}' "$2"; }
        for envf in "$META_DIR"/*.env; do
            [ -f "$envf" ] || continue
            STAGE=$(basename "$envf" .env | sed -E 's/-[0-9]+$//')
            JOB=$(_f JOB "$envf")
            ARR=$(_f ARRAY "$envf")
            OFFSET=$(_f LINE_OFFSET "$envf")
            PART=$(_f PARTITION "$envf")
            MEM=$(_f MEM "$envf")
            K=$(_f MAX_CONCURRENT "$envf")
            COMMIT=$(_f GIT_COMMIT "$envf")
            SUB=$(_f SUBMITTED_AT "$envf")
            printf '%-10s job=%-6s arr=%-10s off=%-5s part=%-10s mem=%-5s K=%-4s commit=%s submit=%s\n' \
                "$STAGE" "$JOB" "$ARR" "${OFFSET:-0}" "$PART" "$MEM" "$K" "${COMMIT:0:7}" "$SUB"
        done | sort -k 9
        echo '```'
    else
        echo "_(no slurm-meta dir at \`$META_DIR\`)_"
    fi
    echo
    echo "## Pixi environment"
    echo
    echo '```'
    # Capture first; piping pixi info through head triggers SIGPIPE which
    # under pipefail aborts the script.
    PIXI_INFO=$(pixi info 2>&1 || echo "(pixi info failed)")
    echo "$PIXI_INFO" | sed -n '1,25p'
    echo '```'
    echo
    echo "## Notes"
    echo
    echo "- Scripts at snapshot time are mirrored under \`scripts/\` next to this file."
    echo "- Each stage's \`slurm-meta/\` keeps the submitted task wrapper plus an env file"
    echo "  for byte-level reproducibility of every \`sbatch\` invocation."
    echo "- This file regenerates if \`scripts/qa/snapshot_run.sh\` is re-run on the same config."
} > "$SNAP_PROV"

echo "[snapshot] wrote ${SNAP_PROV}"
