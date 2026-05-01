# Running on the Roman AWS SLURM cluster

`README.md` walks through the single-instance (`m5a.2xlarge`, 4-way `xargs`)
path. **This doc covers the SLURM cluster path** used for the acceptance
run (and going forward, anything that doesn't fit on one box). Configs use
the `output_base` field to redirect outputs onto the shared mount; per-stage
SLURM drivers (`scripts/slurm_run_*.sh`) replace the `02_run_sims.sh` /
`04_run_mosaic.sh` / `05_run_catalog.sh` xargs drivers.

This is a working draft written 2026-05-01 mid-acceptance run; expect
refinements once the run lands.

## When to use which path

| Path | Scale | Driver | Outputs |
|---|---|---|---|
| `README.md` (xargs) | smoke / full (≤324 sims) | `02_run_sims.sh` etc. | `output/<tag>/...` (local) |
| this doc (SLURM) | acceptance / scale-up (1000s of sims) | `slurm_run_*.sh` | `${output_base}/<tag>/...` (shared mount) |

Stages 00, 01, 03 are **identical** between the two paths — they're cheap and
serial. Stages 02 / 04 / 05 are where the SLURM drivers diverge.

## Prerequisites

- **Cluster access:** SSH access to a head node with `sbatch` / `squeue` / `scontrol`.
- **Shared mount:** All compute nodes must see the same path for both the
  project repo and the output base. The acceptance run uses
  `/data/npadman/1-Projects/roman_l2_job` (NFS) for the repo and
  `/mnt/roman-science/...` for outputs.
- **pixi env on shared storage:** Install once on the shared mount; compute
  nodes activate from there. (If the env was rsync'd from a different host,
  pixi shebangs may have stale paths — `pixi clean && pixi install` from the
  cluster head node fixes this.)

## Cluster shape (as of 2026-05-01)

| Partition | Nodes | Cores/node | RAM/node | Tasks/node @ 8G | K (5 nodes) |
|---|---|---|---|---|---|
| `mem-lg` | 5 | 8 | 124 GB | 8 (CPU-bound) | 40 |
| `cpun-2xlg` | 5 | 36 | 187 GB | **22** (mem-bound) | **110** |
| `cpu-lg` | 5 | 16 | 62 GB | 7 (mem-bound) | 35 |

A romanisim worker peaks ~7.7 GB RSS (CLAUDE.md). `cpun-2xlg` packs the most
work per dollar — preferred for stage 02.

## Configs

The shipped acceptance config (`configs/acceptance.yaml`) demonstrates the pattern:

```yaml
tag: acceptance
output_base: /mnt/roman-science/grs/acceptance-testing-20260430/imaging/2026-04-30
catalog:
  input: /mnt/roman-science/grs/acceptance-testing-20260430/catalogs_padded/metadata.parquet
  input_units: maggies
  bandpass_col: F158
pointings:
  region: { type: box, ra_min: 8.0, ra_max: 12.0, dec_min: -1.2, dec_max: 1.2 }
  bandpass: F158
  design_depth: 6
run: { parallelism: 4 }   # unused under SLURM but schema-required
```

`output_base` is the only field that distinguishes a cluster config from a
single-instance config. When unset (smoke/full), outputs land at
`output/<tag>/`. When set, outputs land at `${output_base}/<tag>/`.

If you bring your own pointings ECSV (most common for cluster runs because the
per-survey-region selection is bespoke), symlink it as
`${PROJ}/pointings_<tag>.ecsv` before running `01_build_script.sh` — the build
step will use it as-is rather than regenerating from `pointings.region`.

## One-time setup (per cluster / per CRDS context bump)

```bash
cd ${PROJ}
pixi install                                     # solves and installs env
pixi run bash scripts/00_setup.sh                # STPSF data, CRDS scaffolding
pixi run python scripts/00_prepare_catalog.py configs/acceptance.yaml
pixi run bash scripts/00_hydrate_crds.sh configs/acceptance.yaml
pixi run python scripts/00_verify_crds.py        # serial sanity check
```

The CRDS context is **pinned** in `pixi.toml` (`CRDS_CONTEXT=roman_0048.pmap`
as of this writing — see CLAUDE.md for the rationale). To bump: edit the pin,
re-run hydrate + verify, then re-launch.

## Stage 02 — sims

Build the script (serial, fast):

```bash
pixi run bash scripts/01_build_script.sh configs/acceptance.yaml
# -> ${output_base}/acceptance/sims.script (one self-contained line per sim)
```

Submit. The driver takes the config and a few env-var knobs:

```bash
SUBSET=1-900 LINE_OFFSET=0 \
  MAX_CONCURRENT=110 PARTITION=cpun-2xlg MEM=8G TIME=30:00 \
  pixi run bash scripts/slurm_run_sims.sh configs/acceptance.yaml
```

**`MaxArraySize=1001` workaround.** SLURM rejects arrays whose **index** exceeds
1000. For sims.scripts ≥ 1001 lines, submit in chunks of ≤900 with `SUBSET=1-N`
and `LINE_OFFSET` mapping back to script line numbers:

| Chunk | SUBSET | LINE_OFFSET | sims.script lines |
|---|---|---|---|
| 1 | `1-900` | `0` | 1-900 |
| 2 | `1-900` | `900` | 901-1800 |
| 3 | `1-900` | `1800` | 1801-2700 |

Sims are skip-if-exists, so re-running the same SUBSET is free for already-done
lines and only retries the missing ones — recovery is just "compute the line
range that's missing and resubmit with appropriate SUBSET / LINE_OFFSET".

Expected wall time: ~10 min/sim steady-state on `cpun-2xlg`. 2700 sims at K=110
≈ 25 batches ≈ ~4 hr total.

## Stage 03 — asn build

Serial, fast. Same script as the xargs path; reads the `output_base` from
config and writes there.

```bash
pixi run bash scripts/03_build_asn.sh configs/acceptance.yaml
# -> ${output_base}/acceptance/asn/*.json
```

## Skycell filter (between 03 and 04)

Skycells with too few overlapping pointings produce shallow / patchy mosaics
that the catalog step doesn't gain from. Drop them:

```bash
# dry-run (counts kept / rejected, no file moves)
pixi run python scripts/filter_asn_skycells.py \
    --asn-dir ${output_base}/acceptance/asn --min-pointings 6
# apply (moves rejected to ${asn-dir}/rejected/)
pixi run python scripts/filter_asn_skycells.py \
    --asn-dir ${output_base}/acceptance/asn --min-pointings 6 --apply
```

`--min-pointings` is the count of distinct (passno, segment, observation,
visit, exposure) tuples per asn — not the count of SCAs.

## Stage 04 — mosaic

```bash
MAX_CONCURRENT=25 PARTITION=mem-lg MEM=24G TIME=2:00:00 \
  pixi run bash scripts/slurm_run_mosaic.sh configs/acceptance.yaml
# -> ${output_base}/acceptance/mosaic/*_coadd.asdf
```

K=25 (5 nodes × 5 mosaics/node @ 24G) is the default. Mosaics typically take
5–15 min. **Note:** if the surviving asn count exceeds 1000, the `MaxArraySize`
limit applies — `slurm_run_mosaic.sh` does not yet have a `LINE_OFFSET` knob;
patch it the same way as `slurm_run_sims.sh`, or split the manifest with `awk`.

## Stage 05 — catalog

```bash
MAX_CONCURRENT=80 PARTITION=mem-lg MEM=8G TIME=30:00 NPIXELS=16 \
  pixi run bash scripts/slurm_run_catalog.sh configs/acceptance.yaml
# -> ${output_base}/acceptance/catalog/*_cat.parquet, *_segm.asdf
```

`NPIXELS=16` is the detection-study default. Catalogs are typically 30 s – 3 min.
Same `MaxArraySize` caveat as stage 04 if survivor count > 1000.

## Audit trail / where things live

| What | Where |
|---|---|
| Submitted task wrapper | `${output_base}/<tag>/slurm-meta/_<stage>_array_task_<TS>.sh` |
| Submission record | `${output_base}/<tag>/slurm-meta/<stage>-<JobID>.{task.sh,env}` |
| Per-sim stdout/stderr | `${output_base}/logs/<output-name>.log` (stage 02) |
| Per-mosaic / per-catalog log | `${output_base}/<tag>/{mosaic,catalog}/logs/...` |
| SLURM scheduler stdout | `/data/npadman/tmp/slurm-logs/{sims,mosaic,catalog}/<JobID>_<TaskID>.out` |
| `crds_context.log` (active context snapshot) | `output/crds_context.log` |

The `<stage>-<JobID>.env` file in `slurm-meta/` records the exact `SUBSET`,
`LINE_OFFSET`, `PARTITION`, `MEM`, `TIME`, `CONFIG`, and git commit at submit
time — sufficient for reproducing the submission.

## Recovery patterns

**Identify failures.** SLURM drivers leave `${output_base}/logs/<name>.log`
per task; the corresponding output file (`cal/`, `mosaic/`, `catalog/`) is
present iff the task succeeded.

```bash
# Stage 02: missing cal files
for l in ${output_base}/logs/*.log; do
  base=$(basename "$l" .log)
  [ ! -f "${output_base}/cal/$base" ] && echo "$base"
done > /tmp/missing.txt

# Map filenames -> sims.script line numbers
while read base; do
  grep -n -F "$base" ${output_base}/<tag>/sims.script | head -1 | cut -d: -f1
done < /tmp/missing.txt | sort -n
```

**Resubmit the failed range.** Sims are skip-if-exists; the 8 successes mixed
into a contiguous failure range cost ~0 to re-touch:

```bash
SKIP_CRDS_VERIFY=1 SUBSET=60-169 LINE_OFFSET=0 \
  MAX_CONCURRENT=110 PARTITION=cpun-2xlg MEM=8G TIME=30:00 \
  pixi run bash scripts/slurm_run_sims.sh configs/acceptance.yaml
```

## Idiosyncrasies / gotchas

- **`sacct` is broken on this cluster.** `slurmdbd` returns `Connection refused`,
  so `sacct -j <JobID>` doesn't work for timing or MaxRSS. Use `squeue` for live
  state; for post-hoc timing, parse log mtimes (`stat -c '%Y'`) on the per-task
  output files.
- **`SKIP_CRDS_VERIFY=1`.** `00_verify_crds.py` flags references whose size
  deviates >X% from the median of similarly-named files. This produces false
  positives for groups with legitimately-different content (e.g. `matable_0002`
  vs `matable_0003` — different MA tables, different sizes). Skip the pre-flight
  for these false positives, but parse the warning manually first.
- **CRDS context drift.** With `CRDS_MODE=auto`, every cold-start worker asks
  the server for the latest operational context — and 100+ parallel workers
  race to download new mappings if the server has published one since
  hydration. The repo pins `CRDS_CONTEXT=roman_0048.pmap` to avoid this. Bump
  the pin only after re-hydrating.
- **OMP_NUM_THREADS=1 etc.** are set in pixi activation. Without them, each
  worker spawns a thread per core and oversubscribes CPU; cluster nodes are
  not BLAS-aware and won't auto-tune.

## Snapshot at end of run

When the full pipeline lands, snapshot the scripts that produced the run for
provenance:

```bash
mkdir -p ${output_base}/scripts
cp -r scripts/ ${output_base}/scripts/
git rev-parse HEAD > ${output_base}/scripts/git_commit.txt
```

Then commit any remaining uncommitted changes (typically the `slurm-meta/`
audit files are gitignored — that's fine; they live with the outputs on the
shared mount).
