# roman_l2_job — Claude project notes

This bundle runs a Roman WFI L2 → L3 mosaic → source-catalog pipeline on a
fresh Linux x86_64 host. Designed to be tarballed from a Mac, scp'd to AWS,
and run end-to-end with `pixi`.

Companion: `README.md` is the quick-start; this file is the deeper context.

## Science goal

The point of this pipeline isn't the L2 / mosaic / catalog plumbing on its own —
it's a **recovery test**. After stage 05 we compare the recovered
`SourceCatalogStep` catalog against the input catalog (`data/metadata.parquet`,
converted to maggies by `00_prepare_catalog.py`) to measure how well the
pipeline recovers photometry and completeness as a function of magnitude.
That comparison is the next step after the full run finishes.

## Provenance & sizing

Built 2026-04-16 from the parent `roman-pipelines/` project on macOS arm64,
shipped to a Yale Spinup AWS Linux instance. Sizing decisions baked in:

- **Instance:** `m5a.2xlarge` (8 vCPU, 32 GB, ~$0.34/hr). AMD EPYC; ~10% slower
  per-core than m5 but ~10% cheaper. 4-way parallel (see Parallelism section —
  the 8 vCPUs are SMT threads over 4 physical cores); going to 4xlarge gives
  only ~1.7× speedup (EBS throughput / memory bandwidth ceiling), so 2xlarge
  wins on total $.
- **Disk:** 250 GB gp3.
- **Wall time / cost:** smoke ~2 hr / ~$0.70 (stage 02 ~120 min at 4-way,
  stages 04–05 add ~5 hr for 140 skycells). Full run stage 02 ~11 hr / ~$4
  at 4-way.
- **`--product-type full`** in stage 03 chosen so each skycell yields exactly one
  deep coadd combining all overlapping exposures from all visits/passes.
  `pass` (×2 sets) or `visit` (×6 sets) are alternatives if comparison products
  are wanted; would also bloat disk.
- **Smoke = 1 visit** (PASS 15 / SEGMENT 521 / VISIT 6) chosen as the smallest
  unit that exercises every pipeline stage end-to-end, including multi-exposure
  coadd within a visit.

## Pipeline shape

Every stage (except `00_setup.sh` and `00_verify_crds.py`) takes a YAML
config path as its single argument — `configs/<tag>.yaml`. The config
carries tag, catalog path + units, pointings region (cone or box),
bandpass, visit restrictions, and parallelism. `scripts/_config.py` is
the loader; bash stages source exports via `eval "$(... _config.py
$CONFIG)"`.

| Stage | Script | Reads | Writes |
|---|---|---|---|
| setup | `00_setup.sh` | — | STPSF data, CRDS dir, `crds_context.log` |
| catalog prep | `00_prepare_catalog.py <config>` | `catalog.input` from config | `catalogs/sources.parquet` (maggies) |
| hydrate CRDS | `00_hydrate_crds.sh <config>` | CRDS server | populated `crds_cache/` |
| verify CRDS | `00_verify_crds.py` | `crds_cache/` | stdout (+nonzero on outliers) |
| 01 build | `01_build_script.sh <config>` | pointings (regenerated from config if missing), `sources.parquet` | `output/<tag>/sims.script` |
| 02 sims | `02_run_sims.sh <config>` | sims.script | `output/cal/*_cal.asdf`, `output/logs/*.log` |
| 03 asn | `03_build_asn.sh <config>` | matching cal files | `output/<tag>/asn/*.json` |
| 04 mosaic | `04_run_mosaic.sh <config>` | asn jsons | `output/<tag>/mosaic/*_coadd.asdf` |
| 05 catalog | `05_run_catalog.sh <config>` | mosaics | `output/<tag>/catalog/*_cat.parquet`, `*_segm.asdf` |
| 06 compare | `06_compare_catalog.py <config>` | sources.parquet, coadds, recovered catalogs | `output/<tag>/compare/{plots/*.png, summary.csv}` |

Stage 02 runs `00_verify_crds.py` as a pre-flight before the parallel xargs;
set `SKIP_CRDS_VERIFY=1` to bypass it.

Tag examples: `smoke` (1 visit, 54 sims) and `full` (6 visits, 324 sims).
The two shipped configs share the same `pointings.region` but differ on
`only_pass/segment/visit`. Smoke and full **share** `output/cal/` (filenames
are unique per visit/exposure/SCA), but **separate** their asn/mosaic/catalog
outputs so a deeper full-run mosaic never collides with a sparser smoke-run
mosaic of the same skycell.

Every stage is idempotent (skip-if-exists). Re-run any stage to pick up
partial state cheaply.

## Why `make_stack.py` is vendored

`romanisim-make-stack` (0.13.1 and main, as of 2026-04-16) crashes without
`--apt`. The offending line in upstream:

```python
apt_metadata = None if not args.apt else ris.parse_apt_file(args.apt)
program = '00001' if apt_metadata is None else apt_metadata['observation']['program']
apt_metadata['observation']['program'] = int(apt_metadata['observation']['program'])
```

The third line unconditionally dereferences `apt_metadata` even when it is
`None` (the no-APT branch the line above just established). We don't have an
APT file, so the script is vendored at `scripts/make_stack.py` with that
statement guarded by `if apt_metadata is not None:`. The fix is 3 lines; the
rest of the vendored file is a verbatim copy of upstream `romanisim-make-stack`.

If/when upstream merges a fix, we can delete `scripts/make_stack.py` and call
`romanisim-make-stack` from the installed env directly.

## Why the seed is rewritten per line

`romanisim-make-stack --rng_seed N --make_script` bakes the same `N` into
every generated `romanisim-make-image` line — meaning every exposure draws
the same noise realization, and dithers don't average down. The fix lives in
`scripts/_postprocess_sims.py`: it replaces the constant seed with
`zlib.crc32(basename) & 0x7FFFFFFF`, giving every (visit, exposure, SCA) its
own deterministic, reproducible seed.

## Environment

- pixi-managed; pinned to **linux-64** + Python 3.12.
- `fftw` from conda must resolve before `romanisim`/GalSim — already in `pixi.toml [dependencies]`.
- `pyarrow` is explicit so `Table.read('*.parquet')` works.
- Activation env sets:
  - `CRDS_PATH=$PIXI_PROJECT_ROOT/crds_cache` (auto-created; references download on demand)
  - `STPSF_PATH=$PIXI_PROJECT_ROOT/stpsf-data` (populated by `00_setup.sh` via
    manual download of the version-pinned tarball — STPSF's own
    `auto_download_stpsf_data()` ignores `STPSF_PATH` and overwrites it with
    `~/data/stpsf-data`, so we use the manual download path documented at
    https://stpsf.readthedocs.io/en/latest/installation.html)
  - `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`
    (so 8 parallel workers don't each spawn 8 BLAS threads)

## Parallelism

`02_run_sims.sh` uses `xargs -P "${PARALLELISM:-$(nproc)}"`. **Do not default to
`nproc` on `m5a.2xlarge`** — set `PARALLELISM=4` explicitly. The 8 vCPUs are
AMD SMT threads over 4 physical cores, so 8-way oversubscribes on the
FFT/numpy-heavy rendering path. 8-way also OOMs: measured peak per-worker RSS
is ~7.7 GB, so 8 workers spike to ~62 GB on a 32 GB box and get killed.

Measured at PARALLELISM=4:
- ~8.5 min per sim steady-state (`--psftype stpsf`).
- 4-worker combined peak RSS ~23 GB (fits in 32 GB with headroom).
- Full run: **~11 hr, ~$4** for stage 02 sims.

Going to 16 vCPU (m5a.4xlarge, 64 GB) and 8-way gets ~1.7× on stage 02 — EBS
throughput and memory bandwidth are the next bottleneck. Cheaper to stay on
2xlarge unless wall time matters more than $.

## CRDS first run

The first sim downloads several GB of F158 reference files to `$CRDS_PATH`.
Allow ~10–15 extra minutes. The full set is then cached for all subsequent
sims. Context is logged at `output/crds_context.log` for reproducibility.

## Disk budget

Sized for 250 GB gp3. Heavy hitters:

- 324 L2 cal files × ~200 MB = **~65 GB**
- L3 mosaics (`--product-type full`, ~25 skycells × 668 MB) = **~20–30 GB**
- CRDS cache (F158): **~5–10 GB**
- pixi env: ~5 GB
- Headroom: ~15 GB

Smoke alone: ~15 GB total.

## Pointings file

`pointings_full.ecsv` is a filtered subset of `catalogs/HLWAS.sim.ecsv` —
simulated Roman High-Latitude Wide-Area Survey pointings covering the full
survey footprint. The filter script is `scripts/filter_pointings.py`. The
exact invocation that produced the in-repo `pointings_full.ecsv`:

```
pixi run python scripts/filter_pointings.py \
    -i catalogs/HLWAS.sim.ecsv -o pointings_full.ecsv \
    --ra 10 --dec 0 --radius 0.5 --bandpass F158
```

Selection:

- `BANDPASS == 'F158'`
- at least one exposure in the visit falls within 0.5° of (RA=10, Dec=0)
- then include every exposure (all 3 dithers) of matching visits

Result:

- 6 visits × 3 dithers = 18 exposures in `pointings_full.ecsv`
- 2 passes (15 & 16), SEGMENTs 490 & 521, covering ~1.2° × 0.5° near the equator
- All MA_TABLE_NUMBER=1007, EXPOSURE_TIME=107.52s, PA=0, TARGET=Wide-Field2

`pointings_smoke.ecsv` adds `--pass 15 --segment 521 --visit 6` to carve out
a single visit = 3 rows.

When scaling up (larger footprint, more filters, NERSC), re-run the filter
with a wider `--radius` / different `--ra`/`--dec` / different `--bandpass`.
The column schema is fixed by `HLWAS.sim.ecsv`; downstream stages consume
those columns by name.

## Output filename convention (from `make_stack`)

```
r{program=00001}{plan:02d=01}{passno:03d}{segment:03d}{observation:03d=001}{visit:03d}_
  {exposure:04d}_wfi{sca:02d}_{bandpass.lower()=f158}_cal.asdf
```

Smoke files: `r0000101015521001006_000{1,2,3}_wfi{01..18}_f158_cal.asdf`.

## Troubleshooting

- **Sim fails partway through stage 02**: re-run `02_run_sims.sh <tag>`. Existing
  files are skipped; missing ones are retried. Failed sim logs live at
  `output/logs/<output_filename>.log`.
- **Out of memory in stage 04**: drop `PARALLELISM` for stage 02 (workers leave
  more room) and re-run; or move to `m5a.4xlarge`. MosaicPipeline already runs
  with `in_memory=False` in `04_run_mosaic.sh`.
- **No L2 cal files in `03_build_asn.sh`**: check `output/cal/` and
  `output/logs/`. Sims may have all failed (likely a CRDS or STPSF setup issue).
- **STPSF data missing on import**: `pixi run python -c "import stpsf; stpsf.utils.download_data()"`.
- **CRDS connection errors**: confirm `echo $CRDS_SERVER_URL` resolves and the
  EC2 SG allows outbound HTTPS.
- **`TypeError: buffer is too small for requested array` inside
  `asdf/tags/core/ndarray.py`** (usually hit in `gather_reference_data`):
  a CRDS reference file on disk is **truncated** — a prior run was OOM-killed
  or interrupted mid-download. CRDS does not retry broken partial files. Run
  `pixi run python scripts/00_verify_crds.py` to find the outliers; delete
  them and re-run `scripts/00_hydrate_crds.sh`, which re-fetches missing
  references serially (no parallel-download race). Stage 02 runs the verify
  step as a pre-flight, so fresh corruption is caught before xargs spins up.
- **Stage 04 / 05 downloads references the first time**. CRDS hydration
  currently covers only stage 02 (romanisim). `MosaicPipeline` and
  `SourceCatalogStep` fetch their own reference types on first use. Since
  stages 04 and 05 run serially today there is no multi-worker race, but
  a crash mid-download has the same partial-file risk — re-run
  `00_verify_crds.py` afterward if in doubt.

## When extending this bundle

- New filter (e.g. F184): re-run `scripts/filter_pointings.py` with
  `--bandpass F184`, write `pointings_<filter>.ecsv`, and ensure the input
  catalog has a column with the bandpass name in maggies.
- New input catalog: scp it into `data/`, re-run `00_prepare_catalog.py` with
  `--input <path>`. Downstream is unchanged.
- New region: re-run `scripts/filter_pointings.py` with different
  `--ra`/`--dec`/`--radius` and regenerate the pointings file.

## Roadmap

Two directions of travel, in rough priority:

1. **Recovery test (the science goal).** `scripts/06_compare_catalog.py`
   does the per-skycell comparison: crossmatches input `sources.parquet`
   against recovered `*_cat.parquet` for the top-N deepest skycells,
   emits a 5-panel PNG per skycell (completeness, flux residual, PSF vs
   aperture, FP rate, astrometric residual) plus a `summary.csv`. v1 is
   per-skycell only and hard-codes Kron mag as the reference recovered
   mag; pooled-across-skycells and type-confusion would be v2.
2. **Scale-up to full metadata.parquet.** Run over a much larger footprint
   (wider `filter_pointings.py --radius`, more pointings, eventually
   multiple filters) on a bigger AWS instance or NERSC Perlmutter. At that
   scale, stage 02 parallelism and stages 04/05 serial-strun overhead both
   become the bottleneck; expect this to require revisiting the xargs model
   (maybe SLURM array jobs, maybe a workflow runner) rather than just
   bumping `PARALLELISM`.
