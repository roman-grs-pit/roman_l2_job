# Tutorial: simulating L2 calibrated images and L3 mosaics

This walks through what the pipeline in this repo actually does — what goes
in, what comes out of each stage, and why the steps look the way they do.
If you just want to run it, see `README.md`. If you want to operate it on a
big instance or debug a failure, see `CLAUDE.md`. This file is for reading.

## What you're building

Starting from a **catalog of simulated sources** and a **list of Roman WFI
pointings**, the pipeline produces:

- **L2 cal files** — per-exposure, per-SCA calibrated images (18 SCAs per
  exposure), one `.asdf` file each.
- **L3 mosaic coadds** — per-skycell deep images that combine every L2
  exposure overlapping that skycell.
- **L3 source catalogs** — photometry + segmentation map per mosaic.

The whole point is to measure how well the detection + photometry pipeline
recovers the input catalog. L2s and L3s are means to that end.

## Inputs

You need two things before running anything:

1. **A source catalog** at `data/metadata.parquet`. Columns: at minimum `ra`,
   `dec`, `type` (star/galaxy), and a per-bandpass magnitude column (e.g.
   `F158`). Magnitudes, not fluxes.
2. **A pointings file** at `pointings_full.ecsv` (and optionally
   `pointings_smoke.ecsv` for a 1-visit shakedown). One row per exposure,
   schema described in `catalogs/HLWAS.sim.ecsv`.

Both are already in the repo for the bundled smoke/full runs. Swapping either
is described in `CLAUDE.md` → "When extending this bundle".

## The pipeline at a glance

```
                  data/metadata.parquet                catalogs/HLWAS.sim.ecsv
                  (input sources, mag)                 (survey pointings)
                           │                                   │
                           ▼                                   ▼
                  00_prepare_catalog.py              scripts/filter_pointings.py
                  catalogs/sources.parquet           pointings_<tag>.ecsv
                  (maggies)                          (one sub-region)
                           │                                   │
                           └─────────────┬─────────────────────┘
                                         ▼
                           01_build_script.sh   →  output/<tag>/sims.script
                                         │        (one romanisim cmd per SCA)
                                         ▼
                           02_run_sims.sh       →  output/cal/*_cal.asdf  (L2)
                                         │
                                         ▼
                           03_build_asn.sh      →  output/<tag>/asn/*_asn.json
                                         │        (one JSON per skycell)
                                         ▼
                           04_run_mosaic.sh     →  output/<tag>/mosaic/*_coadd.asdf (L3)
                                         │
                                         ▼
                           05_run_catalog.sh    →  output/<tag>/catalog/*_cat.parquet
                                                              */_segm.asdf
```

`<tag>` is `smoke` or `full`. The L2 cal files live in the shared `output/cal/`
directory — smoke and full share whatever cal files overlap, saving compute.

## Stage walkthrough

Every stage takes a YAML config (`configs/<tag>.yaml`) that fixes the
source catalog path, input units, pointings region, filter, visit
restrictions, and parallelism. `configs/smoke.yaml` and `configs/full.yaml`
ship with the repo; copy one to start a new run.

### Stage 00a — prepare the source catalog

```
pixi run python scripts/00_prepare_catalog.py configs/smoke.yaml
```

- **Reads:** `catalog.input` from the config.
- **Writes:** `catalogs/sources.parquet`, always in maggies.
- **What happens:** if `catalog.input_units: mag`, converts the column
  named `catalog.bandpass_col` from AB magnitudes to maggies (`10**(-mag/2.5)`).
  If `input_units: maggies`, passes through unchanged.
- **Why:** `romanisim-make-image` takes fluxes, not magnitudes. Keeping the
  unit conversion here means downstream stages are unit-agnostic.

### Stage 00b — pick a region of sky (pointings file)

Regeneration is automatic: stage 01 calls `filter_pointings.py` if
`pointings_<tag>.ecsv` is missing. You can also run it explicitly:

```
pixi run python scripts/filter_pointings.py configs/smoke.yaml
```

- **Reads:** `catalogs/HLWAS.sim.ecsv` (full survey pointings) + config.
- **Writes:** `pointings_<tag>.ecsv` (gitignored — the config is the
  source of truth).
- **What happens:** filters to rows matching `pointings.bandpass`, keeps
  visits whose exposures fall inside `pointings.region` (cone or box),
  then applies optional `only_pass/segment/visit` restrictions. The filter
  keeps every exposure of a matching visit (all 3 dithers).
- **Box example:** set `region.type: box` and `ra_min/ra_max/dec_min/dec_max`
  instead of `ra/dec/radius_deg`.

### Stage 00c (optional but recommended) — hydrate the CRDS cache

```
pixi run bash scripts/00_hydrate_crds.sh configs/smoke.yaml
```

- **Reads:** CRDS server (on first run); reads only the on-disk cache on re-runs.
- **Writes:** populated `crds_cache/references/roman/wfi/*.asdf`.
- **What happens:** calls `crds.getreferences()` once per (SCA, bandpass)
  pair for the 14 reference types our pipeline uses — 10 from romanisim
  (stage 02) plus `apcorr`, `epsf`, `matable`, `skycells` from romancal
  (stages 04/05). No simulation is run — pure CRDS lookup + download.
  Cold-cache cost: whatever it takes to download a few GB of refs.
  Warm-cache cost: under a second.
- **Why:** stage 02 runs several workers in parallel, all sharing one
  CRDS cache. If a worker is OOM-killed mid-download, the partial
  reference file confuses every subsequent sim with a cryptic "buffer is
  too small" crash. Hydrating serially up front rules out that class.
- **Verify:** `pixi run python scripts/00_verify_crds.py` scans the cache
  for size outliers per reference type. Safe to re-run any time; stage 02
  runs it as a pre-flight.

### Stage 01 — build the sims command script

```
pixi run bash scripts/01_build_script.sh configs/smoke.yaml
```

- **Reads:** `pointings_<tag>.ecsv` (regenerated from config if missing),
  `catalogs/sources.parquet`.
- **Writes:** `output/smoke/sims.script` — one line per (exposure × SCA)
  combination. 3 exposures × 18 SCAs = 54 lines for smoke.
- **What happens:** `scripts/make_stack.py` (a vendored, patched
  `romanisim-make-stack`) expands the pointings file into
  `romanisim-make-image` invocations. Each line:
  - writes to `output/cal/r<visit-id>_<exposure>_wfi<sca>_f158_cal.asdf`
  - uses `--psftype stpsf` (realistic per-SCA PSF)
  - uses `--usecrds` (real calibration reference files)
  - has a **per-line deterministic seed** — `_postprocess_sims.py` rewrites
    `--rng_seed` with `crc32(basename)` so each exposure draws its own noise
    realization (upstream `make_stack` bakes the same seed into every line,
    which defeats dither averaging)
  - is wrapped in `[ -f <out> ] || { ... > log 2>&1; }` so re-running skips
    already-finished sims and keeps a per-sim log

The raw (pre-postprocess) script is kept at `output/<tag>/sims_raw.script`
for reference.

### Stage 02 — run the sims (L2)

```
pixi run bash scripts/02_run_sims.sh configs/smoke.yaml
# PARALLELISM=<N> to override the config's run.parallelism
```

- **Reads:** `sims.script`.
- **Writes:** `output/cal/*_cal.asdf` (L2 files, ~200 MB each),
  `output/logs/*.log`.
- **What happens:** `xargs -P` runs the lines in parallel. Each line calls
  `romanisim-make-image`, which:
  1. Reads the source catalog, projects sources through the WCS for that
     SCA, renders them with the STPSF point-spread function.
  2. Applies the MA table (`MA_TABLE_NUMBER=1007` for our pointings =
     High-Latitude-Wide MA table, 5 resultants of 107.52 s total).
  3. Reads CRDS reference files (saturation, linearity, flat, readnoise,
     dark, distortion, …), applies them, writes the calibrated L2 asdf.
- **First run:** CRDS references (~5–10 GB) stream down on demand. Expect
  an extra 10–15 min of wall time on the first sim per SCA before compute
  kicks in.

Output naming (from `make_stack`):

```
r{program:05d=00001}{plan:02d=01}{pass:03d}{segment:03d}{obs:03d=001}{visit:03d}_
  {exposure:04d}_wfi{sca:02d}_{bandpass.lower()}_cal.asdf
```

So for smoke you end up with
`r0000101015521001006_000{1,2,3}_wfi{01..18}_f158_cal.asdf`.

### Stage 03 — build skycell associations

```
pixi run bash scripts/03_build_asn.sh configs/smoke.yaml
```

- **Reads:** the L2 cal files that match `pointings_smoke.ecsv` (selected
  by `_select_cal_files.py` so smoke asn never picks up full-only cals).
- **Writes:** `output/smoke/asn/*_asn.json` — one JSON per skycell.
- **What a skycell is:** the Roman survey tessellates the sky into fixed
  skycells (see `romancal.skycell`). Each skycell has a known WCS and
  pixel grid. An L2 exposure overlaps some number of skycells; `skycell_asn`
  figures out, for each skycell, which L2 exposures contribute. That
  mapping is the "association".
- **`--product-type full`:** one coadd per skycell that combines every
  contributing exposure from every visit/pass. Alternatives are `pass`
  (one product per skycell per pass) or `visit` (even more granular) —
  useful for comparison but eats disk.

For smoke (54 cal files over a ~1° patch) you get ~140 skycells.

### Stage 04 — coadd into L3 mosaics

```
pixi run bash scripts/04_run_mosaic.sh configs/smoke.yaml
```

- **Reads:** each `*_asn.json`.
- **Writes:** `output/smoke/mosaic/<skycell>_coadd.asdf`.
- **What happens:** runs `romancal.pipeline.MosaicPipeline`, which:
  1. Reads the L2 exposures listed in the asn.
  2. Resamples them onto the skycell's pixel grid (drizzle-style).
  3. Combines them with inverse-variance weighting.
  4. Writes a single coadded `.asdf` with image, error, weight, context,
     plus metadata.
- **Memory:** `--steps.resample.in_memory=False` is passed so the resampler
  streams through exposures instead of loading them all at once. Important
  on smaller instances.

Skycell coverage varies: a skycell fully covered by all exposures gets a
~450 MB coadd; one clipped at the edge of the footprint is much smaller.
Edge coadds can be tiny (<10 MB) and that is fine.

### Stage 05 — extract source catalogs

```
pixi run bash scripts/05_run_catalog.sh configs/smoke.yaml
```

- **Reads:** each `*_coadd.asdf`.
- **Writes:** for each mosaic, a `<skycell>_cat.parquet` (the source
  catalog) and `<skycell>_segm.asdf` (the segmentation map).
- **What happens:** runs `romancal.source_catalog.SourceCatalogStep`, which
  applies a segmentation-based detection algorithm, deblends, measures
  aperture and Kron fluxes, PSF-fit fluxes, shape parameters, etc.

The parquet catalog is what you compare against your input
`metadata.parquet` to measure recovery — that comparison is the **actual
science goal** of this pipeline and is done by stage 06 below.

### Stage 06 — compare recovered vs input (recovery test)

```
pixi run python scripts/06_compare_catalog.py configs/smoke.yaml
# knobs: --max-skycells 10  --match-arcsec 0.3
```

- **Reads:** `catalogs/sources.parquet` (input), each skycell's coadd
  (for WCS + weight map = footprint), each skycell's recovered
  `_cat.parquet`.
- **Writes:** `output/<tag>/compare/plots/<skycell>.png` (five panels:
  completeness vs mag split by PSF/SER, flux residual vs mag,
  PSF-vs-aperture mag, false-positive rate vs recovered mag, astrometric
  residual vs mag, plus a text summary pane), and a one-row-per-skycell
  `summary.csv` with N_input / N_recovered / N_matched / N_FP,
  50%-completeness mag, median Δmag, median Δpos.
- **Scope (v1):** per-skycell only (no pooling across skycells), top-N
  deepest by cal-file count. Kron mag is the reference recovered
  magnitude for the flux-residual panel. Aperture column is `aper08_flux`;
  PSF is filtered to `psf_flags == 0`.
- **Source of truth:** an input source is "expected to be detectable"
  in a skycell only if its (ra, dec) projects onto a `weight > 0` pixel
  of that skycell's coadd. Non-detections at the footprint edge don't
  count against completeness.

## Running the whole thing

Smoke test:

```bash
CFG=configs/smoke.yaml
pixi run bash scripts/00_setup.sh
pixi run python scripts/00_prepare_catalog.py "$CFG"
pixi run bash scripts/00_hydrate_crds.sh   "$CFG"   # optional
for stage in 01_build_script 02_run_sims 03_build_asn 04_run_mosaic 05_run_catalog; do
    pixi run bash scripts/${stage}.sh "$CFG"
done
pixi run bash scripts/00_status.sh "$CFG"
```

Full run: same, replacing `configs/smoke.yaml` with `configs/full.yaml`.

## Inspecting outputs

```bash
# How many of each output type exists:
pixi run bash scripts/00_status.sh configs/smoke.yaml

# Peek at a cal file:
pixi run python -c "
import asdf
with asdf.open('output/cal/r0000101015521001006_0001_wfi01_f158_cal.asdf') as f:
    print(f.tree['roman']['meta'].keys())
    print('data shape:', f.tree['roman']['data'].shape)
"

# Peek at a mosaic:
pixi run python -c "
import asdf
with asdf.open('output/smoke/mosaic/smoke_p_full_010p00x50y50_f158_coadd.asdf') as f:
    print(f.tree['roman']['meta']['wcs'])
    print('data shape:', f.tree['roman']['data'].shape)
"

# Peek at a catalog:
pixi run python -c "
import pyarrow.parquet as pq
t = pq.read_table('output/smoke/catalog/smoke_p_full_010p00x50y50_f158_cat.parquet')
print(t.column_names[:15])
print(t.num_rows, 'sources')
"
```

## Where to go next

- **Compare recovered vs input catalogs** — the science step. Match sources
  between `data/metadata.parquet` and the union of `output/*/catalog/*.parquet`,
  measure flux bias + completeness as a function of magnitude. Not implemented
  yet in this repo.
- **Scale up** — see `CLAUDE.md` → Parallelism and Roadmap sections.
  Write a new `configs/<tag>.yaml` with a larger cone radius or box
  bounds, then re-run the stages against that config. Everything else
  (idempotency, cal sharing) still holds.
- **New filter** — add the filter's column (maggies) to the input catalog,
  copy `configs/smoke.yaml` to e.g. `configs/f184.yaml`, set
  `catalog.bandpass_col: F184` and `pointings.bandpass: F184`.

## Key design choices (one-liners)

- **Idempotent stages.** Every stage writes to a directory and skips outputs
  that already exist. Safe to re-run at any time.
- **Vendored `make_stack.py`.** Upstream crashes without `--apt`; see
  `CLAUDE.md`. Delete it once upstream is fixed.
- **Per-line deterministic seeds.** `_postprocess_sims.py` overrides
  `make_stack`'s constant seed so every exposure has its own noise.
- **Shared L2 dir, split L3 dirs.** `output/cal/` is shared so smoke cals
  feed the full run for free. `output/smoke/` and `output/full/` are
  separate so the sparser smoke mosaics never clobber the deeper full ones.
