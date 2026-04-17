# roman_l2_job

Self-contained bundle to run a Roman WFI L2 → L3 mosaic → source-catalog pipeline
on a Linux x86_64 machine.

## What this does

1. Simulates 18 F158 SCAs × N exposures of a Wide-Field2 sub-region (RA≈10, Dec≈0)
   using `romanisim-make-image`.
2. Builds skycell associations from those L2 files with `skycell_asn`.
3. Coadds each skycell with `romancal.MosaicPipeline`.
4. Runs `SourceCatalogStep` on each mosaic.

Two scopes:

- **smoke** — 1 visit (3 exposures × 18 SCAs = 54 sims). End-to-end test, ~15 min.
- **full**  — all 6 visits (18 × 18 = 324 sims). ~2 h on m5a.2xlarge, 8-way parallel.

## One-time setup

```bash
# put the catalog in place (43 MB):
scp metadata.parquet user@host:roman_l2_job/data/

# install env + STPSF + CRDS scaffolding:
pixi install
pixi run bash scripts/00_setup.sh
pixi run python scripts/00_prepare_catalog.py

# (optional, recommended) pre-populate the CRDS reference cache serially,
# so parallel stage-02 workers never race on the same download:
pixi run bash scripts/00_hydrate_crds.sh
```

## Smoke test

```bash
pixi run bash scripts/01_build_script.sh smoke            # writes output/smoke/sims.script
PARALLELISM=4 pixi run bash scripts/02_run_sims.sh smoke  # ~2 hr (4-way), populates output/cal/
pixi run bash scripts/03_build_asn.sh  smoke              # output/smoke/asn/
pixi run bash scripts/04_run_mosaic.sh smoke              # output/smoke/mosaic/
pixi run bash scripts/05_run_catalog.sh smoke             # output/smoke/catalog/
pixi run bash scripts/00_status.sh     smoke
```

## Full run

Same steps, replace `smoke` with `full`. The 54 cal files from the smoke test are
reused (same filenames, skip-if-exists), so no compute is wasted.

```bash
for stage in 01_build_script 02_run_sims 03_build_asn 04_run_mosaic 05_run_catalog; do
    pixi run bash scripts/${stage}.sh full
done
pixi run bash scripts/00_status.sh full
```

## Tuning parallelism

```bash
PARALLELISM=8 pixi run bash scripts/02_run_sims.sh full
```

Default = `nproc`. BLAS threads are pinned to 1 in the pixi activation env so
workers don't oversubscribe.

## Outputs

```
output/
├── cal/                   # L2 _cal.asdf  (shared, ~200 MB each)
├── logs/                  # one log per sim
├── crds_context.log
├── smoke/{asn,mosaic,catalog}/
└── full/{asn,mosaic,catalog}/
```

`MosaicPipeline` writes `<root>_coadd.asdf`; `SourceCatalogStep` adds
`<root>_segm.asdf` and `<root>_cat.parquet`.

## Re-running

Every stage is idempotent (skip-if-exists). Delete an output to force a re-run.
For partial sim failures, just re-run `02_run_sims.sh` — it picks up where it left off.
