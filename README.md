# roman_l2_job

Self-contained bundle to run a Roman WFI L2 → L3 mosaic → source-catalog pipeline
on a Linux x86_64 machine.

## What this does

1. Simulates 18 F158 SCAs × N exposures of a sub-region of the HLWAS survey
   using `romanisim-make-image`.
2. Builds skycell associations from those L2 files with `skycell_asn`.
3. Coadds each skycell with `romancal.MosaicPipeline`.
4. Runs `SourceCatalogStep` on each mosaic.

Every run is driven by a YAML config in `configs/`. Two shipped:

- **`configs/smoke.yaml`** — 1 visit (3 exposures × 18 SCAs = 54 sims). ~2 hr at 4-way on m5a.2xlarge.
- **`configs/full.yaml`** — all 6 matching visits (324 sims). ~11 hr / ~$4 on m5a.2xlarge.

## One-time setup

```bash
# put the catalog in place (43 MB):
scp metadata.parquet user@host:roman_l2_job/data/

# install env + STPSF + CRDS scaffolding:
pixi install
pixi run bash scripts/00_setup.sh
pixi run python scripts/00_prepare_catalog.py configs/smoke.yaml

# (optional, recommended) pre-populate the CRDS reference cache serially,
# so parallel stage-02 workers never race on the same download:
pixi run bash scripts/00_hydrate_crds.sh configs/smoke.yaml
```

## Smoke test

```bash
pixi run bash scripts/01_build_script.sh configs/smoke.yaml   # pointings + sims.script
pixi run bash scripts/02_run_sims.sh     configs/smoke.yaml   # ~2 hr (4-way), populates output/cal/
pixi run bash scripts/03_build_asn.sh    configs/smoke.yaml   # output/smoke/asn/
pixi run bash scripts/04_run_mosaic.sh   configs/smoke.yaml   # output/smoke/mosaic/
pixi run bash scripts/05_run_catalog.sh  configs/smoke.yaml   # output/smoke/catalog/
pixi run bash scripts/00_status.sh       configs/smoke.yaml
```

## Full run

Same sequence, swap in `configs/full.yaml`. Cal files in `output/cal/` from the
smoke run are reused (filenames are unique per visit/exposure/SCA + idempotent
skip-if-exists), so no compute is wasted.

```bash
for stage in 01_build_script 02_run_sims 03_build_asn 04_run_mosaic 05_run_catalog; do
    pixi run bash scripts/${stage}.sh configs/full.yaml
done
pixi run bash scripts/00_status.sh configs/full.yaml
```

## Tuning parallelism

Edit `run.parallelism` in the config, or override ad-hoc:

```bash
PARALLELISM=8 pixi run bash scripts/02_run_sims.sh configs/full.yaml
```

The config's `run.parallelism` is the default; the env var overrides. BLAS
threads are pinned to 1 in the pixi activation env so workers don't oversubscribe.

## Making a new config

Copy `configs/smoke.yaml`, pick a new `tag`, adjust `pointings.region` (cone
or box), `pointings.bandpass`, and the optional `only_pass/segment/visit`
restrictions. Everything is required — no hidden defaults in code.

```yaml
pointings:
  region:
    type: box            # or "cone"
    ra_min: 9.5
    ra_max: 10.5
    dec_min: -0.25
    dec_max: 0.25
  bandpass: F158
  only_pass: null
  only_segment: null
  only_visit: null
```

The pointings ECSV (e.g. `pointings_<tag>.ecsv`) is regenerated from the
config by stage 01; it's gitignored.

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
