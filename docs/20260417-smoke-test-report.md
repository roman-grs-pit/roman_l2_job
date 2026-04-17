# Smoke test report — 2026-04-17

Host: AWS Yale Spinup, Linux m5a.2xlarge (8 vCPU, 32 GB).
Pipeline: `roman_l2_job` smoke tag (1 visit, 54 sims → 140 skycells).

## Result: SUCCESS

Final `00_status.sh smoke`:

```
== Status: tag=smoke ==
pointings:      3 exposures
L2 cal files:   54 / 54  (output/cal/)
asn (skycells): 140      (output/smoke/asn)
mosaics:        140      (output/smoke/mosaic)
catalogs:      140      (output/smoke/catalog)  [both _cat.parquet and _segm.asdf]
```

All four success criteria met:
- 54/54 L2 cal files in `output/cal/`
- ≥1 asn json in `output/smoke/asn/` (140)
- ≥1 mosaic `_coadd.asdf` in `output/smoke/mosaic/` (140)
- ≥1 `_cat.parquet` AND `_segm.asdf` in `output/smoke/catalog/` (140 of each)

## Timing

Stage 02 (sim generation) at PARALLELISM=4:
- Pass 1: ~116 min for 53 sims (one SCA crashed immediately on truncated CRDS file — see below).
- Pass 2 (SCA1 retry, CRDS re-download + sim): 500 s = ~8 min.
- Effective clean 4-worker wall time for all 54 sims: **~120 min**.

Prior 8-worker attempt OOM-killed at ~11 min into rendering.

Stages 03/04/05 ran serially after stage 02. Stage 04 mosaic produced 140 coadds from 140 asn files in roughly 3.7 hr (03:45 → 07:36). Stage 05 catalog finished in ~80 min.

## Memory

Measured via a 5 s polling `ps rss=` sampler over the whole stage 02 run:

| metric | value |
|---|---|
| Single-worker peak RSS | **~7.7 GB** (8,104,060 KB) |
| 4-worker combined peak RSS | **~22.7 GB** (23,846,040 KB) |
| 4-worker combined avg RSS | ~13.2 GB |

## Sizing implication for the full run

4-way fits comfortably in 32 GB. 8-way would spike to ~31 GB peak (right on the OOM edge) and actually crashed the prior run. Options for the 324-sim full run:

- **Stay on m5a.2xlarge, PARALLELISM=4:** ~12 h for stage 02 sims, no OOM risk. Cheapest.
- **Upgrade to m5a.4xlarge (16 vCPU, 64 GB), PARALLELISM=8:** ~6 h for stage 02 sims (EBS/memory bandwidth prevents full 2× scaling). Faster, ~1.7× the cost.

Recommendation: stay on 2xlarge + 4-way unless wall time matters more than $.

## Code change

One fix to `scripts/00_status.sh` — the existing `ls <glob> | wc -l` pattern bailed under `set -euo pipefail` when a glob matched nothing (stage 03/04/05 output dirs before those stages had run). Replaced with a `find`-based helper that returns 0 cleanly on no matches.

```diff
-NEXP=$(awk '!/^#/ && NF { if (++n>1) print }' "$POINTINGS" | wc -l | tr -d ' ')
-EXPECTED_CAL=$(( NEXP * 18 ))
-ACTUAL_CAL=$(ls output/cal/*_cal.asdf 2>/dev/null | wc -l | tr -d ' ')
-
-ASN_DIR="output/${TAG}/asn"
-MOSAIC_DIR="output/${TAG}/mosaic"
-CAT_DIR="output/${TAG}/catalog"
-N_ASN=$(ls "${ASN_DIR}"/*.json 2>/dev/null | wc -l | tr -d ' ')
-N_MOSAIC=$(ls "${MOSAIC_DIR}"/*_coadd.asdf 2>/dev/null | wc -l | tr -d ' ')
-N_CAT=$(ls "${CAT_DIR}"/*_cat.parquet 2>/dev/null | wc -l | tr -d ' ')
+NEXP=$(awk '!/^#/ && NF { if (++n>1) print }' "$POINTINGS" | wc -l | tr -d ' ')
+EXPECTED_CAL=$(( NEXP * 18 ))
+count() { find "$1" -maxdepth 1 -name "$2" 2>/dev/null | wc -l | tr -d ' '; }
+ACTUAL_CAL=$(count output/cal '*_cal.asdf')
+
+ASN_DIR="output/${TAG}/asn"
+MOSAIC_DIR="output/${TAG}/mosaic"
+CAT_DIR="output/${TAG}/catalog"
+N_ASN=$(count "${ASN_DIR}" '*.json')
+N_MOSAIC=$(count "${MOSAIC_DIR}" '*_coadd.asdf')
+N_CAT=$(count "${CAT_DIR}" '*_cat.parquet')
```

## Operational note (no code change)

The prior 8-worker crash left `crds_cache/references/roman/wfi/roman_wfi_readnoise_0024.asdf` truncated (40 MB vs 65 MB canonical) — a partial download. SCA1 in the retry hit that file and failed with `TypeError: buffer is too small for requested array` in `asdf/tags/core/ndarray.py`. Removed the truncated file and re-ran stage 02; CRDS refetched it cleanly and SCA1 simulated successfully.

Worth watching if future CRDS downloads get interrupted — check `ls -lh crds_cache/references/roman/wfi/roman_wfi_readnoise_*.asdf` for any outlier below 65 MB.
