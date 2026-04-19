# Data model reference

Empirical notes on the file formats this pipeline reads and writes.
Focused on fields / columns that matter for analysis code. Verified
against the smoke-run outputs as of 2026-04-19 with
`roman_datamodels` / `romancal` / `romanisim` current at that date.

## Recovered source catalog (`output/<tag>/catalog/<skycell>_cat.parquet`)

Produced by `romancal.source_catalog.SourceCatalogStep`. Open with
`astropy.table.Table.read(p)` or `pyarrow.parquet.read_table(p)`.

**Unit convention:** fluxes are **nanojanskys (nJy)**. The AB zeropoint
is 3631 Jy = 3.631×10¹⁴ nJy, so:

```python
mag_AB = -2.5 * np.log10(flux_nJy) + 2.5 * np.log10(3.631e14)   # ≈ 31.4
# verify: kron_flux and kron_abmag round-trip through this with ratio ≈ 1.0
```

**Columns worth knowing about:**

| Column | Notes |
|---|---|
| `ra`, `dec` | Primary sky position (weighted centroid). |
| `ra_centroid_win`, `dec_centroid_win` | Windowed centroid (slightly tighter for PSF sources). |
| `ra_psf`, `dec_psf` | Position from PSF-fit, available when PSF fit succeeded. |
| `x_centroid`, `y_centroid`, `x_psf`, `y_psf` | Pixel-space counterparts. |
| `kron_flux`, `kron_flux_err` | Kron aperture flux (nJy). Under-measures extended galaxies — residual vs input typically +0.1 to +0.2 mag. |
| `kron_abmag`, `kron_abmag_err` | AB mag directly (same information as `kron_flux`). |
| `aper01_flux` … `aper16_flux` | Fixed-radius apertures. Numbers are radii in units set by the romancal config — on our runs they're pixel radii (verify with metadata if extending). `aper08` is a sensible "standard" aperture for PSF-scale sources. |
| `psf_flux`, `psf_flux_err` | PSF-fit flux. **Only trust rows with `psf_flags == 0`.** Other values (we've seen 1, 24, 25) indicate PSF fit failures / quality flags. |
| `psf_gof` | Goodness-of-fit scalar; useful for filtering. |
| `segment_flux` | Flux integrated over the segmentation pixels. |
| `segment_area` | Segmentation area in pixels. |
| `is_extended` | Boolean-ish flag. **Caveat:** in our smoke catalogs this is overwhelmingly `True` (669/679 in one skycell). Don't trust this alone for star/galaxy split. |
| `fwhm`, `ellipticity`, `kron_radius`, `fluxfrac_radius_50` | Morphology. |
| `warning_flags`, `image_flags`, `psf_flags` | Quality bits. We've only ever seen 0 in `warning_flags` so far. |

**Schema gotcha — empty catalogs are different.** If a skycell has no
detections, the `.parquet` has `num_rows == 0` **and** a shorter column
list (e.g. only `aper00_flux` instead of the `aper01/02/04/08/16` split,
no `psf_flux`). Guard against this when iterating over skycells:

```python
t = Table.read(p)
if len(t) == 0:
    continue
```

## L3 coadd (`output/<tag>/mosaic/<skycell>_coadd.asdf`)

Produced by `romancal.pipeline.MosaicPipeline`. Open with
`roman_datamodels.datamodels.open(p)` (returns a `MosaicModel` or
similar). The ASDF tree's user-facing keys on the returned model:

| Attribute | Shape | Notes |
|---|---|---|
| `m.data` | `(ny, nx)` float | Coadded image. Default 5000×5000 for our runs. |
| `m.err` | `(ny, nx)` float | Per-pixel error. |
| `m.weight` | `(ny, nx)` float | Inverse-variance-weighted coverage. **`weight > 0` is the right footprint mask** — sources outside this shouldn't count. |
| `m.var_poisson`, `m.var_rnoise` | `(ny, nx)` float | Variance components. |
| `m.context` | `(n_layers, ny, nx)` uint32 | Bitmask: bit `j` of layer `k` is 1 iff the `(k * 32 + j)`-th input contributed to that pixel. Use `np.bitwise_count` for per-pixel contributor count. |

**Per-pixel depth** (how many cal files actually contributed):

```python
import numpy as np
import roman_datamodels.datamodels as rdm

m = rdm.open(coadd_path)
ctx = np.asarray(m.context)           # shape is usually (1, ny, nx) for our runs
if ctx.ndim == 3:
    popcount = np.zeros(ctx.shape[1:], dtype=np.int32)
    for layer in ctx:
        popcount += np.bitwise_count(layer).astype(np.int32)
else:
    popcount = np.bitwise_count(ctx).astype(np.int32)
# popcount now has contributor count per pixel
```

For our smoke (1 visit × 3 dithers), **max per-pixel depth is 3**. The
asn JSON's member count can be larger (e.g. 8) because it counts cal
files whose polygon *touches* the skycell, many of which only graze the
edge. The asn count is a rough proxy for depth; use `context` for the
real answer.

**WCS** is at `m.meta.wcs` (a `gwcs` object):

```python
x, y = m.meta.wcs.world_to_pixel_values(ra, dec)   # NaN if outside valid domain
ra0, dec0 = m.meta.wcs.pixel_to_world_values(x, y)
```

**Zeropoints live under `m.meta.photometry`**:
`conversion_megajanskys`, `conversion_microjanskys`, `pixel_area` (sr).
We don't typically need these because the recovered catalog fluxes are
already in nJy.

## L2 cal file (`output/cal/<visit-id>_<exposure>_wfi<sca>_<bandpass>_cal.asdf`)

Produced by `romanisim-make-image`. Open with
`roman_datamodels.datamodels.open(p)`.

**Identifiers** (from `m.meta`):

- `instrument.name = "WFI"`, `instrument.detector = "WFI01".."WFI18"`,
  `instrument.optical_element = "F158"` (or whatever bandpass).
- `exposure.ma_table_number` (1007 for HLWAS), `exposure.type = "WFI_IMAGE"`,
  `exposure.start_time`.
- `observation.program / pass / segment / visit / observation / exposure`
  — the same fields are encoded in the filename (see the "Output
  filename convention" section of `CLAUDE.md`).

**`ref_file` block** records which CRDS references were used:
`m.meta.ref_file.readnoise`, `.flat`, `.dark`, `.linearity`, etc. are
each `"crds://<filename>"` strings. Useful for provenance.

## Asn JSON (`output/<tag>/asn/<skycell>_asn.json`)

Produced by `skycell_asn`. Plain JSON, not ASDF. Structure (what we use):

```python
with open(asn_path) as f:
    d = json.load(f)
members = d["products"][0]["members"]      # list of {"expname": <path>, "exptype": ...}
n_touching = len(members)                   # = number of cal files whose polygon clips this skycell
```

See the depth note above — `len(members)` ≥ actual per-pixel contributor
count, strictly.

## Input source catalog (`catalogs/sources.parquet`)

Produced by `00_prepare_catalog.py` from `data/metadata.parquet`. Columns
we use + others that ride along:

| Column | Notes |
|---|---|
| `ra`, `dec` | Sky coordinates in degrees. |
| `type` | `"PSF"` or `"SER"` in our current catalog. Treat as authoritative input type. |
| `n`, `half_light_radius`, `pa`, `ba` | Sersic shape params (ignored for PSF). |
| `F158` | Flux in **maggies** (linear AB). `mag_AB = -2.5 * log10(F158)`. For a new bandpass add `F184` etc. |
| `z_obs`, `z_cosmo`, `sed_index`, `flux_scale`, `sim`, `src_index` | Upstream-provided; not currently consumed. |

The `F158` column is in the config's `catalog.bandpass_col`; the input's
unit (mag vs maggies) is at `catalog.input_units`.

## HLWAS pointings (`catalogs/HLWAS.sim.ecsv`)

See the "Pointings file" section of `CLAUDE.md` for what each column
means. Schema, as emitted by the upstream HLWAS simulation:

`RA, DEC, PA, BANDPASS, MA_TABLE_NUMBER, DURATION, PLAN, PASS, SEGMENT,
OBSERVATION, VISIT, EXPOSURE, EXPOSURE_TIME, TARGET_NAME`.

`filter_pointings.py` treats `(PLAN, PASS, SEGMENT, OBSERVATION, VISIT)`
as the composite visit key.

## CRDS dataset dict (for `crds.getreferences`)

The minimal Roman WFI header dict sufficient for `crds.getreferences` to
resolve any of the reftypes our pipeline uses. Verified via
`_hydrate_crds.py`:

```python
{
    "roman.meta.instrument.name": "WFI",
    "roman.meta.instrument.detector": "WFI01",       # ... through WFI18
    "roman.meta.instrument.optical_element": "F158",
    "roman.meta.exposure.type": "WFI_IMAGE",
    "roman.meta.exposure.ma_table_number": 1007,
    "roman.meta.exposure.start_time": "2026-01-01T00:00:00.000",
    "roman.meta.observation.start_time": "2026-01-01T00:00:00.000",
}
```

For ref-type coverage, see the `REFTYPES` list + the methodology comment
at the top of `scripts/_hydrate_crds.py`.

## Where these checks live in code

| File | What it reads |
|---|---|
| `scripts/06_compare_catalog.py` | Coadd `.data/.weight/.context/.meta.wcs`, catalog parquet, input parquet. |
| `scripts/_hydrate_crds.py` | CRDS (via `crds.getreferences`). |
| `scripts/_select_cal_files.py` | Pointings ECSV → cal file paths. |
| `scripts/03_build_asn.sh` | Writes asn JSONs via `skycell_asn`. |

Extend / cross-check against these when in doubt.
