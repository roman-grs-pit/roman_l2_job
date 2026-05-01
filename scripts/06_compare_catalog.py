#!/usr/bin/env python3
"""Photometry recovery test: per-skycell completeness + flux residual.

Given a run's config, walks **all** skycells in output/<tag>/asn/, and for
each one:

1. Loads the coadd, reads its WCS, weight map, and context bitmask. Computes
   per-pixel contributor depth via np.bitwise_count.
2. Filters the input `catalogs/sources.parquet` to sources whose (ra, dec)
   project onto a weight>0 pixel. The per-pixel depth at each source's
   position becomes a per-source `depth`.
3. Loads the recovered catalog, crossmatches by sky position (default 0.3").
4. Computes headline numbers (completeness, flux residual, FP rate, etc.)
   on the "all" population and on the "deep" subset (sources at pixels with
   depth strictly greater than this skycell's median per-pixel depth).
5. Emits a 5-panel figure per skycell; completeness and flux-residual
   panels overlay the deep subset on top of the all-sources curves.
6. Appends one row per skycell to summary.csv.
7. After the loop, writes `depth_distribution.png` — aggregated histogram
   of per-pixel depths across every skycell (weighted by pixel count) —
   so you can see the overall distribution of per-pixel coverage for the run.

Units: recovered catalog fluxes are nJy; AB mag = 31.4 − 2.5·log10(flux).
Input F158 is maggies; AB mag = -2.5·log10(maggies).

Usage:
    pixi run python scripts/06_compare_catalog.py configs/smoke.yaml
    pixi run python scripts/06_compare_catalog.py configs/full.yaml \\
        --match-arcsec 0.5 --max-skycells 20    # optional debugging knobs
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roman_datamodels.datamodels as rdm
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u

from _config import load_config


# AB zero point in nJy: 3631 Jy × 1e9 nJy/Jy
AB_NJY = 3631.0 * 1e9


def flux_njy_to_abmag(flux):
    """Vectorized nJy → AB magnitude. Non-positive flux → NaN."""
    f = np.asarray(flux, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -2.5 * np.log10(f / AB_NJY)
    out[f <= 0] = np.nan
    return out


def maggies_to_abmag(maggies):
    m = np.asarray(maggies, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -2.5 * np.log10(m)
    out[m <= 0] = np.nan
    return out


@dataclass
class SkycellAnalysis:
    name: str
    n_asn_members: int
    # WCS: center of coadd + bounding box in RA/Dec from the 4 pixel corners
    center_ra: float
    center_dec: float
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float
    min_depth: int
    median_depth: float
    max_depth: int
    mean_depth: float
    n_recovered: int
    n_fp: int
    # "all": sources in weight>0 footprint
    n_input_all: int
    n_matched_all: int
    completeness_all: float
    mag_50pct_all: float
    median_dmag_all: float
    median_dpos_all: float
    # "design": sources at pixels where depth >= config's pointings.design_depth
    n_input_design: int
    n_matched_design: int
    completeness_design: float
    mag_50pct_design: float
    median_dmag_design: float
    median_dpos_design: float


def _depth_from_context(ctx: np.ndarray) -> np.ndarray:
    """Per-pixel contributor count from a (possibly multi-layer) context array."""
    if ctx.ndim == 3:
        pc = np.zeros(ctx.shape[1:], dtype=np.int32)
        for layer in ctx:
            pc += np.bitwise_count(layer).astype(np.int32)
        return pc
    return np.bitwise_count(ctx).astype(np.int32)


def list_skycells(asn_dir: Path, max_skycells: int | None,
                  depth_csv: Path | None = None):
    """All skycells with their asn member count. Returns [(base, n_asn), ...].

    When `max_skycells` is set, sort to put the genuinely-deepest cells first
    (median per-pixel depth from `depth_csv` if available, else asn-member
    count -- which is a poor proxy: a skycell with 16 distinct pointings
    intersecting it can still have median pixel depth = 4 because the
    visit-overlap regions are slivers, not the bulk of the cell)."""
    results = []
    for p in sorted(asn_dir.glob("*_asn.json")):
        with p.open() as f:
            d = json.load(f)
        n = len(d["products"][0]["members"])
        base = p.name[: -len("_asn.json")]
        results.append((base, n))
    if max_skycells is not None:
        depth_by_base: dict[str, float] = {}
        if depth_csv is not None and depth_csv.is_file():
            with depth_csv.open() as f:
                for r in csv.DictReader(f):
                    depth_by_base[r["name"]] = float(r["dmedian"])
        if depth_by_base:
            results.sort(key=lambda x: (-depth_by_base.get(x[0], -1), -x[1], x[0]))
        else:
            results.sort(key=lambda x: (-x[1], x[0]))
        results = results[:max_skycells]
        results.sort(key=lambda x: x[0])
    return results


def inside_footprint(wcs, weight: np.ndarray, ras: np.ndarray, decs: np.ndarray):
    """(mask, ix, iy): mask is True where (ra,dec) lands on a weight>0 pixel.
    ix, iy are the rounded pixel indices (only valid where mask is True)."""
    x, y = wcs.world_to_pixel_values(ras, decs)
    finite = np.isfinite(x) & np.isfinite(y)
    ny, nx = weight.shape
    mask = np.zeros(len(ras), dtype=bool)
    ix_out = np.full(len(ras), -1, dtype=np.int64)
    iy_out = np.full(len(ras), -1, dtype=np.int64)
    if not finite.any():
        return mask, ix_out, iy_out
    ix = np.rint(x[finite]).astype(np.int64)
    iy = np.rint(y[finite]).astype(np.int64)
    in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    hit = np.zeros(finite.sum(), dtype=bool)
    hit[in_bounds] = weight[iy[in_bounds], ix[in_bounds]] > 0
    finite_idx = np.flatnonzero(finite)
    mask[finite_idx] = hit
    # Write ix/iy only where valid so downstream indexing never hits OOB.
    valid = finite_idx[hit]
    ix_out[valid] = ix[in_bounds][hit[in_bounds]]
    iy_out[valid] = iy[in_bounds][hit[in_bounds]]
    return mask, ix_out, iy_out


def mag_50_completeness(mag_edges, completeness):
    """Linearly interpolate the magnitude at which completeness == 0.5.
    Bins where completeness is NaN (e.g. too few sources) are skipped —
    the walk from bright to faint treats NaN bins as absent.
    """
    centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    comp = np.asarray(completeness, dtype=np.float64)
    valid = np.flatnonzero(np.isfinite(comp))
    if valid.size < 2:
        return float("nan")
    above = comp[valid] >= 0.5
    if not above.any() or above.all():
        return float("nan")
    for k in range(len(valid) - 1):
        i, j = valid[k], valid[k + 1]
        if above[k] and not above[k + 1]:
            c0, c1 = comp[i], comp[j]
            m0, m1 = centers[i], centers[j]
            if c0 == c1:
                return float(m0)
            frac = (c0 - 0.5) / (c0 - c1)
            return float(m0 + frac * (m1 - m0))
    return float("nan")


MIN_N_PER_BIN = 5  # completeness in a bin is NaN below this count


def _binned_completeness(mag_bins, mag, matched, mask=None, min_n=MIN_N_PER_BIN):
    """For a given input-source population (boolean mask), compute per-bin
    (N_input, N_matched, completeness). Bins with N < min_n get
    completeness = NaN so they don't perturb downstream interpolation."""
    centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    nbin = len(centers)
    if mask is None:
        mask = np.ones_like(mag, dtype=bool)
    N = np.zeros(nbin, dtype=int)
    M = np.zeros(nbin, dtype=int)
    for i, (lo, hi) in enumerate(zip(mag_bins[:-1], mag_bins[1:])):
        sel = mask & (mag >= lo) & (mag < hi)
        N[i] = sel.sum()
        M[i] = (sel & matched).sum()
    comp = np.where(N >= min_n, M / np.maximum(N, 1), np.nan)
    return N, M, comp


def analyze_skycell(base: str, n_asn_members: int, cfg, match_arcsec: float,
                    mag_bins: np.ndarray, mosaic_dir: Path, cat_dir: Path,
                    input_tbl: Table, plots_dir: Path) -> tuple[SkycellAnalysis, np.ndarray]:
    """Returns (analysis, depth_histogram) where depth_histogram is a 1-D
    np.int64 array indexed by per-pixel depth (over weight>0 pixels),
    suitable for aggregation across skycells."""
    coadd_path = mosaic_dir / f"{base}_coadd.asdf"
    cat_path = cat_dir / f"{base}_cat.parquet"
    if not coadd_path.is_file() or not cat_path.is_file():
        raise FileNotFoundError(f"missing coadd or catalog for {base}")

    m = rdm.open(coadd_path)
    weight = np.asarray(m.weight)
    wcs = m.meta.wcs
    depth = _depth_from_context(np.asarray(m.context))

    ny, nx = m.data.shape
    center_ra, center_dec = wcs.pixel_to_world_values(nx / 2.0, ny / 2.0)
    corner_ras, corner_decs = wcs.pixel_to_world_values(
        [0, nx - 1, nx - 1, 0], [0, 0, ny - 1, ny - 1])
    ra_min, ra_max = float(np.min(corner_ras)), float(np.max(corner_ras))
    dec_min, dec_max = float(np.min(corner_decs)), float(np.max(corner_decs))
    center_ra, center_dec = float(center_ra), float(center_dec)

    valid = weight > 0
    depth_valid = depth[valid]
    if depth_valid.size:
        depth_hist = np.bincount(depth_valid)
        d_min = int(depth_valid.min())
        d_max = int(depth_valid.max())
        d_mean = float(depth_valid.mean())
        d_median = float(np.median(depth_valid))
    else:
        depth_hist = np.zeros(1, dtype=np.int64)
        d_min = d_max = 0
        d_mean = d_median = 0.0

    ras_in = np.asarray(input_tbl["ra"], dtype=np.float64)
    decs_in = np.asarray(input_tbl["dec"], dtype=np.float64)
    types_in = np.asarray(input_tbl["type"])
    mag_in = maggies_to_abmag(input_tbl[cfg.catalog.bandpass_col])

    foot_mask, ix_in, iy_in = inside_footprint(wcs, weight, ras_in, decs_in)
    n_input_all = int(foot_mask.sum())

    in_ra = ras_in[foot_mask]
    in_dec = decs_in[foot_mask]
    in_type = types_in[foot_mask]
    in_mag = mag_in[foot_mask]
    # Per-source depth (at the source's pixel):
    in_depth = depth[iy_in[foot_mask], ix_in[foot_mask]]
    design_depth = cfg.pointings.design_depth
    design_mask = in_depth >= design_depth  # hit the config's design depth
    n_input_design = int(design_mask.sum())

    rec_bytes = cat_path.stat().st_size
    if rec_bytes:
        rec = Table.read(cat_path).to_pandas()
    else:
        rec = None
    if rec is None or len(rec) == 0:
        rec_ra = rec_dec = rec_kron_mag = np.empty(0)
        rec_psf_flux = rec_aper8 = np.empty(0)
        rec_psf_flags = np.empty(0)
    else:
        rec_ra = rec["ra"].to_numpy(dtype=np.float64)
        rec_dec = rec["dec"].to_numpy(dtype=np.float64)
        rec_kron_mag = rec["kron_abmag"].to_numpy(dtype=np.float64)
        rec_psf_flux = rec["psf_flux"].to_numpy(dtype=np.float64)
        rec_aper8 = rec["aper08_flux"].to_numpy(dtype=np.float64)
        rec_psf_flags = rec["psf_flags"].to_numpy()

    n_recovered = len(rec_ra)

    if n_input_all and n_recovered:
        c_in = SkyCoord(ra=in_ra * u.deg, dec=in_dec * u.deg)
        c_rec = SkyCoord(ra=rec_ra * u.deg, dec=rec_dec * u.deg)
        idx, sep, _ = c_in.match_to_catalog_sky(c_rec)
        sep_arcsec = sep.to(u.arcsec).value
        matched = sep_arcsec < match_arcsec
        # Reverse match for FP rate.
        idx_r, sep_r, _ = c_rec.match_to_catalog_sky(c_in)
        rec_matched = sep_r.to(u.arcsec).value < match_arcsec
    else:
        idx = np.empty(n_input_all, dtype=int)
        sep_arcsec = np.full(n_input_all, np.inf)
        matched = np.zeros(n_input_all, dtype=bool)
        rec_matched = np.zeros(n_recovered, dtype=bool)

    n_matched_all = int(matched.sum())
    n_matched_design = int((matched & design_mask).sum())
    n_fp = int((~rec_matched).sum())

    def pop_stats(sub_mask):
        """Headline stats for a sub-population of the in-footprint sources."""
        if sub_mask.sum() == 0:
            return {"comp": float("nan"), "mag50": float("nan"),
                    "dmag": float("nan"), "dpos": float("nan")}
        comp = (matched & sub_mask).sum() / sub_mask.sum()
        _, _, C = _binned_completeness(mag_bins, in_mag, matched, mask=sub_mask)
        mag50 = mag_50_completeness(mag_bins, C)
        sel = matched & sub_mask
        if sel.any():
            matched_kron = rec_kron_mag[idx[sel]]
            dmag = float(np.nanmedian(matched_kron - in_mag[sel]))
            dpos = float(np.median(sep_arcsec[sel]))
        else:
            dmag = float("nan")
            dpos = float("nan")
        return {"comp": comp, "mag50": mag50, "dmag": dmag, "dpos": dpos}

    all_mask = np.ones_like(in_mag, dtype=bool)
    stats_all = pop_stats(all_mask)
    stats_design = pop_stats(design_mask)

    # Pre-computed curves for plotting.
    centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    N_psf, _, C_psf = _binned_completeness(mag_bins, in_mag, matched,
                                           mask=(in_type == "PSF"))
    N_ser, _, C_ser = _binned_completeness(mag_bins, in_mag, matched,
                                           mask=(in_type == "SER"))
    _, _, C_all = _binned_completeness(mag_bins, in_mag, matched, mask=all_mask)
    _, _, C_design = _binned_completeness(mag_bins, in_mag, matched,
                                          mask=design_mask)

    # ----- figure -----
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ((ax_c, ax_d, ax_ap), (ax_fp, ax_ast, ax_meta)) = axes

    # Panel 1: completeness (all + design + type splits)
    ax_c.step(centers, C_all, where="mid", color="k", lw=1.5,
              label=f"all (n={n_input_all})")
    ax_c.step(centers, C_design, where="mid", color="C3", lw=1.5, ls="--",
              label=f"design d>={design_depth} (n={n_input_design})")
    ax_c.step(centers, C_psf, where="mid", color="C0", alpha=0.6,
              label=f"PSF (n={int(N_psf.sum())})")
    ax_c.step(centers, C_ser, where="mid", color="C1", alpha=0.6,
              label=f"SER (n={int(N_ser.sum())})")
    ax_c.axhline(0.5, ls=":", color="gray", lw=1)
    if np.isfinite(stats_all["mag50"]):
        ax_c.axvline(stats_all["mag50"], ls=":", color="k", lw=1)
    if np.isfinite(stats_design["mag50"]):
        ax_c.axvline(stats_design["mag50"], ls=":", color="C3", lw=1)
    ax_c.set_xlabel("input mag (AB)")
    ax_c.set_ylabel("completeness")
    ax_c.set_ylim(-0.05, 1.1)
    ax_c.set_title("Detection completeness")
    ax_c.legend(loc="lower left", fontsize=8)
    ax_c.grid(True, alpha=0.3)

    # Panel 2: flux residual scatter, matched only; highlight design in red.
    if n_matched_all:
        all_sel = matched
        design_sel = matched & design_mask
        ax_d.scatter(in_mag[all_sel & ~design_sel],
                     rec_kron_mag[idx[all_sel & ~design_sel]] - in_mag[all_sel & ~design_sel],
                     s=5, alpha=0.25, color="gray", label="below design depth")
        ax_d.scatter(in_mag[design_sel],
                     rec_kron_mag[idx[design_sel]] - in_mag[design_sel],
                     s=5, alpha=0.45, color="C3", label="at design depth")
        ax_d.legend(fontsize=8, loc="upper left")
    ax_d.axhline(0, color="k", lw=0.8)
    if np.isfinite(stats_design["dmag"]):
        ax_d.axhline(stats_design["dmag"], color="C3", ls="--", lw=1,
                     label=f"design median={stats_design['dmag']:+.3f}")
    ax_d.set_xlabel("input mag (AB)")
    ax_d.set_ylabel("recovered Kron − input (mag)")
    ax_d.set_title(f"Flux residual (n_matched={n_matched_all}, "
                   f"design={n_matched_design})")
    ax_d.set_ylim(-2, 2)
    ax_d.grid(True, alpha=0.3)

    # Panel 3: aperture vs PSF, split by input type. For input PSF sources,
    # PSF-fit and aperture mags should agree (Δ~0). For extended sources,
    # PSF-fit underestimates (Δ>0).
    ap_mag = flux_njy_to_abmag(rec_aper8)
    psf_mag = flux_njy_to_abmag(rec_psf_flux)
    psf_trust = (rec_psf_flags == 0) & np.isfinite(ap_mag) & np.isfinite(psf_mag)
    # Map trusted recovered rows back to their matched input type (if any)
    # so we can colour by PSF vs SER input. Length = n_recovered always.
    in_type_for_rec = np.full(n_recovered, "?", dtype=object)
    if n_input_all and n_recovered:
        in_type_for_rec[rec_matched] = in_type[idx_r[rec_matched]]
    is_psf_rec = (in_type_for_rec == "PSF") & psf_trust
    is_ser_rec = (in_type_for_rec == "SER") & psf_trust
    if is_psf_rec.any():
        ax_ap.scatter(ap_mag[is_psf_rec],
                      psf_mag[is_psf_rec] - ap_mag[is_psf_rec],
                      s=5, alpha=0.5, color="C0",
                      label=f"PSF input ({int(is_psf_rec.sum())})")
    if is_ser_rec.any():
        ax_ap.scatter(ap_mag[is_ser_rec],
                      psf_mag[is_ser_rec] - ap_mag[is_ser_rec],
                      s=5, alpha=0.35, color="C1",
                      label=f"SER input ({int(is_ser_rec.sum())})")
    ax_ap.axhline(0, color="k", lw=0.8)
    ax_ap.set_xlabel("aper08 mag (AB)")
    ax_ap.set_ylabel("psf − aper08 (mag)")
    ax_ap.set_title(f"PSF vs aper08 (psf_flags=0, n={int(psf_trust.sum())})")
    # Wide enough to show the input-PSF cloud near Δ ~ +0.6 and the input-SER
    # cloud out near Δ ~ +2.5.
    ax_ap.set_ylim(-1, 4)
    ax_ap.legend(fontsize=8, loc="upper left")
    ax_ap.grid(True, alpha=0.3)

    # Panel 4: FP fraction vs recovered Kron mag
    if n_recovered:
        rec_kron_finite = np.isfinite(rec_kron_mag)
        nbin = len(centers)
        N_r = np.zeros(nbin, dtype=int)
        N_fp = np.zeros(nbin, dtype=int)
        for i, (lo, hi) in enumerate(zip(mag_bins[:-1], mag_bins[1:])):
            sel = rec_kron_finite & (rec_kron_mag >= lo) & (rec_kron_mag < hi)
            N_r[i] = sel.sum()
            N_fp[i] = (sel & ~rec_matched).sum()
        fp_frac = np.where(N_r > 0, N_fp / np.maximum(N_r, 1), np.nan)
        ax_fp.step(centers, fp_frac, where="mid", color="C3")
        ax_fp.set_ylim(-0.05, 1.05)
    ax_fp.set_xlabel("recovered Kron mag (AB)")
    ax_fp.set_ylabel("FP fraction")
    ax_fp.set_title(f"False-positive rate (n_fp={n_fp} / {n_recovered})")
    ax_fp.grid(True, alpha=0.3)

    # Panel 5: astrometric residual
    if n_matched_all:
        ax_ast.scatter(in_mag[matched], sep_arcsec[matched], s=5, alpha=0.4,
                       c=["C0" if t == "PSF" else "C1" for t in in_type[matched]])
        if np.isfinite(stats_all["dpos"]):
            ax_ast.axhline(stats_all["dpos"], color="r", ls="--", lw=1,
                           label=f"median={stats_all['dpos']:.3f}\"")
            ax_ast.legend(fontsize=9)
    ax_ast.set_xlabel("input mag (AB)")
    ax_ast.set_ylabel("|Δ| (arcsec)")
    ax_ast.set_ylim(0, match_arcsec)
    ax_ast.set_title("Astrometric residual")
    ax_ast.grid(True, alpha=0.3)

    # Panel 6: metadata
    ax_meta.axis("off")
    info = [
        f"skycell: {base}",
        f"n_asn_members: {n_asn_members}",
        f"coadd shape: {m.data.shape[0]}×{m.data.shape[1]}",
        f"footprint fraction: {valid.mean():.3f}",
        "",
        f"depth: min={d_min} median={d_median:.1f} mean={d_mean:.2f} max={d_max}",
        f"design threshold: depth >= {design_depth}",
        "",
        "— all sources —",
        f"  n_input={n_input_all}  n_matched={n_matched_all}",
        f"  completeness={stats_all['comp']:.3f}",
        f"  50% mag={stats_all['mag50']:.2f}" if np.isfinite(stats_all['mag50']) else "  50% mag=—",
        f"  median Δmag={stats_all['dmag']:+.3f}" if np.isfinite(stats_all['dmag']) else "  median Δmag=—",
        f"  median Δpos={stats_all['dpos']:.3f}\"" if np.isfinite(stats_all['dpos']) else "  median Δpos=—",
        "",
        "— design subset —",
        f"  n_input={n_input_design}  n_matched={n_matched_design}",
        f"  completeness={stats_design['comp']:.3f}",
        f"  50% mag={stats_design['mag50']:.2f}" if np.isfinite(stats_design['mag50']) else "  50% mag=—",
        f"  median Δmag={stats_design['dmag']:+.3f}" if np.isfinite(stats_design['dmag']) else "  median Δmag=—",
        "",
        f"n_recovered={n_recovered}  n_fp={n_fp}",
        f"match radius: {match_arcsec:.2f}\"",
    ]
    ax_meta.text(0.02, 0.98, "\n".join(info), va="top", ha="left",
                 family="monospace", fontsize=9)

    fig.suptitle(
        f"{base}  (asn_members={n_asn_members}, "
        f"depth min/med/max={d_min}/{d_median:.1f}/{d_max})",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = plots_dir / f"{base}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    return (SkycellAnalysis(
        name=base, n_asn_members=n_asn_members,
        center_ra=center_ra, center_dec=center_dec,
        ra_min=ra_min, ra_max=ra_max,
        dec_min=dec_min, dec_max=dec_max,
        min_depth=d_min, median_depth=d_median,
        max_depth=d_max, mean_depth=d_mean,
        n_recovered=n_recovered, n_fp=n_fp,
        n_input_all=n_input_all, n_matched_all=n_matched_all,
        completeness_all=stats_all["comp"],
        mag_50pct_all=stats_all["mag50"],
        median_dmag_all=stats_all["dmag"],
        median_dpos_all=stats_all["dpos"],
        n_input_design=n_input_design, n_matched_design=n_matched_design,
        completeness_design=stats_design["comp"],
        mag_50pct_design=stats_design["mag50"],
        median_dmag_design=stats_design["dmag"],
        median_dpos_design=stats_design["dpos"],
    ), depth_hist)


def write_depth_distribution(hists: list[np.ndarray],
                             results: list[SkycellAnalysis],
                             out_path: Path):
    """Aggregate per-skycell per-pixel depth histograms into a single
    run-level depth distribution. The plot overlays two readouts:
      - bars: pixel count at each depth
      - line (right-axis): cumulative fraction of pixels with depth >= X
        — useful to read "what fraction of coverage meets a given
        design-depth threshold" directly off the figure."""
    if not hists:
        return
    max_len = max(len(h) for h in hists)
    agg = np.zeros(max_len, dtype=np.int64)
    for h in hists:
        agg[: len(h)] += h
    depths = np.arange(len(agg))
    total = agg.sum()
    if total == 0:
        return

    # CDF from the right: fraction of pixels with depth >= depth[i]
    frac_ge = np.cumsum(agg[::-1])[::-1] / total

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(depths, agg, color="C0", edgecolor="k", lw=0.5,
           label="# pixels")
    ax.set_xlabel("per-pixel depth (# cal files)")
    ax.set_ylabel("# pixels (aggregated over all skycells)")
    ax.set_title(f"Per-pixel depth distribution "
                 f"({total:,} pixels, {len(hists)} skycells)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(depths)

    ax_r = ax.twinx()
    ax_r.plot(depths, frac_ge, marker="o", color="C3",
              label="fraction ≥ depth")
    ax_r.set_ylim(0, 1.05)
    ax_r.set_ylabel("fraction of pixels with depth ≥ X", color="C3")
    ax_r.tick_params(axis="y", labelcolor="C3")

    # Single combined legend
    lines, labels = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines + lines_r, labels + labels_r, loc="upper right",
              fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to YAML config file")
    ap.add_argument("--max-skycells", type=int, default=None,
                    help="if set, analyze only this many skycells (debug)")
    ap.add_argument("--match-arcsec", type=float, default=0.3,
                    help="crossmatch radius in arcsec (default 0.3)")
    ap.add_argument("--mag-min", type=float, default=18.0)
    ap.add_argument("--mag-max", type=float, default=26.0)
    ap.add_argument("--mag-step", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)

    base = Path(cfg.output_base) if getattr(cfg, "output_base", None) else Path("output")
    asn_dir = base / cfg.tag / "asn"
    mosaic_dir = base / cfg.tag / "mosaic"
    cat_dir = base / cfg.tag / "catalog"
    out_dir = base / cfg.tag / "compare"
    depth_csv = base / cfg.tag / "qa" / "coadd_depth_summary.csv"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] {args.config}  tag={cfg.tag}")
    print(f"[io] asn={asn_dir}  mosaic={mosaic_dir}  cat={cat_dir}")

    skycells = list_skycells(asn_dir, args.max_skycells, depth_csv=depth_csv)
    print(f"[rank] processing {len(skycells)} skycells"
          + (f" (cut to {args.max_skycells})" if args.max_skycells else ""))

    print("[input] loading catalogs/sources.parquet ...")
    input_tbl = Table.read("catalogs/sources.parquet")
    print(f"         {len(input_tbl):,} sources "
          f"(types: {sorted(set(input_tbl['type']))})")

    mag_bins = np.arange(args.mag_min, args.mag_max + args.mag_step / 2,
                         args.mag_step)

    results: list[SkycellAnalysis] = []
    depth_hists: list[np.ndarray] = []
    for i, (base, n_asn) in enumerate(skycells, start=1):
        try:
            r, hist = analyze_skycell(
                base, n_asn, cfg, args.match_arcsec, mag_bins,
                mosaic_dir, cat_dir, input_tbl, plots_dir,
            )
        except FileNotFoundError as e:
            print(f"  [{i:3d}/{len(skycells)}] SKIP: {e}")
            continue
        results.append(r)
        depth_hists.append(hist)
        print(f"  [{i:3d}/{len(skycells)}] {base}  "
              f"asn={n_asn:3d}  depth min/med/max={r.min_depth}/{r.median_depth:.0f}/{r.max_depth}  "
              f"comp_all={r.completeness_all:.3f}  comp_design={r.completeness_design:.3f}  "
              f"mag50_all={r.mag_50pct_all:.2f}  mag50_design={r.mag_50pct_design:.2f}")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "skycell", "n_asn_members",
            "center_ra", "center_dec",
            "ra_min", "ra_max", "dec_min", "dec_max",
            "min_depth", "median_depth", "max_depth", "mean_depth",
            "n_recovered", "n_fp",
            "n_input_all", "n_matched_all",
            "completeness_all", "mag_50pct_all",
            "median_dmag_all", "median_dpos_all",
            "n_input_design", "n_matched_design",
            "completeness_design", "mag_50pct_design",
            "median_dmag_design", "median_dpos_design",
        ])
        for r in results:
            def fmt(x, nd=4):
                return f"{x:.{nd}f}" if np.isfinite(x) else "nan"
            def sfmt(x, nd=4):
                return f"{x:+.{nd}f}" if np.isfinite(x) else "nan"
            w.writerow([
                r.name, r.n_asn_members,
                f"{r.center_ra:.6f}", f"{r.center_dec:.6f}",
                f"{r.ra_min:.6f}", f"{r.ra_max:.6f}",
                f"{r.dec_min:.6f}", f"{r.dec_max:.6f}",
                r.min_depth, f"{r.median_depth:.2f}",
                r.max_depth, f"{r.mean_depth:.4f}",
                r.n_recovered, r.n_fp,
                r.n_input_all, r.n_matched_all,
                fmt(r.completeness_all), fmt(r.mag_50pct_all, 3),
                sfmt(r.median_dmag_all), fmt(r.median_dpos_all),
                r.n_input_design, r.n_matched_design,
                fmt(r.completeness_design), fmt(r.mag_50pct_design, 3),
                sfmt(r.median_dmag_design), fmt(r.median_dpos_design),
            ])

    depth_plot = out_dir / "depth_distribution.png"
    write_depth_distribution(depth_hists, results, depth_plot)

    print(f"[done] wrote {summary_path} "
          f"({len(results)} rows) and {len(results)} plots -> {plots_dir}")
    print(f"[done] wrote {depth_plot}")


if __name__ == "__main__":
    main()
