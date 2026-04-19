#!/usr/bin/env python3
"""Photometry recovery test: per-skycell completeness + flux residual.

Given a run's config, walks the top-N deepest skycells (by cal-file count
in their asn JSON), and for each one:

1. Loads the coadd (`output/<tag>/mosaic/<skycell>_coadd.asdf`), reads its
   WCS + weight map, and identifies which input sources from
   `catalogs/sources.parquet` fall inside the coadd's footprint (weight>0).
2. Loads the recovered catalog (`output/<tag>/catalog/<skycell>_cat.parquet`).
3. Crossmatches in-footprint input sources against recovered sources by
   position (default 0.3").
4. Emits a 5-panel figure per skycell:
     - completeness vs input magnitude, split by input type (PSF / SER)
     - recovered mag − input mag vs input mag (Kron-based)
     - aperture (aper08) vs PSF magnitude, trusted PSF only (psf_flags==0)
     - false-positive rate vs recovered magnitude (Kron)
     - astrometric residual (|Δpos|) vs input magnitude
5. Appends one row per skycell to summary.csv with headline numbers.

Units: recovered catalog fluxes are nJy; AB mag = 31.4 − 2.5·log10(flux).
Input F158 is maggies; AB mag = -2.5·log10(maggies).

Usage:
    pixi run python scripts/06_compare_catalog.py configs/smoke.yaml
    # extra knobs:
    pixi run python scripts/06_compare_catalog.py configs/smoke.yaml \\
        --max-skycells 20 --match-arcsec 0.5
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
    n_asn_members: int          # #cal files whose footprint touches the skycell
    max_depth: int              # max per-pixel contributor count (from context)
    pct_depth_3: float          # fraction of valid pixels with depth == 3
    mean_depth: float           # mean per-pixel depth over weight>0 pixels
    n_input: int                # input sources inside weight>0 footprint
    n_recovered: int            # rows in recovered catalog
    n_matched: int
    n_fp: int                   # recovered with no input match
    completeness_overall: float
    mag_50pct: float
    median_dmag: float          # recovered Kron − input (matched only)
    median_dpos_arcsec: float


def _depth_from_context(ctx: np.ndarray) -> np.ndarray:
    """Per-pixel contributor count from a (possibly multi-layer) context array."""
    if ctx.ndim == 3:
        pc = np.zeros(ctx.shape[1:], dtype=np.int32)
        for layer in ctx:
            pc += np.bitwise_count(layer).astype(np.int32)
        return pc
    return np.bitwise_count(ctx).astype(np.int32)


def skycell_depth_stats(coadd_path: Path) -> dict:
    """Open a coadd and summarize per-pixel coverage (depth) stats."""
    m = rdm.open(coadd_path)
    ctx = np.asarray(m.context)
    w = np.asarray(m.weight)
    valid = w > 0
    depth = _depth_from_context(ctx)
    dv = depth[valid]
    if dv.size == 0:
        return {"max_depth": 0, "mean_depth": 0.0, "pct_depth_3": 0.0}
    return {
        "max_depth": int(dv.max()),
        "mean_depth": float(dv.mean()),
        "pct_depth_3": float((dv == 3).mean()),
    }


def rank_skycells_by_depth(asn_dir: Path, mosaic_dir: Path,
                           max_skycells: int, prefilter_factor: int = 3):
    """Rank skycells by the fraction of footprint pixels with full per-pixel
    depth (from the coadd's context bitmask). We first pre-filter by asn
    member count (which overestimates depth but is cheap to read), then open
    the coadds for those candidates and re-rank by true depth.

    Returns: [(base, n_asn_members, depth_stats), ...]
    """
    all_asn_counts = []
    for p in sorted(asn_dir.glob("*_asn.json")):
        with p.open() as f:
            d = json.load(f)
        n = len(d["products"][0]["members"])
        base = p.name[: -len("_asn.json")]
        all_asn_counts.append((base, n))
    all_asn_counts.sort(key=lambda x: (-x[1], x[0]))

    candidates = all_asn_counts[: max(max_skycells * prefilter_factor,
                                      max_skycells)]
    enriched = []
    for base, n_asn in candidates:
        coadd_path = mosaic_dir / f"{base}_coadd.asdf"
        if not coadd_path.is_file():
            continue
        stats = skycell_depth_stats(coadd_path)
        enriched.append((base, n_asn, stats))
    enriched.sort(key=lambda x: (-x[2]["pct_depth_3"], -x[2]["mean_depth"]))
    return enriched[:max_skycells]


def inside_footprint(wcs, weight: np.ndarray, ras: np.ndarray, decs: np.ndarray):
    """Boolean mask: which (ra, dec) land on a weight>0 pixel of the coadd?"""
    x, y = wcs.world_to_pixel_values(ras, decs)
    # world_to_pixel may return NaN for sources outside the valid WCS domain;
    # treat those as out-of-footprint before casting.
    finite = np.isfinite(x) & np.isfinite(y)
    ny, nx = weight.shape
    out = np.zeros(len(ras), dtype=bool)
    if not finite.any():
        return out
    ix = np.rint(x[finite]).astype(int)
    iy = np.rint(y[finite]).astype(int)
    in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    hit = np.zeros(finite.sum(), dtype=bool)
    hit[in_bounds] = weight[iy[in_bounds], ix[in_bounds]] > 0
    out[finite] = hit
    return out


def mag_50_completeness(mag_edges, completeness):
    """Linearly interpolate the magnitude at which completeness == 0.5.
    Returns NaN if it never dips below 0.5 or never reaches above 0.5."""
    centers = 0.5 * (mag_edges[:-1] + mag_edges[1:])
    # Find first bin where completeness drops below 0.5 while a brighter bin was above.
    above = np.array(completeness) >= 0.5
    if not above.any() or above.all():
        return float("nan")
    # Walk from bright to faint; find first 1→0 transition.
    for i in range(len(above) - 1):
        if above[i] and not above[i + 1]:
            c0, c1 = completeness[i], completeness[i + 1]
            m0, m1 = centers[i], centers[i + 1]
            # Linear interp on magnitude (monotonic) toward 0.5.
            if c0 == c1:
                return float(m0)
            frac = (c0 - 0.5) / (c0 - c1)
            return float(m0 + frac * (m1 - m0))
    return float("nan")


def analyze_skycell(base: str, n_asn_members: int, depth_stats: dict,
                    cfg, match_arcsec: float,
                    mag_bins: np.ndarray, mosaic_dir: Path, cat_dir: Path,
                    input_tbl: Table, plots_dir: Path) -> SkycellAnalysis:
    coadd_path = mosaic_dir / f"{base}_coadd.asdf"
    cat_path = cat_dir / f"{base}_cat.parquet"
    if not coadd_path.is_file() or not cat_path.is_file():
        raise FileNotFoundError(
            f"missing coadd or catalog for skycell {base}")

    m = rdm.open(coadd_path)
    weight = np.asarray(m.weight)
    wcs = m.meta.wcs

    ras_in = np.asarray(input_tbl["ra"], dtype=np.float64)
    decs_in = np.asarray(input_tbl["dec"], dtype=np.float64)
    types_in = np.asarray(input_tbl["type"])
    mag_in = maggies_to_abmag(input_tbl[cfg.catalog.bandpass_col])

    foot_mask = inside_footprint(wcs, weight, ras_in, decs_in)
    n_input = int(foot_mask.sum())

    in_ra = ras_in[foot_mask]
    in_dec = decs_in[foot_mask]
    in_type = types_in[foot_mask]
    in_mag = mag_in[foot_mask]

    rec = Table.read(cat_path).to_pandas() if cat_path.stat().st_size else None
    if rec is None or len(rec) == 0:
        rec_ra = np.empty(0)
        rec_dec = np.empty(0)
        rec_kron_mag = np.empty(0)
        rec_psf_flux = np.empty(0)
        rec_aper8 = np.empty(0)
        rec_psf_flags = np.empty(0)
    else:
        rec_ra = rec["ra"].to_numpy(dtype=np.float64)
        rec_dec = rec["dec"].to_numpy(dtype=np.float64)
        rec_kron_mag = rec["kron_abmag"].to_numpy(dtype=np.float64)
        rec_psf_flux = rec["psf_flux"].to_numpy(dtype=np.float64)
        rec_aper8 = rec["aper08_flux"].to_numpy(dtype=np.float64)
        rec_psf_flags = rec["psf_flags"].to_numpy()

    n_recovered = len(rec_ra)

    # Cross-match: each in-footprint input → nearest recovered.
    if n_input and n_recovered:
        c_in = SkyCoord(ra=in_ra * u.deg, dec=in_dec * u.deg)
        c_rec = SkyCoord(ra=rec_ra * u.deg, dec=rec_dec * u.deg)
        idx, sep, _ = c_in.match_to_catalog_sky(c_rec)
        sep_arcsec = sep.to(u.arcsec).value
        matched = sep_arcsec < match_arcsec
    else:
        idx = np.empty(n_input, dtype=int)
        sep_arcsec = np.full(n_input, np.inf)
        matched = np.zeros(n_input, dtype=bool)

    n_matched = int(matched.sum())

    # For false-positive rate we need: which recovered sources are NOT
    # matched to any input within match_arcsec. Do the reverse match.
    if n_input and n_recovered:
        c_rec = SkyCoord(ra=rec_ra * u.deg, dec=rec_dec * u.deg)
        c_in_all = SkyCoord(ra=in_ra * u.deg, dec=in_dec * u.deg)
        idx_r, sep_r, _ = c_rec.match_to_catalog_sky(c_in_all)
        rec_matched = sep_r.to(u.arcsec).value < match_arcsec
    else:
        rec_matched = np.zeros(n_recovered, dtype=bool)
    n_fp = int((~rec_matched).sum())

    # --- binning ---
    centers = 0.5 * (mag_bins[:-1] + mag_bins[1:])
    nbin = len(centers)

    def binned(mask_source, mag_source):
        """Return (count, matched, completeness) per bin for the given population."""
        m = np.asarray(mask_source)
        mag = np.asarray(mag_source)
        N = np.zeros(nbin, dtype=int)
        M = np.zeros(nbin, dtype=int)
        for i, (lo, hi) in enumerate(zip(mag_bins[:-1], mag_bins[1:])):
            sel = m & (mag >= lo) & (mag < hi)
            N[i] = sel.sum()
            M[i] = (sel & matched).sum()
        comp = np.where(N > 0, M / np.maximum(N, 1), np.nan)
        return N, M, comp

    N_psf, M_psf, C_psf = binned(in_type == "PSF", in_mag)
    N_ser, M_ser, C_ser = binned(in_type == "SER", in_mag)
    N_all, M_all, C_all = binned(np.ones_like(in_type, dtype=bool), in_mag)

    mag50 = mag_50_completeness(mag_bins, C_all)
    comp_overall = (n_matched / n_input) if n_input else float("nan")

    # --- matched-pair quantities ---
    if n_matched:
        matched_kron_mag = rec_kron_mag[idx[matched]]
        dmag = matched_kron_mag - in_mag[matched]
        median_dmag = float(np.nanmedian(dmag))
        median_dpos = float(np.median(sep_arcsec[matched]))
    else:
        dmag = np.array([])
        median_dmag = float("nan")
        median_dpos = float("nan")

    # Aperture-vs-PSF comparison (recovered-only).
    ap_mag = flux_njy_to_abmag(rec_aper8)
    psf_mag = flux_njy_to_abmag(rec_psf_flux)
    psf_trust = (rec_psf_flags == 0) & np.isfinite(ap_mag) & np.isfinite(psf_mag)

    # --- figure ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ((ax_c, ax_d, ax_ap), (ax_fp, ax_ast, ax_meta)) = axes

    # Panel 1: completeness
    ax_c.step(centers, C_all, where="mid", color="k", label=f"all (n={n_input})")
    ax_c.step(centers, C_psf, where="mid", color="C0",
              label=f"PSF (n={int(N_psf.sum())})")
    ax_c.step(centers, C_ser, where="mid", color="C1",
              label=f"SER (n={int(N_ser.sum())})")
    ax_c.axhline(0.5, ls=":", color="gray", lw=1)
    if np.isfinite(mag50):
        ax_c.axvline(mag50, ls=":", color="k", lw=1)
        ax_c.text(mag50, 0.05, f" 50% @ {mag50:.2f}", fontsize=9)
    ax_c.set_xlabel("input mag (AB)")
    ax_c.set_ylabel("completeness")
    ax_c.set_ylim(-0.05, 1.1)
    ax_c.set_title("Detection completeness")
    ax_c.legend(loc="lower left", fontsize=9)
    ax_c.grid(True, alpha=0.3)

    # Panel 2: flux residual (recovered Kron - input) vs input mag, matched only
    if n_matched:
        ax_d.scatter(in_mag[matched], dmag, s=5, alpha=0.4,
                     c=["C0" if t == "PSF" else "C1" for t in in_type[matched]])
    ax_d.axhline(0, color="k", lw=0.8)
    if np.isfinite(median_dmag):
        ax_d.axhline(median_dmag, color="r", ls="--", lw=1,
                     label=f"median={median_dmag:+.3f}")
        ax_d.legend(fontsize=9)
    ax_d.set_xlabel("input mag (AB)")
    ax_d.set_ylabel("recovered Kron − input (mag)")
    ax_d.set_title(f"Flux residual (n_matched={n_matched})")
    ax_d.set_ylim(-2, 2)
    ax_d.grid(True, alpha=0.3)

    # Panel 3: aperture vs PSF mag
    if psf_trust.any():
        ax_ap.scatter(ap_mag[psf_trust], psf_mag[psf_trust] - ap_mag[psf_trust],
                      s=5, alpha=0.4, color="C2")
    ax_ap.axhline(0, color="k", lw=0.8)
    ax_ap.set_xlabel("aper08 mag (AB)")
    ax_ap.set_ylabel("psf − aper08 (mag)")
    ax_ap.set_title(f"PSF vs aper08 (psf_flags=0, n={int(psf_trust.sum())})")
    ax_ap.set_ylim(-2, 2)
    ax_ap.grid(True, alpha=0.3)

    # Panel 4: false-positive fraction vs recovered Kron mag
    if n_recovered:
        rec_kron_finite = np.isfinite(rec_kron_mag)
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

    # Panel 5: astrometric residual vs input mag (matched only)
    if n_matched:
        ax_ast.scatter(in_mag[matched], sep_arcsec[matched], s=5, alpha=0.4,
                       c=["C0" if t == "PSF" else "C1" for t in in_type[matched]])
        ax_ast.axhline(median_dpos, color="r", ls="--", lw=1,
                       label=f"median={median_dpos:.3f}\"")
        ax_ast.legend(fontsize=9)
    ax_ast.set_xlabel("input mag (AB)")
    ax_ast.set_ylabel("|Δ| (arcsec)")
    ax_ast.set_ylim(0, match_arcsec)
    ax_ast.set_title("Astrometric residual")
    ax_ast.grid(True, alpha=0.3)

    # Panel 6: metadata / summary numbers
    ax_meta.axis("off")
    info = [
        f"skycell: {base}",
        f"n_asn_members: {n_asn_members}",
        f"coadd shape: {m.data.shape[0]}×{m.data.shape[1]}",
        f"footprint fraction: {(weight > 0).mean():.3f}",
        "",
        f"max per-pixel depth: {depth_stats['max_depth']}",
        f"mean per-pixel depth: {depth_stats['mean_depth']:.2f}",
        f"pct pixels at depth 3: {depth_stats['pct_depth_3']:.1%}",
        "",
        f"n_input in footprint: {n_input}",
        f"n_recovered: {n_recovered}",
        f"n_matched: {n_matched}",
        f"n_false_positive: {n_fp}",
        "",
        f"completeness: {comp_overall:.3f}",
        f"50% mag: {mag50:.2f}" if np.isfinite(mag50) else "50% mag: —",
        f"median Δmag (Kron): {median_dmag:+.3f}" if np.isfinite(median_dmag) else "median Δmag: —",
        f"median Δpos: {median_dpos:.3f}\"" if np.isfinite(median_dpos) else "median Δpos: —",
        f"match radius: {match_arcsec:.2f}\"",
    ]
    ax_meta.text(0.02, 0.98, "\n".join(info), va="top", ha="left",
                 family="monospace", fontsize=10)

    fig.suptitle(
        f"{base}  (asn_members={n_asn_members}, "
        f"pct_depth_3={depth_stats['pct_depth_3']:.1%})",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = plots_dir / f"{base}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

    return SkycellAnalysis(
        name=base,
        n_asn_members=n_asn_members,
        max_depth=depth_stats["max_depth"],
        pct_depth_3=depth_stats["pct_depth_3"],
        mean_depth=depth_stats["mean_depth"],
        n_input=n_input, n_recovered=n_recovered,
        n_matched=n_matched, n_fp=n_fp,
        completeness_overall=comp_overall,
        mag_50pct=mag50,
        median_dmag=median_dmag,
        median_dpos_arcsec=median_dpos,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to YAML config file")
    ap.add_argument("--max-skycells", type=int, default=10,
                    help="analyze the top-N deepest skycells (default 10)")
    ap.add_argument("--match-arcsec", type=float, default=0.3,
                    help="crossmatch radius in arcsec (default 0.3)")
    ap.add_argument("--mag-min", type=float, default=18.0)
    ap.add_argument("--mag-max", type=float, default=26.0)
    ap.add_argument("--mag-step", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config(args.config)

    asn_dir = Path(f"output/{cfg.tag}/asn")
    mosaic_dir = Path(f"output/{cfg.tag}/mosaic")
    cat_dir = Path(f"output/{cfg.tag}/catalog")
    out_dir = Path(f"output/{cfg.tag}/compare")
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"[config] {args.config}  tag={cfg.tag}")
    print(f"[io] asn={asn_dir}  mosaic={mosaic_dir}  cat={cat_dir}")

    ranked = rank_skycells_by_depth(asn_dir, mosaic_dir, args.max_skycells)
    print(f"[rank] processing top-{len(ranked)} skycells by pct_depth_3 "
          "(fraction of valid pixels with full 3-exposure coverage):")
    for base, n_asn, stats in ranked:
        print(f"         pct_d3={stats['pct_depth_3']:.1%}  "
              f"mean_d={stats['mean_depth']:.2f}  asn={n_asn:2d}  {base}")

    print("[input] loading catalogs/sources.parquet ...")
    input_tbl = Table.read("catalogs/sources.parquet")
    print(f"         {len(input_tbl):,} sources "
          f"(types: {sorted(set(input_tbl['type']))})")

    mag_bins = np.arange(args.mag_min, args.mag_max + args.mag_step / 2,
                         args.mag_step)

    results: list[SkycellAnalysis] = []
    for base, n_asn, stats in ranked:
        print(f"[skycell] {base}  asn={n_asn}  pct_d3={stats['pct_depth_3']:.1%} ...")
        try:
            r = analyze_skycell(
                base, n_asn, stats, cfg, args.match_arcsec, mag_bins,
                mosaic_dir, cat_dir, input_tbl, plots_dir,
            )
        except FileNotFoundError as e:
            print(f"           SKIP: {e}")
            continue
        results.append(r)
        print(f"           n_input={r.n_input}  n_rec={r.n_recovered}  "
              f"n_matched={r.n_matched}  n_fp={r.n_fp}  "
              f"comp={r.completeness_overall:.3f}  "
              f"mag50={r.mag_50pct:.2f}  "
              f"dmag={r.median_dmag:+.3f}  dpos={r.median_dpos_arcsec:.3f}\"")

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "skycell", "n_asn_members", "max_depth", "pct_depth_3", "mean_depth",
            "n_input", "n_recovered", "n_matched", "n_fp",
            "completeness_overall", "mag_50pct", "median_dmag_kron",
            "median_dpos_arcsec",
        ])
        for r in results:
            w.writerow([
                r.name, r.n_asn_members, r.max_depth,
                f"{r.pct_depth_3:.4f}", f"{r.mean_depth:.4f}",
                r.n_input, r.n_recovered, r.n_matched, r.n_fp,
                f"{r.completeness_overall:.4f}",
                f"{r.mag_50pct:.3f}" if np.isfinite(r.mag_50pct) else "nan",
                f"{r.median_dmag:+.4f}" if np.isfinite(r.median_dmag) else "nan",
                f"{r.median_dpos_arcsec:.4f}" if np.isfinite(r.median_dpos_arcsec) else "nan",
            ])
    print(f"[done] wrote {summary_path} and {len(results)} plots -> {plots_dir}")


if __name__ == "__main__":
    main()
