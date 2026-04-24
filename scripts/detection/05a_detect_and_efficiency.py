#!/usr/bin/env python
"""Phase 5a: source detection + efficiency analysis for one skycell's mosaic.

Reads the MosaicPipeline side-effect catalog (`*_cat.parquet`),
crossmatches recovered sources against the Phase-3 truth (stars +
galaxies) by position, and emits

- per-mag-bin detection efficiency for stars and galaxies
- a 50%-completeness magnitude summary
- false-positive estimate (recovered sources with no truth match inside
  the cross-match radius)

Defaults: `SourceCatalogStep` settings are the MosaicPipeline defaults
(`kernel_fwhm=2.0`, `snr_threshold=3.0`, `npixels=25`, `deblend=False`).
The side-effect catalog captures these exactly. Phase 5b runs the step
standalone with other parameters.

Usage:  pixi run python scripts/detection/05a_detect_and_efficiency.py --cell 11119
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
import astropy.units as u

REPO = Path(__file__).resolve().parents[2]
OUT_PLOTS = REPO / "output/detection/phase5a"
CROSSMATCH_ARCSEC = 0.3  # conservative — ~ 5× the F158 PSF FWHM


def crossmatch(truth: pd.DataFrame, recovered: pd.DataFrame,
               max_sep_arcsec: float = CROSSMATCH_ARCSEC) -> pd.DataFrame:
    """Nearest-neighbour crossmatch. Returns truth + boolean `recovered`
    + separation in arcsec + recovered-catalog index (or -1)."""
    t = SkyCoord(ra=truth["ra"].values * u.deg, dec=truth["dec"].values * u.deg)
    r = SkyCoord(ra=recovered["ra"].values * u.deg,
                 dec=recovered["dec"].values * u.deg)
    idx, sep, _ = match_coordinates_sky(t, r)
    out = truth.copy()
    out["matched_idx"] = idx.astype(int)
    out["match_sep_arcsec"] = sep.arcsec
    out["recovered"] = sep.arcsec < max_sep_arcsec
    out.loc[~out["recovered"], "matched_idx"] = -1
    return out


def efficiency_by_mag(matched: pd.DataFrame, label: str) -> pd.DataFrame:
    """Detection efficiency per magnitude bin."""
    g = matched.groupby("mag")
    eff = (g["recovered"].sum() / g.size()).rename("efficiency")
    n = g.size().rename("n_truth")
    n_rec = g["recovered"].sum().rename("n_recovered")
    out = pd.concat([n, n_rec, eff], axis=1).reset_index()
    out["type"] = label
    return out


def interpolate_mag_at_efficiency(eff_df: pd.DataFrame, target: float) -> float:
    """Linearly interpolate the magnitude at which efficiency crosses `target`
    (from high to low completeness as mag increases)."""
    df = eff_df.sort_values("mag")
    above = df[df["efficiency"] >= target]
    below = df[df["efficiency"] < target]
    if above.empty:
        return float("nan")
    if below.empty:
        return float(df["mag"].iloc[-1])
    m0 = above["mag"].iloc[-1]
    e0 = above["efficiency"].iloc[-1]
    # First "below" with mag > m0
    after = below[below["mag"] > m0]
    if after.empty:
        return float("nan")
    m1 = after["mag"].iloc[0]
    e1 = after["efficiency"].iloc[0]
    if e0 == e1:
        return (m0 + m1) / 2
    return float(m0 + (target - e0) * (m1 - m0) / (e1 - e0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    ap.add_argument("--crossmatch-arcsec", type=float, default=CROSSMATCH_ARCSEC)
    args = ap.parse_args()

    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    l3_dir = REPO / "output/detection/l3"
    coadds = sorted(l3_dir.glob(f"sky_{args.cell}_*_coadd.asdf"))
    if not coadds:
        raise SystemExit(f"No coadd found for skycell {args.cell} in {l3_dir}")
    coadd = coadds[0]
    base = coadd.stem.removesuffix("_coadd")
    cat_path = l3_dir / f"{base}_cat.parquet"
    if not cat_path.exists():
        raise SystemExit(f"No side-effect catalog: {cat_path}")

    # --- recovered ---
    recovered = Table.read(cat_path).to_pandas()
    print(f"Recovered catalog: {len(recovered)} sources")

    # --- truth ---
    sk = Table.read(REPO / "catalogs/detection/selected_skycells.ecsv",
                    format="ascii.ecsv").to_pandas()
    skycell_name = str(sk[sk["SKYCELL_ID"] == args.cell]["skycell_name"].iloc[0])
    stars = Table.read(
        REPO / f"catalogs/detection/catalogs/skycell_{skycell_name}_stars.parquet"
    ).to_pandas()
    galaxies = Table.read(
        REPO / f"catalogs/detection/catalogs/skycell_{skycell_name}_galaxies.parquet"
    ).to_pandas()
    print(f"Truth: {len(stars)} stars + {len(galaxies)} galaxies "
          f"({skycell_name})")

    # --- crossmatch each type separately ---
    m_stars = crossmatch(stars, recovered, args.crossmatch_arcsec)
    m_gals = crossmatch(galaxies, recovered, args.crossmatch_arcsec)

    # Stars and galaxies share positions, so a recovered source near a
    # matched position could be claimed by both. Attribution ambiguity:
    # for this first validation we let each claim its nearest match — a
    # recovered source attributed to a star IS attributed to the galaxy
    # too at the same position. That's OK for efficiency per-type (each
    # is a separate denominator).
    eff_stars = efficiency_by_mag(m_stars, "stars")
    eff_gals = efficiency_by_mag(m_gals, "galaxies")

    # --- false positives ---
    all_truth = SkyCoord(
        ra=np.concatenate([stars["ra"], galaxies["ra"]]) * u.deg,
        dec=np.concatenate([stars["dec"], galaxies["dec"]]) * u.deg,
    )
    r = SkyCoord(ra=recovered["ra"].values * u.deg,
                 dec=recovered["dec"].values * u.deg)
    idx, sep, _ = match_coordinates_sky(r, all_truth)
    is_fp = sep.arcsec > args.crossmatch_arcsec
    n_fp = int(is_fp.sum())
    print(f"\nCrossmatch radius: {args.crossmatch_arcsec}\"")
    print(f"  recovered sources with no truth within radius (likely FPs): "
          f"{n_fp} / {len(recovered)} ({n_fp/len(recovered):.1%})")

    # --- summary ---
    mag_50_stars = interpolate_mag_at_efficiency(eff_stars, 0.5)
    mag_50_gals = interpolate_mag_at_efficiency(eff_gals, 0.5)
    mag_90_stars = interpolate_mag_at_efficiency(eff_stars, 0.9)
    mag_90_gals = interpolate_mag_at_efficiency(eff_gals, 0.9)
    print(f"\n50% completeness: stars {mag_50_stars:.2f}, "
          f"galaxies {mag_50_gals:.2f}")
    print(f"90% completeness: stars {mag_90_stars:.2f}, "
          f"galaxies {mag_90_gals:.2f}")

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(eff_stars["mag"], eff_stars["efficiency"],
            "o-", color="tab:blue", label="stars (PSF)")
    ax.plot(eff_gals["mag"], eff_gals["efficiency"],
            "s--", color="tab:red", label="galaxies (SER)")
    ax.axhline(0.5, color="k", ls=":", lw=0.6, alpha=0.5)
    ax.axhline(0.9, color="k", ls=":", lw=0.6, alpha=0.5)
    ax.set_xlabel("F158 magnitude (input)")
    ax.set_ylabel("detection efficiency")
    ax.set_title(f"Skycell {args.cell} — {skycell_name} (default SourceCatalogStep)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(23.0, 26.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    # recovered-position scatter (truth diamonds + recovered crosses)
    ax = axes[1]
    ax.scatter(stars["ra"], stars["dec"], marker=".", s=15,
               facecolor="none", edgecolor="tab:blue", alpha=0.5,
               label=f"{len(stars)} truth stars")
    ax.scatter(galaxies["ra"], galaxies["dec"], marker=".", s=15,
               facecolor="none", edgecolor="tab:red", alpha=0.5,
               label=f"{len(galaxies)} truth galaxies")
    ax.scatter(recovered["ra"], recovered["dec"], marker="x", s=10,
               color="k", alpha=0.6, linewidth=0.5,
               label=f"{len(recovered)} recovered")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.set_title("truth + recovered positions")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = OUT_PLOTS / f"efficiency_sky_{args.cell}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"\nWrote {out}")

    # --- save tables ---
    eff_df = pd.concat([eff_stars, eff_gals])
    eff_df.to_csv(OUT_PLOTS / f"efficiency_sky_{args.cell}.csv", index=False)
    summary = pd.DataFrame([{
        "SKYCELL_ID": args.cell,
        "skycell_name": skycell_name,
        "n_truth_stars": len(stars),
        "n_truth_galaxies": len(galaxies),
        "n_recovered": len(recovered),
        "n_false_positive": n_fp,
        "mag_50pct_stars": mag_50_stars,
        "mag_50pct_galaxies": mag_50_gals,
        "mag_90pct_stars": mag_90_stars,
        "mag_90pct_galaxies": mag_90_gals,
        "crossmatch_radius_arcsec": args.crossmatch_arcsec,
    }])
    summary.to_csv(OUT_PLOTS / f"summary_sky_{args.cell}.csv", index=False)


if __name__ == "__main__":
    main()
