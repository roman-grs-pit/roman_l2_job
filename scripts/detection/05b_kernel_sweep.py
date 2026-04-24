#!/usr/bin/env python
"""Phase 5b: `kernel_fwhm` sweep for one skycell's mosaic.

Runs `SourceCatalogStep` on the existing L3 coadd with
`kernel_fwhm ∈ {2, 3, 5, 7}` pix (kernel 2 matches the Phase 5a
baseline; 3/5/7 widen the matched filter for extended sources).
Other parameters stay at the MosaicPipeline defaults:
`snr_threshold=3`, `npixels=25`, `deblend=False`.

Each kernel run produces a `_cat.parquet`; we crossmatch against the
Phase-3 truth (stars + galaxies) and overlay the per-kernel efficiency
curves, separately for stars and galaxies.

Wall time: each `SourceCatalogStep` run is ~20-40 s on a 5000×5000
mosaic, so the full sweep is ~3 min.
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
import astropy.units as u

REPO = Path(__file__).resolve().parents[2]
OUT_PLOTS = REPO / "output/detection/phase5b"
KERNELS = [2.0, 3.0, 5.0, 7.0]
CROSSMATCH_ARCSEC = 0.3


def run_source_catalog(coadd: Path, kernel_fwhm: float,
                       out_dir: Path, log_path: Path) -> Path:
    """strun SourceCatalogStep with one kernel_fwhm value. Return the
    path of the produced *_cat.parquet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"k{int(kernel_fwhm*10):03d}"  # k020, k030, k050, k070
    out_base = coadd.stem.removesuffix("_coadd") + f"_{tag}"
    cat_path = out_dir / f"{out_base}_cat.parquet"
    if cat_path.exists():
        print(f"  [{tag}] cached — skipping rerun")
        return cat_path

    cmd = [
        "pixi", "run", "--manifest-path", str(REPO / "pixi.toml"),
        "strun", "romancal.source_catalog.SourceCatalogStep",
        str(coadd),
        "--output_dir", str(out_dir),
        f"--suffix={tag}_cat",
        f"--kernel_fwhm={kernel_fwhm}",
        "--snr_threshold=3.0",
        "--npixels=25",
        "--deblend=False",
        "--save_results=true",
    ]
    with log_path.open("w") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    produced = list(out_dir.glob(f"*{tag}_cat*.parquet"))
    if not produced:
        raise SystemExit(f"No catalog produced for kernel {kernel_fwhm} — see {log_path}")
    # Pick the one with the expected base if multiple
    best = sorted(produced, key=lambda p: -len(p.stem))[0]
    if best != cat_path:
        best.rename(cat_path)
    return cat_path


def efficiency_by_mag(truth: pd.DataFrame, recovered: pd.DataFrame,
                      max_sep_arcsec: float) -> pd.DataFrame:
    t = SkyCoord(ra=truth["ra"].values * u.deg, dec=truth["dec"].values * u.deg)
    r = SkyCoord(ra=recovered["ra"].values * u.deg,
                 dec=recovered["dec"].values * u.deg)
    _, sep, _ = match_coordinates_sky(t, r)
    hit = sep.arcsec < max_sep_arcsec
    df = truth.assign(recovered=hit)
    g = df.groupby("mag")
    eff = (g["recovered"].sum() / g.size()).rename("efficiency")
    n_truth = g.size().rename("n_truth")
    n_rec = g["recovered"].sum().rename("n_recovered")
    return pd.concat([n_truth, n_rec, eff], axis=1).reset_index()


def mag_at_efficiency(eff_df: pd.DataFrame, target: float) -> float:
    df = eff_df.sort_values("mag")
    above = df[df["efficiency"] >= target]
    below = df[df["efficiency"] < target]
    if above.empty or below.empty:
        return float("nan")
    m0 = above["mag"].iloc[-1]; e0 = above["efficiency"].iloc[-1]
    after = below[below["mag"] > m0]
    if after.empty:
        return float("nan")
    m1 = after["mag"].iloc[0]; e1 = after["efficiency"].iloc[0]
    if e0 == e1:
        return float((m0 + m1) / 2)
    return float(m0 + (target - e0) * (m1 - m0) / (e1 - e0))


def fp_rate(recovered: pd.DataFrame, all_truth: pd.DataFrame,
            max_sep_arcsec: float) -> tuple[int, int]:
    if len(recovered) == 0:
        return 0, 0
    all_t = SkyCoord(ra=all_truth["ra"].values * u.deg,
                      dec=all_truth["dec"].values * u.deg)
    r = SkyCoord(ra=recovered["ra"].values * u.deg,
                 dec=recovered["dec"].values * u.deg)
    _, sep, _ = match_coordinates_sky(r, all_t)
    is_fp = sep.arcsec > max_sep_arcsec
    return int(is_fp.sum()), len(recovered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    args = ap.parse_args()

    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    sweep_dir = REPO / f"output/detection/phase5b/catalogs_{args.cell}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    l3_dir = REPO / "output/detection/l3"
    coadds = sorted(l3_dir.glob(f"sky_{args.cell}_*_coadd.asdf"))
    if not coadds:
        raise SystemExit(f"No coadd for skycell {args.cell} in {l3_dir}")
    coadd = coadds[0]

    # Truth
    sk = Table.read(REPO / "catalogs/detection/selected_skycells.ecsv",
                    format="ascii.ecsv").to_pandas()
    skycell_name = str(sk[sk["SKYCELL_ID"] == args.cell]["skycell_name"].iloc[0])
    stars = Table.read(
        REPO / f"catalogs/detection/catalogs/skycell_{skycell_name}_stars.parquet"
    ).to_pandas()
    galaxies = Table.read(
        REPO / f"catalogs/detection/catalogs/skycell_{skycell_name}_galaxies.parquet"
    ).to_pandas()
    all_truth = pd.concat([stars[["ra", "dec"]], galaxies[["ra", "dec"]]])

    results = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cmap = plt.get_cmap("viridis")

    for i, k in enumerate(KERNELS):
        print(f"\n=== kernel_fwhm = {k} ===")
        log_path = sweep_dir / f"k{int(k*10):03d}.log"
        t0 = time.time()
        cat_path = run_source_catalog(coadd, k, sweep_dir, log_path)
        print(f"  {cat_path.name} ({time.time()-t0:.1f}s)")
        rec = Table.read(cat_path).to_pandas()
        eff_s = efficiency_by_mag(stars, rec, CROSSMATCH_ARCSEC)
        eff_g = efficiency_by_mag(galaxies, rec, CROSSMATCH_ARCSEC)
        n_fp, n_rec = fp_rate(rec, all_truth, CROSSMATCH_ARCSEC)
        m50_s = mag_at_efficiency(eff_s, 0.5)
        m50_g = mag_at_efficiency(eff_g, 0.5)
        m90_s = mag_at_efficiency(eff_s, 0.9)
        m90_g = mag_at_efficiency(eff_g, 0.9)
        print(f"  n_recovered = {n_rec}, n_fp = {n_fp} ({n_fp/max(n_rec,1):.1%})")
        print(f"  50% completeness: stars {m50_s:.2f}, galaxies {m50_g:.2f}")
        print(f"  90% completeness: stars {m90_s:.2f}, galaxies {m90_g:.2f}")
        results.append({
            "kernel_fwhm": k,
            "n_recovered": n_rec,
            "n_false_positive": n_fp,
            "fp_rate": n_fp / max(n_rec, 1),
            "mag_50pct_stars": m50_s,
            "mag_50pct_galaxies": m50_g,
            "mag_90pct_stars": m90_s,
            "mag_90pct_galaxies": m90_g,
        })
        c = cmap(i / (len(KERNELS) - 1))
        axes[0].plot(eff_s["mag"], eff_s["efficiency"], "o-",
                     color=c, label=f"kernel_fwhm = {k}")
        axes[1].plot(eff_g["mag"], eff_g["efficiency"], "s-",
                     color=c, label=f"kernel_fwhm = {k}")

    for ax, title in zip(axes, ["stars (PSF)", "galaxies (SER n=1, r=0.275″)"]):
        ax.axhline(0.5, color="k", ls=":", lw=0.6, alpha=0.5)
        ax.axhline(0.9, color="k", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel("F158 magnitude (input)")
        ax.set_xlim(23.0, 26.0)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.set_title(title)
        ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("detection efficiency")
    fig.suptitle(
        f"Phase 5b — skycell {args.cell} / {skycell_name}: kernel_fwhm sweep",
        fontsize=12
    )
    fig.tight_layout()
    out = OUT_PLOTS / f"sweep_sky_{args.cell}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)

    df = pd.DataFrame(results)
    df.to_csv(OUT_PLOTS / f"sweep_summary_sky_{args.cell}.csv", index=False)
    print(f"\nWrote {out}")
    print(f"Wrote {OUT_PLOTS / f'sweep_summary_sky_{args.cell}.csv'}")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
