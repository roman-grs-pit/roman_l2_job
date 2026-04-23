#!/usr/bin/env python
"""Phase 5c: `npixels` sweep for one pair's mosaic.

Rerun `SourceCatalogStep` with `npixels ∈ {9, 16, 25}` (default 25 plus
two lower values) at the best-from-5b `kernel_fwhm = 2.0`. All other
params at defaults.
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
OUT_PLOTS = REPO / "output/detection/phase5c"
NPIXELS_VALUES = [9, 16, 25]
KERNEL_FWHM = 2.0
CROSSMATCH_ARCSEC = 0.3


def run_source_catalog(coadd: Path, npixels: int, out_dir: Path,
                       log_path: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"n{npixels:03d}"
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
        f"--kernel_fwhm={KERNEL_FWHM}",
        "--snr_threshold=3.0",
        f"--npixels={npixels}",
        "--deblend=False",
        "--save_results=true",
    ]
    with log_path.open("w") as log:
        log.write("# " + " ".join(cmd) + "\n")
        log.flush()
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    produced = list(out_dir.glob(f"*{tag}_cat*.parquet"))
    if not produced:
        raise SystemExit(f"No catalog produced for npixels={npixels} — see {log_path}")
    best = sorted(produced, key=lambda p: -len(p.stem))[0]
    if best != cat_path:
        best.rename(cat_path)
    return cat_path


def efficiency_by_mag(truth, recovered, max_sep):
    t = SkyCoord(ra=truth["ra"].values * u.deg, dec=truth["dec"].values * u.deg)
    r = SkyCoord(ra=recovered["ra"].values * u.deg,
                 dec=recovered["dec"].values * u.deg)
    _, sep, _ = match_coordinates_sky(t, r)
    hit = sep.arcsec < max_sep
    df = truth.assign(recovered=hit)
    g = df.groupby("mag")
    return pd.concat([g.size().rename("n_truth"),
                       g["recovered"].sum().rename("n_recovered"),
                       (g["recovered"].sum() / g.size()).rename("efficiency")],
                      axis=1).reset_index()


def mag_at(eff_df, target):
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


def fp(recovered, all_truth, max_sep):
    if len(recovered) == 0:
        return 0, 0
    all_t = SkyCoord(ra=all_truth["ra"].values * u.deg,
                      dec=all_truth["dec"].values * u.deg)
    r = SkyCoord(ra=recovered["ra"].values * u.deg,
                 dec=recovered["dec"].values * u.deg)
    _, sep, _ = match_coordinates_sky(r, all_t)
    return int((sep.arcsec > max_sep).sum()), len(recovered)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, required=True)
    args = ap.parse_args()
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)
    sweep_dir = REPO / f"output/detection/phase5c/catalogs_{args.pair}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    l3_dir = REPO / "output/detection/l3"
    coadd = sorted(l3_dir.glob(f"pair_{args.pair}_*_coadd.asdf"))[0]

    sk = Table.read(REPO / "catalogs/detection/selected_skycells.ecsv",
                    format="ascii.ecsv").to_pandas()
    skycell_name = str(sk[sk["PAIR_ID"] == args.pair]["skycell_name"].iloc[0])
    stars = Table.read(REPO / f"catalogs/detection/catalogs/skycell_{skycell_name}_stars.parquet").to_pandas()
    galaxies = Table.read(REPO / f"catalogs/detection/catalogs/skycell_{skycell_name}_galaxies.parquet").to_pandas()
    all_truth = pd.concat([stars[["ra", "dec"]], galaxies[["ra", "dec"]]])

    results = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    cmap = plt.get_cmap("viridis")

    for i, n in enumerate(NPIXELS_VALUES):
        print(f"\n=== npixels = {n} ===")
        cat = run_source_catalog(coadd, n, sweep_dir,
                                  sweep_dir / f"n{n:03d}.log")
        print(f"  {cat.name}")
        rec = Table.read(cat).to_pandas()
        eff_s = efficiency_by_mag(stars, rec, CROSSMATCH_ARCSEC)
        eff_g = efficiency_by_mag(galaxies, rec, CROSSMATCH_ARCSEC)
        n_fp, n_rec = fp(rec, all_truth, CROSSMATCH_ARCSEC)
        m50s = mag_at(eff_s, 0.5); m50g = mag_at(eff_g, 0.5)
        m90s = mag_at(eff_s, 0.9); m90g = mag_at(eff_g, 0.9)
        print(f"  n_recovered={n_rec}, n_fp={n_fp} ({n_fp/max(n_rec,1):.1%})")
        print(f"  50% completeness: stars {m50s:.2f}, galaxies {m50g:.2f}")
        print(f"  90% completeness: stars {m90s:.2f}, galaxies {m90g:.2f}")
        results.append({
            "npixels": n, "n_recovered": n_rec, "n_false_positive": n_fp,
            "fp_rate": n_fp / max(n_rec, 1),
            "mag_50pct_stars": m50s, "mag_50pct_galaxies": m50g,
            "mag_90pct_stars": m90s, "mag_90pct_galaxies": m90g,
        })
        c = cmap(i / max(len(NPIXELS_VALUES) - 1, 1))
        axes[0].plot(eff_s["mag"], eff_s["efficiency"], "o-",
                     color=c, label=f"npixels = {n}")
        axes[1].plot(eff_g["mag"], eff_g["efficiency"], "s-",
                     color=c, label=f"npixels = {n}")

    for ax, title in zip(axes, ["stars (PSF)", "galaxies (SER n=1, r=0.275″)"]):
        ax.axhline(0.5, color="k", ls=":", lw=0.6, alpha=0.5)
        ax.axhline(0.9, color="k", ls=":", lw=0.6, alpha=0.5)
        ax.set_xlabel("F158 magnitude (input)")
        ax.set_xlim(23.0, 26.0); ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3); ax.set_title(title); ax.legend(loc="lower left", fontsize=9)
    axes[0].set_ylabel("detection efficiency")
    fig.suptitle(
        f"Phase 5c — pair {args.pair} / {skycell_name}: npixels sweep "
        f"(kernel_fwhm={KERNEL_FWHM})",
        fontsize=12,
    )
    fig.tight_layout()
    out = OUT_PLOTS / f"sweep_pair_{args.pair}.png"
    fig.savefig(out, dpi=140); plt.close(fig)

    df = pd.DataFrame(results)
    df.to_csv(OUT_PLOTS / f"sweep_summary_pair_{args.pair}.csv", index=False)
    print(f"\nWrote {out}")
    print("\nSummary:\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
