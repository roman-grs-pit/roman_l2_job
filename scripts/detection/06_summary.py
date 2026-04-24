#!/usr/bin/env python
"""Phase 6: roll up per-skycell Phase 5 results.

Reads:
  output/detection/phase5a/summary_sky_*.csv  — baseline
  output/detection/phase5b/sweep_summary_sky_*.csv — kernel sweep
  output/detection/phase5c/sweep_summary_sky_*.csv — npixels sweep

Plus catalogs/detection/selected_skycells.ecsv for (zodi, coverage)
labels + the Phase-4 mosaic depth (read from each coadd's context
array).

Emits:
  output/detection/phase6/
    - baseline_all.png   : efficiency curves overlaid across mosaics
    - mag50_vs_depth.png : 50% mag vs per-pixel mosaic depth, coloured
                           by zodi bin, separate panels for stars/gals
    - param_sweep_summary.png : Δ(50% mag) for kernel_fwhm and npixels
                           sweeps, vs mosaic depth
    - all_skycells.csv      : one row per skycell consolidating baseline +
                           sweep results
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
import astropy.units as u
import roman_datamodels.datamodels as rdm

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output/detection/phase6"


def mosaic_depth_median(coadd: Path) -> float:
    m = rdm.open(coadd)
    ctx = np.asarray(m.context)
    if ctx.ndim == 3:
        depth = np.zeros(ctx.shape[-2:], dtype=int)
        for plane in ctx:
            depth += np.array([bin(int(x)).count("1") for x in plane.ravel()]
                               ).reshape(plane.shape)
    else:
        depth = np.array([bin(int(x)).count("1") for x in ctx.ravel()]).reshape(ctx.shape)
    interior = depth[200:-200, 200:-200]
    return float(np.median(interior))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sk = Table.read(REPO / "catalogs/detection/selected_skycells.ecsv",
                    format="ascii.ecsv").to_pandas()

    rows = []
    for _, s in sk.iterrows():
        pid = int(s["SKYCELL_ID"])
        skycell = s["skycell_name"]
        coadds = sorted((REPO / "output/detection/l3").glob(
            f"sky_{pid}_*_coadd.asdf"))
        if not coadds:
            continue
        depth = mosaic_depth_median(coadds[0])
        baseline = REPO / f"output/detection/phase5a/summary_sky_{pid}.csv"
        if baseline.exists():
            b = pd.read_csv(baseline).iloc[0].to_dict()
        else:
            continue
        row = {
            "SKYCELL_ID": pid,
            "skycell_name": skycell,
            "ECL_LAT_DEG": s["ECL_LAT_DEG"],
            "zodi_bin": s["zodi_bin"],
            "coverage_bin": s["coverage_bin"],
            "SKY_E_PER_PIX_PER_S": s.get("SKY_E_PER_PIX_PER_S"),
            "depth_median": depth,
            "mag_50_stars_default": b["mag_50pct_stars"],
            "mag_50_gals_default": b["mag_50pct_galaxies"],
            "mag_90_stars_default": b["mag_90pct_stars"],
            "mag_90_gals_default": b["mag_90pct_galaxies"],
        }
        # Attach kernel and npixels sweep deltas
        k = REPO / f"output/detection/phase5b/sweep_summary_sky_{pid}.csv"
        n = REPO / f"output/detection/phase5c/sweep_summary_sky_{pid}.csv"
        if k.exists():
            kdf = pd.read_csv(k)
            row["mag_50_stars_k3"] = kdf.loc[kdf["kernel_fwhm"]==3]["mag_50pct_stars"].iloc[0] if 3 in kdf["kernel_fwhm"].values else np.nan
            row["mag_50_gals_k3"] = kdf.loc[kdf["kernel_fwhm"]==3]["mag_50pct_galaxies"].iloc[0] if 3 in kdf["kernel_fwhm"].values else np.nan
            row["mag_50_stars_k7"] = kdf.loc[kdf["kernel_fwhm"]==7]["mag_50pct_stars"].iloc[0] if 7 in kdf["kernel_fwhm"].values else np.nan
        if n.exists():
            ndf = pd.read_csv(n)
            row["mag_50_stars_n16"] = ndf.loc[ndf["npixels"]==16]["mag_50pct_stars"].iloc[0] if 16 in ndf["npixels"].values else np.nan
            row["mag_50_gals_n16"] = ndf.loc[ndf["npixels"]==16]["mag_50pct_galaxies"].iloc[0] if 16 in ndf["npixels"].values else np.nan
            row["mag_50_stars_n9"] = ndf.loc[ndf["npixels"]==9]["mag_50pct_stars"].iloc[0] if 9 in ndf["npixels"].values else np.nan
            row["mag_50_gals_n9"] = ndf.loc[ndf["npixels"]==9]["mag_50pct_galaxies"].iloc[0] if 9 in ndf["npixels"].values else np.nan
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("SKYCELL_ID").reset_index(drop=True)
    df.to_csv(OUT / "all_skycells.csv", index=False)
    print(df.to_string(index=False))

    # Plot 1: mag_50% vs depth, coloured by zodi
    zodi_colours = {"low_zodi": "tab:blue", "mid_zodi": "tab:green",
                    "high_zodi": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for kind, ax, col in [
        ("stars", axes[0], "mag_50_stars_default"),
        ("galaxies (SER)", axes[1], "mag_50_gals_default"),
    ]:
        for zb, sub in df.groupby("zodi_bin"):
            ax.scatter(sub["depth_median"], sub[col], s=90,
                       color=zodi_colours.get(zb, "k"),
                       edgecolor="k", lw=0.5, label=zb)
            for _, r in sub.iterrows():
                ax.annotate(f" {int(r['SKYCELL_ID'])}",
                            (r["depth_median"], r[col]), fontsize=8)
        ax.set_xlabel("per-pixel mosaic depth (median interior)")
        ax.set_ylabel("F158 mag at 50% completeness (default params)")
        ax.set_title(f"{kind} — default SourceCatalogStep")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "mag50_vs_depth.png", dpi=140); plt.close(fig)
    print(f"Wrote {OUT / 'mag50_vs_depth.png'}")

    # Plot 2: parameter-sweep delta vs depth
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for kind, ax, base, k3, n16 in [
        ("stars", axes[0],
         "mag_50_stars_default", "mag_50_stars_k3", "mag_50_stars_n16"),
        ("galaxies", axes[1],
         "mag_50_gals_default", "mag_50_gals_k3", "mag_50_gals_n16"),
    ]:
        ax.scatter(df["depth_median"], df[k3] - df[base],
                   s=80, color="tab:blue", label="Δmag_50: kernel 3 − default")
        ax.scatter(df["depth_median"], df[n16] - df[base],
                   s=80, color="tab:red", marker="s",
                   label="Δmag_50: npixels 16 − default")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("per-pixel mosaic depth")
        ax.set_ylabel("Δ 50%-completeness magnitude (positive = deeper)")
        ax.set_title(kind); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "param_sweep_summary.png", dpi=140); plt.close(fig)
    print(f"Wrote {OUT / 'param_sweep_summary.png'}")

    # Plot 3: overlay all default efficiency curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("viridis")
    depths = df["depth_median"].values
    dmin, dmax = depths.min(), depths.max()
    for _, row in df.iterrows():
        pid = row["SKYCELL_ID"]
        eff = REPO / f"output/detection/phase5a/efficiency_sky_{pid}.csv"
        if not eff.exists():
            continue
        e = pd.read_csv(eff)
        c = cmap((row["depth_median"] - dmin) / max(dmax - dmin, 1e-6))
        es = e[e["type"] == "stars"]
        eg = e[e["type"] == "galaxies"]
        axes[0].plot(es["mag"], es["efficiency"], "-", color=c, alpha=0.85,
                     label=f"sky {int(pid)} (d≈{row['depth_median']:.0f})")
        axes[1].plot(eg["mag"], eg["efficiency"], "-", color=c, alpha=0.85,
                     label=f"sky {int(pid)}")
    for ax, title in zip(axes, ["stars (PSF)", "galaxies (SER)"]):
        ax.axhline(0.5, color="k", ls=":", lw=0.5)
        ax.axhline(0.9, color="k", ls=":", lw=0.5)
        ax.set_xlabel("F158 magnitude (input)")
        ax.set_ylabel("detection efficiency")
        ax.set_xlim(23.0, 26.0); ax.set_ylim(-0.05, 1.05)
        ax.set_title(title); ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=7)
    fig.suptitle("Default-parameter efficiency curves across all mosaics "
                 "(colour = mosaic depth)")
    fig.tight_layout()
    fig.savefig(OUT / "baseline_all.png", dpi=140); plt.close(fig)
    print(f"Wrote {OUT / 'baseline_all.png'}")

    # Plot 4: per-skycell panels — one row per skycell, stars + galaxies
    # side by side with all three param settings overlaid.
    n = len(df)
    ncols = 2  # stars, galaxies
    nrows = n
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.5 * nrows),
                              sharex=True, sharey=True)
    if nrows == 1:
        axes = np.array([axes])
    for row_i, (_, row) in enumerate(df.iterrows()):
        sid = int(row["SKYCELL_ID"])
        label = (f"sky {sid} — {row['skycell_name']} | "
                 f"ecl_lat={row['ECL_LAT_DEG']:+.1f}°, "
                 f"depth={row['depth_median']:.0f}")
        # Default (phase 5a)
        eff_a = REPO / f"output/detection/phase5a/efficiency_sky_{sid}.csv"
        # Kernel sweep (phase 5b) — pick kernel 2 (=default) and 3 / 5
        # npixels sweep (phase 5c) — pick n9, n16, n25
        for kind, ax_col in [("stars", 0), ("galaxies", 1)]:
            ax = axes[row_i, ax_col]
            if eff_a.exists():
                e = pd.read_csv(eff_a)
                sub = e[e["type"] == kind]
                ax.plot(sub["mag"], sub["efficiency"], "-", color="k",
                        lw=1.5, label="default (k=2, n=25)")
            # npixels variants (kernel=2 baseline from 5c)
            for npix, colour in [(9, "tab:red"), (16, "tab:orange"),
                                  (25, "tab:blue")]:
                cat = (REPO / "output/detection/phase5c" /
                       f"catalogs_{sid}" /
                       f"sky_{sid}_*_n{npix:03d}_cat.parquet")
                matches = list(cat.parent.glob(cat.name))
                if not matches:
                    continue
                rec = Table.read(matches[0]).to_pandas()
                truth_file = (REPO / "catalogs/detection/catalogs" /
                              f"skycell_{row['skycell_name']}_"
                              f"{'stars' if kind=='stars' else 'galaxies'}.parquet")
                if not truth_file.exists():
                    continue
                truth = Table.read(truth_file).to_pandas()
                tcoord = SkyCoord(ra=truth["ra"].values*u.deg,
                                   dec=truth["dec"].values*u.deg)
                rcoord = SkyCoord(ra=rec["ra"].values*u.deg,
                                   dec=rec["dec"].values*u.deg)
                _, sep, _ = match_coordinates_sky(tcoord, rcoord)
                truth["recovered"] = sep.arcsec < 0.3
                g = truth.groupby("mag")["recovered"]
                eff = (g.sum()/g.size())
                ax.plot(eff.index, eff.values, "--", color=colour,
                        lw=1.0, alpha=0.7, label=f"n={npix}")
            if kind == "stars" and ax_col == 0 and row_i == 0:
                ax.legend(loc="lower left", fontsize=7)
            ax.axhline(0.5, color="k", ls=":", lw=0.5, alpha=0.5)
            ax.grid(alpha=0.3)
            ax.set_xlim(23.0, 26.0); ax.set_ylim(-0.05, 1.05)
            if ax_col == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=5)
            if row_i == 0:
                ax.set_title(f"{kind}")
            if row_i == nrows - 1:
                ax.set_xlabel("F158 magnitude")
    fig.suptitle("Per-skycell efficiency curves — solid = default, "
                 "dashed = npixels variants at kernel_fwhm=2",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(OUT / "per_skycell_panels.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT / 'per_skycell_panels.png'}")


if __name__ == "__main__":
    main()
