#!/usr/bin/env python
"""Phase 3: truth-catalog generation for each selected skycell.

For every skycell in `catalogs/detection/selected_skycells.ecsv`, we
generate two matched source catalogs:

- `skycell_<name>_stars.parquet` — 210 point sources (`type="PSF"`)
- `skycell_<name>_galaxies.parquet` — 210 circular-exponential galaxies
  at **identical RA/Dec and magnitudes** (`type="SER"`, Sersic n=1,
  half_light_radius=0.275", axis ratio 1, PA 0).

The matched design means detection efficiency can be compared
point-for-point between stars and galaxies at the same sky position and
brightness — the core question the study answers.

Layout
------
Sources are 21 magnitudes × 10 instances = 210 points per skycell per
source type. Magnitudes run 23.0–25.0 in 0.1-mag steps; fluxes are
stored as maggies in the `F158` column (= 10**(-mag/2.5)).

Pixel positions are drawn from a **2-D Sobol low-discrepancy sequence**
inside `[MARGIN, NX - MARGIN] × [MARGIN, NY - MARGIN]` of each skycell.
Sobol gives quasi-random uniform coverage — every mag-bin subset of 10
points still spans the skycell, so we avoid the "all 23.0-mag stars on
one row" artefact of a regular grid.

Magnitudes are then assigned to the Sobol positions by a seeded shuffle
so that each mag bin samples the full skycell area — there is no
systematic correlation between magnitude and position.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.table import Table
from romancal.skycell.skymap import SkyCells
from scipy.stats import qmc

# 210 is not a power of 2 — scipy warns that the Sobol balance
# properties degrade slightly. We deliberately use 210 = 21×10 (plan
# spec); the warning is cosmetic.
warnings.filterwarnings(
    "ignore",
    message="The balance properties of Sobol' points require n to be a power of 2.",
)

REPO = Path(__file__).resolve().parents[2]
SKYCELLS_IN = REPO / "catalogs/detection/selected_skycells.ecsv"
OUT_CAT = REPO / "catalogs/detection/catalogs"
OUT_PLOTS = REPO / "output/detection/phase3"

# 5000 × 5000 pixel skycells overlap adjacent skycells by
# `skycell_border_pixels = 100` on each side (so the unique core is
# [100, 4900]²). A 200-pixel margin puts us 100 pixels *inside* the
# unique core, with room to spare for PSF wings and drizzle kernels.
MARGIN_PIX = 200

# Source grid: 21 mag bins × 10 sources per type = 210 stars + 210 galaxies.
# Positions are **disjoint** between stars and galaxies (420 Sobol points
# total, split in half), so detection sees them as separate sources.
MAG_MIN, MAG_MAX = 23.0, 26.0
N_MAG_BINS = 21
SOURCES_PER_MAG = 10
BANDPASS_COL = "F158"

# Galaxy shape (per the plan): circular exponential, n=1, r_h=0.275"
GAL_N = 1.0
GAL_REFF_ARCSEC = 0.275
GAL_PA = 0.0
GAL_BA = 1.0

RNG_SEED = 20260422


def build_mag_array() -> np.ndarray:
    mags = np.linspace(MAG_MIN, MAG_MAX, N_MAG_BINS)
    return np.repeat(mags, SOURCES_PER_MAG)


def sobol_pixel_positions(nx: int, ny: int, n: int, seed: int,
                          margin: int = MARGIN_PIX) -> np.ndarray:
    """n×2 array of Sobol quasi-random pixel positions in the skycell
    core `[margin, nx-margin] × [margin, ny-margin]`."""
    eng = qmc.Sobol(d=2, scramble=True, seed=seed)
    points = eng.random(n=n)  # shape (n, 2) in [0, 1)
    out = np.empty_like(points)
    out[:, 0] = margin + points[:, 0] * (nx - 2 * margin)
    out[:, 1] = margin + points[:, 1] * (ny - 2 * margin)
    return out


def catalog_for_skycell(skycell_idx: int, skycell_name: str,
                        rng_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build (stars_df, galaxies_df) for one skycell with disjoint
    positions. Generates 2 × 210 = 420 Sobol positions; the first half
    are stars, the second half are galaxies. Each type's 210 positions
    get its own seeded magnitude shuffle."""
    cells = SkyCells([skycell_idx])
    wcs = cells.wcs[0]
    nx, ny = cells.pixel_shape  # (5000, 5000)

    n_per_type = N_MAG_BINS * SOURCES_PER_MAG  # 210
    n_total = 2 * n_per_type  # 420 disjoint positions
    pix = sobol_pixel_positions(nx, ny, n_total, seed=rng_seed)
    pix_stars = pix[:n_per_type]
    pix_gals = pix[n_per_type:]

    ra_s, dec_s = wcs(pix_stars[:, 0], pix_stars[:, 1])
    ra_g, dec_g = wcs(pix_gals[:, 0], pix_gals[:, 1])
    ra_s = np.asarray(ra_s); dec_s = np.asarray(dec_s)
    ra_g = np.asarray(ra_g); dec_g = np.asarray(dec_g)

    rng_s = np.random.default_rng(rng_seed + 1)
    rng_g = np.random.default_rng(rng_seed + 2)
    mags_base = build_mag_array()
    mags_s = mags_base[rng_s.permutation(n_per_type)]
    mags_g = mags_base[rng_g.permutation(n_per_type)]
    flux_s = np.power(10.0, -mags_s / 2.5)
    flux_g = np.power(10.0, -mags_g / 2.5)

    stars = pd.DataFrame({
        "ra": ra_s,
        "dec": dec_s,
        "type": ["PSF"] * n_per_type,
        "n": np.zeros(n_per_type, dtype=np.float32),
        "half_light_radius": np.zeros(n_per_type, dtype=np.float32),
        "pa": np.zeros(n_per_type, dtype=np.float32),
        "ba": np.ones(n_per_type, dtype=np.float32),
        BANDPASS_COL: flux_s,
        "mag": mags_s,
        "src_index": np.arange(n_per_type, dtype=np.int32),
        "pix_x": pix_stars[:, 0].astype(np.float32),
        "pix_y": pix_stars[:, 1].astype(np.float32),
    })

    galaxies = pd.DataFrame({
        "ra": ra_g,
        "dec": dec_g,
        "type": ["SER"] * n_per_type,
        "n": np.full(n_per_type, GAL_N, dtype=np.float32),
        "half_light_radius": np.full(n_per_type, GAL_REFF_ARCSEC,
                                       dtype=np.float32),
        "pa": np.full(n_per_type, GAL_PA, dtype=np.float32),
        "ba": np.full(n_per_type, GAL_BA, dtype=np.float32),
        BANDPASS_COL: flux_g,
        "mag": mags_g,
        "src_index": np.arange(n_per_type, dtype=np.int32) + n_per_type,
        "pix_x": pix_gals[:, 0].astype(np.float32),
        "pix_y": pix_gals[:, 1].astype(np.float32),
    })

    return stars, galaxies


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet with the exact column dtypes romanisim expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = Table.from_pandas(df)
    t.write(path, overwrite=True)


def qa_plot(skycell_idx: int, skycell_name: str,
            stars_df: pd.DataFrame, galaxies_df: pd.DataFrame,
            out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ss = ax.scatter(stars_df["pix_x"], stars_df["pix_y"],
                    c=stars_df["mag"], cmap="viridis", vmin=MAG_MIN, vmax=MAG_MAX,
                    s=30, edgecolor="k", lw=0.2, marker="o",
                    label=f"{len(stars_df)} stars (PSF)")
    ax.scatter(galaxies_df["pix_x"], galaxies_df["pix_y"],
                c=galaxies_df["mag"], cmap="viridis", vmin=MAG_MIN, vmax=MAG_MAX,
                s=30, edgecolor="k", lw=0.2, marker="s",
                label=f"{len(galaxies_df)} galaxies (SER)")
    ax.plot([100, 4900, 4900, 100, 100],
            [100, 100, 4900, 4900, 100],
            ls=":", color="k", lw=0.8, alpha=0.4,
            label="unique core edge (100 px border)")
    ax.plot([MARGIN_PIX, 5000 - MARGIN_PIX, 5000 - MARGIN_PIX,
             MARGIN_PIX, MARGIN_PIX],
            [MARGIN_PIX, MARGIN_PIX, 5000 - MARGIN_PIX,
             5000 - MARGIN_PIX, MARGIN_PIX],
            ls="--", color="k", lw=0.8, alpha=0.6,
            label=f"injection boundary ({MARGIN_PIX} px margin)")
    ax.set_xlabel("pixel x"); ax.set_ylabel("pixel y")
    ax.set_aspect("equal"); ax.set_xlim(0, 5000); ax.set_ylim(0, 5000)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=7)
    ax.set_title(f"{skycell_name} (idx {skycell_idx}) — disjoint Sobol layout")
    fig.colorbar(ss, ax=ax, label="F158 magnitude")

    ax = axes[1]
    combined = pd.concat([stars_df, galaxies_df])
    # Histogram of nearest-neighbour separations
    from scipy.spatial import cKDTree
    xy = combined[["pix_x", "pix_y"]].to_numpy()
    tree = cKDTree(xy)
    dists, _ = tree.query(xy, k=2)
    nn = dists[:, 1]
    ax.hist(nn * 0.055, bins=40, color="tab:blue", alpha=0.8)
    ax.set_xlabel("nearest-neighbour separation (arcsec)")
    ax.set_ylabel("# sources")
    ax.axvline(5, color="tab:red", ls="--", lw=1, label="~5″ blending floor")
    ax.legend()
    ax.set_title("Nearest-neighbour separations (all 420 sources)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only-pair", type=int, default=None,
                   help="Only generate catalogs for this SKYCELL_ID (for validation)")
    args = p.parse_args()

    OUT_CAT.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    skycells = Table.read(SKYCELLS_IN, format="ascii.ecsv").to_pandas()
    if args.only_pair is not None:
        skycells = skycells[skycells["SKYCELL_ID"] == args.only_pair]
    print(f"Generating catalogs for {len(skycells)} skycell(s)…")

    summary = []
    for _, row in skycells.iterrows():
        idx = int(row["skycell_idx"])
        name = str(row["skycell_name"])
        # Deterministic per-skycell seed so rerunning one skycell doesn't
        # shift another's positions.
        seed = RNG_SEED + idx % 100000
        stars, galaxies = catalog_for_skycell(idx, name, seed)
        stars_path = OUT_CAT / f"skycell_{name}_stars.parquet"
        gal_path = OUT_CAT / f"skycell_{name}_galaxies.parquet"
        write_parquet(stars, stars_path)
        write_parquet(galaxies, gal_path)
        qa_plot(idx, name, stars, galaxies,
                OUT_PLOTS / f"grid_{name}.png")
        summary.append({
            "SKYCELL_ID": int(row["SKYCELL_ID"]),
            "skycell_idx": idx,
            "skycell_name": name,
            "n_stars": len(stars),
            "n_galaxies": len(galaxies),
            "mag_min": float(stars["mag"].min()),
            "mag_max": float(stars["mag"].max()),
            "stars_path": str(stars_path.relative_to(REPO)),
            "galaxies_path": str(gal_path.relative_to(REPO)),
        })
        print(f"  {name}: {len(stars)} sources each → {stars_path.name}, "
              f"{gal_path.name}")

    summary_df = pd.DataFrame(summary)
    summary_path = OUT_CAT.parent / "catalogs_index.ecsv"
    Table.from_pandas(summary_df).write(
        summary_path, format="ascii.ecsv", overwrite=True
    )
    print(f"\nWrote index: {summary_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
