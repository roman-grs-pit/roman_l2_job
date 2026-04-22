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

# Source grid: 21 mag bins × 10 sources = 210 per skycell per type
MAG_MIN, MAG_MAX, MAG_STEP = 23.0, 25.0, 0.1
SOURCES_PER_MAG = 10
BANDPASS_COL = "F158"

# Galaxy shape (per the plan): circular exponential, n=1, r_h=0.275"
GAL_N = 1.0
GAL_REFF_ARCSEC = 0.275
GAL_PA = 0.0
GAL_BA = 1.0

RNG_SEED = 20260422


def build_mag_array() -> np.ndarray:
    mags = np.arange(MAG_MIN, MAG_MAX + 0.5 * MAG_STEP, MAG_STEP)
    assert len(mags) == 21, f"expected 21 bins, got {len(mags)}"
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
    """Build (stars_df, galaxies_df) for one skycell."""
    cells = SkyCells([skycell_idx])
    wcs = cells.wcs[0]
    nx, ny = cells.pixel_shape  # (5000, 5000)

    n_total = 21 * SOURCES_PER_MAG  # 210
    pix = sobol_pixel_positions(nx, ny, n_total, seed=rng_seed)
    # WCS convention: (x, y) → (ra, dec) in deg
    ra, dec = wcs(pix[:, 0], pix[:, 1])
    ra = np.asarray(ra); dec = np.asarray(dec)

    rng = np.random.default_rng(rng_seed + 1)
    mags = build_mag_array()
    order = rng.permutation(n_total)
    mags = mags[order]  # shuffled magnitudes
    flux_maggies = np.power(10.0, -mags / 2.5)

    stars = pd.DataFrame({
        "ra": ra,
        "dec": dec,
        "type": ["PSF"] * n_total,
        "n": np.zeros(n_total, dtype=np.float32),
        "half_light_radius": np.zeros(n_total, dtype=np.float32),
        "pa": np.zeros(n_total, dtype=np.float32),
        "ba": np.ones(n_total, dtype=np.float32),
        BANDPASS_COL: flux_maggies,
        "mag": mags,  # tracked for our own bookkeeping
        "src_index": np.arange(n_total, dtype=np.int32),
        "pix_x": pix[:, 0].astype(np.float32),
        "pix_y": pix[:, 1].astype(np.float32),
    })

    galaxies = stars.copy()
    galaxies["type"] = "SER"
    galaxies["n"] = np.full(n_total, GAL_N, dtype=np.float32)
    galaxies["half_light_radius"] = np.full(n_total, GAL_REFF_ARCSEC,
                                             dtype=np.float32)
    galaxies["pa"] = np.full(n_total, GAL_PA, dtype=np.float32)
    galaxies["ba"] = np.full(n_total, GAL_BA, dtype=np.float32)

    return stars, galaxies


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write parquet with the exact column dtypes romanisim expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t = Table.from_pandas(df)
    t.write(path, overwrite=True)


def qa_plot(skycell_idx: int, skycell_name: str, stars_df: pd.DataFrame,
            out_path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: pixel layout, coloured by magnitude
    ax = axes[0]
    sc = ax.scatter(stars_df["pix_x"], stars_df["pix_y"],
                    c=stars_df["mag"], cmap="viridis", s=20, edgecolor="k", lw=0.2)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    ax.set_aspect("equal")
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 5000)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{skycell_name} (idx {skycell_idx}) — Sobol layout")
    # Draw the 100-px skycell-overlap border (dotted) and the
    # injection boundary (dashed)
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
    ax.legend(loc="lower left", fontsize=7)
    fig.colorbar(sc, ax=ax, label="F158 magnitude")

    # Right: per-mag-bin position scatter to show uniform coverage per bin
    ax = axes[1]
    bins = np.round(stars_df["mag"].unique(), 2)
    cmap = plt.get_cmap("viridis")
    for mag in bins:
        sub = stars_df[np.isclose(stars_df["mag"], mag)]
        ax.scatter(sub["pix_x"], sub["pix_y"], s=36,
                   color=cmap((mag - MAG_MIN) / (MAG_MAX - MAG_MIN)),
                   edgecolor="k", lw=0.3, alpha=0.9)
    ax.set_xlabel("pixel x")
    ax.set_ylabel("pixel y")
    ax.set_aspect("equal")
    ax.set_xlim(0, 5000)
    ax.set_ylim(0, 5000)
    ax.grid(True, alpha=0.3)
    ax.set_title("per-magnitude bin coverage (same colouring)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only-pair", type=int, default=None,
                   help="Only generate catalogs for this PAIR_ID (for validation)")
    args = p.parse_args()

    OUT_CAT.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    skycells = Table.read(SKYCELLS_IN, format="ascii.ecsv").to_pandas()
    if args.only_pair is not None:
        skycells = skycells[skycells["PAIR_ID"] == args.only_pair]
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
        qa_plot(idx, name, stars,
                OUT_PLOTS / f"grid_{name}.png")
        summary.append({
            "PAIR_ID": int(row["PAIR_ID"]),
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
