#!/usr/bin/env python
"""Phase 4a validation: inspect the L3 mosaic for one pair.

Checks we care about:
- depth (n-exposure map) in the core reaches the expected value (6 for our design)
- weight map tracks depth
- truth positions overlay sensibly on the image
- image has no obvious edge/rejection artifacts

Usage:  pixi run python scripts/detection/04d_validate_mosaic.py --cell 11119
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
import roman_datamodels.datamodels as rdm

REPO = Path(__file__).resolve().parents[2]
OUT_PLOTS = REPO / "output/detection/phase4a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True)
    args = ap.parse_args()

    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    # Find the coadd
    l3_dir = REPO / "output/detection/l3"
    candidates = sorted(l3_dir.glob(f"sky_{args.cell}_*_coadd.asdf"))
    if not candidates:
        raise SystemExit(f"No coadd found in {l3_dir} for pair {args.cell}")
    coadd_path = candidates[0]
    print(f"Loading {coadd_path.relative_to(REPO)}")

    model = rdm.open(coadd_path)
    data = np.asarray(model.data)
    weight = np.asarray(model.weight)
    # Roman coadd also has a 'context' array giving per-pixel n_exposures
    context = np.asarray(model.context) if hasattr(model, "context") else None
    if context is not None and context.ndim == 3:
        # context is a bit-mask stack per plane; sum of set bits ~ depth
        depth = np.zeros(context.shape[-2:], dtype=int)
        for plane in context:
            depth += np.array(
                [bin(int(x)).count("1") for x in plane.ravel()]
            ).reshape(plane.shape)
    else:
        depth = None

    # Truth positions (RA/Dec)
    skycell_name = coadd_path.stem.split("_")[-2]  # sky_<PID>_<relid>_<skycell>_coadd
    # Fallback: read from selected_skycells.ecsv
    sk = Table.read(REPO / "catalogs/detection/selected_skycells.ecsv",
                    format="ascii.ecsv").to_pandas()
    skycell_name = str(
        sk[sk["SKYCELL_ID"] == args.cell]["skycell_name"].iloc[0])
    stars = Table.read(
        REPO / f"catalogs/detection/catalogs/skycell_{skycell_name}_stars.parquet"
    ).to_pandas()
    gals = Table.read(
        REPO / f"catalogs/detection/catalogs/skycell_{skycell_name}_galaxies.parquet"
    ).to_pandas()

    # Convert truth RA/Dec to pixel via mosaic WCS
    wcs = model.meta.wcs
    sx, sy = wcs.world_to_pixel_values(stars["ra"].to_numpy(),
                                         stars["dec"].to_numpy())
    gx, gy = wcs.world_to_pixel_values(gals["ra"].to_numpy(),
                                         gals["dec"].to_numpy())

    # --- Image stats ---
    mean, med, std = sigma_clipped_stats(data, sigma=3, maxiters=3)
    print(f"Image stats: median={med:.3f}, sigma={std:.3f} "
          f"(MJy/sr units from MosaicPipeline)")
    print(f"Shape: {data.shape}")
    print(f"Weight range: {weight.min():.2f} .. {weight.max():.2f} "
          f"(median {np.median(weight):.2f})")
    if depth is not None:
        print(f"Depth map: max = {depth.max()}, median (interior) = "
              f"{np.median(depth[200:-200, 200:-200])}")

    # --- Plot: 2x2 grid ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    # Image
    ax = axes[0, 0]
    vmin, vmax = med - 3 * std, med + 10 * std
    im = ax.imshow(data, origin="lower", cmap="gray_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(f"L3 image — skycell {skycell_name}")
    fig.colorbar(im, ax=ax, shrink=0.7, label="MJy/sr")

    # Weight map
    ax = axes[0, 1]
    im = ax.imshow(weight, origin="lower", cmap="viridis",
                   interpolation="nearest")
    ax.set_title("weight map")
    fig.colorbar(im, ax=ax, shrink=0.7)

    # Depth map (if available)
    ax = axes[1, 0]
    if depth is not None:
        im = ax.imshow(depth, origin="lower", cmap="viridis",
                       interpolation="nearest", vmin=0, vmax=int(depth.max()))
        ax.set_title(f"mosaic depth map (max = {depth.max()})")
        fig.colorbar(im, ax=ax, shrink=0.7, label="n exposures")
    else:
        ax.set_title("depth map — no context array in model")
        ax.axis("off")

    # Truth overlay
    ax = axes[1, 1]
    im = ax.imshow(data, origin="lower", cmap="gray_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.scatter(sx, sy, facecolor="none", edgecolor="tab:blue", s=25, lw=0.6,
               label=f"{len(stars)} stars (PSF)")
    ax.scatter(gx, gy, facecolor="none", edgecolor="tab:red", s=25, lw=0.6,
               marker="s", label=f"{len(gals)} galaxies (SER)")
    ax.set_title("truth positions overlaid")
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7, label="MJy/sr")

    for ax in axes.ravel():
        ax.set_xlabel("x (pix)")
        ax.set_ylabel("y (pix)")
    fig.suptitle(
        f"Phase 4a — pair {args.cell} validation ({skycell_name})",
        fontsize=14
    )
    fig.tight_layout()
    out = OUT_PLOTS / f"validate_sky_{args.cell}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Wrote {out.relative_to(REPO)}")

    # Thumbnail zoom around the brightest truth source — at mag 23 it
    # should be obvious in any depth-2+ mosaic. If not, something is
    # wrong with catalog ingestion.
    i = int(stars["mag"].idxmin())
    cx, cy = int(sx[i]), int(sy[i])
    half = 400
    xs = slice(max(cx - half, 0), min(cx + half, data.shape[1]))
    ys = slice(max(cy - half, 0), min(cy + half, data.shape[0]))
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(data[ys, xs], origin="lower", cmap="gray_r",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    # Overlay truth sources inside the zoom
    in_box = (sx > cx - half) & (sx < cx + half) & (sy > cy - half) & (sy < cy + half)
    ax.scatter(sx[in_box] - (cx - half), sy[in_box] - (cy - half),
               facecolor="none", edgecolor="tab:blue", s=80, lw=0.8,
               label="stars")
    in_boxg = (gx > cx - half) & (gx < cx + half) & (gy > cy - half) & (gy < cy + half)
    ax.scatter(gx[in_boxg] - (cx - half), gy[in_boxg] - (cy - half),
               facecolor="none", edgecolor="tab:red", s=80, lw=0.8,
               marker="s", label="galaxies")
    ax.set_title(f"Zoom around star {i} (mag {stars['mag'].iloc[i]:.2f})")
    ax.legend()
    fig.colorbar(im, ax=ax, label="MJy/sr")
    fig.tight_layout()
    out2 = OUT_PLOTS / f"validate_sky_{args.cell}_zoom.png"
    fig.savefig(out2, dpi=140)
    plt.close(fig)
    print(f"Wrote {out2.relative_to(REPO)}")


if __name__ == "__main__":
    main()
