#!/usr/bin/env python3
"""Skycell coverage map — QA artifact for the recovery-test pipeline.

Two-panel figure: each skycell drawn as a tile colored by

  (left)  distinct-pointing count from the asn JSON (= number of unique
          (passno, segment, OBS, visit, exposure) tuples among members)
  (right) median per-pixel depth read from the coadd context bitmask
          (uses ${output_base}/<tag>/qa/coadd_depth_summary.csv if present;
          falls back to recomputing on demand)

The two panels look quite different in practice: a skycell with high
asn-count has many footprints intersecting it, but only the central
overlap regions see all those footprints. Median pixel depth is the
distribution's bulk and is the right "is this cell deep" metric.

Imaging visit centers and (optionally) GRISM spectro pointings are
overlaid as scatter markers.

Output: ${output_base}/<tag>/qa/coverage_map.png

Usage:
    pixi run python scripts/qa/coverage_map.py configs/<tag>.yaml
    pixi run python scripts/qa/coverage_map.py configs/<tag>.yaml \\
        --spectro /path/to/spectro_pointings.ecsv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _config import load_config

SCA_RX = re.compile(r"_wfi\d{2}_")
SKYCELL_NAME_RX = re.compile(r"\d{3}p\d{2}x\d{2,3}y\d{2,3}")


def _pkey(name: str):
    m = SCA_RX.search(name)
    return name[: m.start()] if m else None


def load_skycells(asn_dir: Path):
    cells = []
    for p in sorted(asn_dir.glob("*_asn.json")):
        with p.open() as f:
            d = json.load(f)
        wcs = d.get("skycell_wcs_info", {})
        keys = set()
        for prod in d.get("products", []):
            for mem in prod.get("members", []):
                k = _pkey(mem.get("expname", ""))
                if k is not None:
                    keys.add(k)
        cells.append(
            dict(
                name=wcs.get("name"),
                ra=wcs.get("ra_center"),
                dec=wcs.get("dec_center"),
                size_deg=wcs.get("nx", 0) * wcs.get("pixel_scale", 0),
                n_pointings=len(keys),
            )
        )
    return cells


def load_depth_csv(csv_path: Path) -> dict[str, float]:
    """Map skycell *short name* (NNNpNNxNNyNN) -> median pixel depth.
    The CSV uses the coadd basename (e.g. "acceptance_p_full_010p00x35y59_f158");
    the asn JSON's wcs.name uses just the short skycell tag, so we key on that."""
    if not csv_path.is_file():
        return {}
    out = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            m = SKYCELL_NAME_RX.search(r["name"])
            if m:
                out[m.group(0)] = float(r["dmedian"])
    return out


def load_pointings_ecsv(path: Path):
    import astropy.table as at

    t = at.Table.read(str(path), format="ascii.ecsv")
    return list(zip(t["RA"].tolist(), t["DEC"].tolist()))


def _draw_panel(ax, cells, values, *, vmin, vmax, cmap, cbar_label,
                visits=None, spectro=None, title=""):
    """One coverage-map panel: tiles per cell, colored by `values`."""
    ras = np.array([c["ra"] for c in cells])
    decs = np.array([c["dec"] for c in cells])
    sizes = np.array([c["size_deg"] for c in cells])
    half = sizes / 2.0
    cos_dec = np.cos(np.deg2rad(decs))

    patches = [
        mpatches.Rectangle(
            (ras[i] - half[i] / cos_dec[i], decs[i] - half[i]),
            sizes[i] / cos_dec[i],
            sizes[i],
        )
        for i in range(len(cells))
    ]
    pc = PatchCollection(patches, cmap=cmap, edgecolor="none")
    pc.set_array(np.asarray(values))
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)

    if visits is not None and len(visits):
        vra, vdec = zip(*visits)
        ax.plot(vra, vdec, "x", color="red", markersize=8, markeredgewidth=1.5,
                label=f"{len(visits)} imaging pointings")
    if spectro is not None and len(spectro):
        sra, sdec = zip(*spectro)
        ax.plot(sra, sdec, "+", color="cyan", markersize=10, markeredgewidth=1.5,
                label=f"{len(spectro)} GRISM pointings")
    if visits or spectro:
        ax.legend(loc="upper left", fontsize=8)

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.median(decs))))
    ax.invert_xaxis()
    ax.set_title(title)
    cb = plt.colorbar(pc, ax=ax, label=cbar_label, shrink=0.85)
    return cb


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config")
    ap.add_argument("--spectro", default=None,
                    help="optional ecsv with spectroscopic pointings to overlay")
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = Path(cfg.output_base) if getattr(cfg, "output_base", None) else Path("output")
    asn_dir = base / cfg.tag / "asn"
    qa_dir = base / cfg.tag / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    out_png = qa_dir / "coverage_map.png"
    depth_csv = qa_dir / "coadd_depth_summary.csv"

    cells = load_skycells(asn_dir)
    print(f"loaded {len(cells)} skycells")

    name_to_depth = load_depth_csv(depth_csv)
    if not name_to_depth:
        print(f"WARNING: {depth_csv} not found; right panel will be empty.")
        print(f"  run scripts/qa/coadd_depth_summary.py {args.config} first.")
    medians = np.array(
        [name_to_depth.get(c["name"], np.nan) for c in cells], dtype=float
    )
    counts = np.array([c["n_pointings"] for c in cells])

    # Imaging pointings: pointings_<tag>.ecsv at repo root; OBS=1 is
    # representative since OBS=2 has identical (RA, DEC, PA).
    visits = None
    pe = Path(f"pointings_{cfg.tag}.ecsv")
    if pe.is_file():
        import astropy.table as at
        t = at.Table.read(str(pe), format="ascii.ecsv")
        if "OBSERVATION" in t.colnames:
            t = t[t["OBSERVATION"] == 1]
        visits = list(zip(t["RA"].tolist(), t["DEC"].tolist()))

    spectro = load_pointings_ecsv(Path(args.spectro)) if args.spectro else None

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    cbar_max_count = max(int(np.nanmax(counts)), 16)
    _draw_panel(
        axes[0], cells, counts,
        vmin=0, vmax=cbar_max_count, cmap="viridis",
        cbar_label="distinct-pointing count",
        visits=visits, spectro=spectro,
        title=f"{cfg.tag}: distinct-pointing count per skycell",
    )
    if name_to_depth and np.isfinite(medians).any():
        cbar_max_depth = max(int(np.nanmax(medians)), 6)
        _draw_panel(
            axes[1], cells, medians,
            vmin=0, vmax=cbar_max_depth, cmap="viridis",
            cbar_label="median per-pixel depth",
            visits=visits, spectro=spectro,
            title=f"{cfg.tag}: median per-pixel depth (design = 6)",
        )
    else:
        axes[1].text(0.5, 0.5, "no depth CSV\n(run scripts/qa/coadd_depth_summary.py first)",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_axis_off()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
