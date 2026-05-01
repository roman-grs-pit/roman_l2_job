#!/usr/bin/env python3
"""Per-coadd pixel-depth statistics — QA artifact for the recovery-test pipeline.

For every coadd in `${output_base}/<tag>/mosaic/`:
  - read context bitmask + weight map
  - per-pixel depth via popcount, restricted to weight>0 pixels
  - emit n_pix, p05, median, mean, p95 to CSV

Also writes a histogram of the per-skycell median pixel depth.

Why per-pixel depth, not asn-member count: a skycell with 16 distinct
pointings ("count=16") doesn't necessarily mean every pixel sees 16
contributors. Most "high-count" cells have median pixel depth = 6
(design depth) or even 4, because the visit-overlap regions are smaller
fraction of the cell. Sorting / selecting by asn count is misleading; the
right metric is per-pixel depth from the coadd context bitmask.

Output: ${output_base}/<tag>/qa/coadd_depth_summary.csv
        ${output_base}/<tag>/qa/coadd_depth_distribution.png

Usage:
    pixi run python scripts/qa/coadd_depth_summary.py configs/<tag>.yaml
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import roman_datamodels as rdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _config import load_config


def _depth_stats(coadd_path: str) -> dict:
    m = rdm.open(coadd_path)
    ctx = np.asarray(m.context)
    w = np.asarray(m.weight)
    if ctx.ndim == 3:
        pc = np.zeros(ctx.shape[1:], dtype=np.int32)
        for layer in ctx:
            pc += np.bitwise_count(layer).astype(np.int32)
    else:
        pc = np.bitwise_count(ctx).astype(np.int32)
    inside = w > 0
    name = Path(coadd_path).name.replace("_coadd.asdf", "")
    if not inside.any():
        return dict(name=name, n_pix=0, dp05=0.0, dmedian=0.0, dmean=0.0, dp95=0.0)
    d = pc[inside]
    p05, p95 = np.percentile(d, [5, 95])
    return dict(
        name=name,
        n_pix=int(inside.sum()),
        dp05=float(p05),
        dmedian=float(np.median(d)),
        dmean=float(d.mean()),
        dp95=float(p95),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="path to configs/<tag>.yaml")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    base = Path(cfg.output_base) if getattr(cfg, "output_base", None) else Path("output")
    mosaic_dir = base / cfg.tag / "mosaic"
    qa_dir = base / cfg.tag / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    out_csv = qa_dir / "coadd_depth_summary.csv"
    out_png = qa_dir / "coadd_depth_distribution.png"

    coadds = sorted(str(p) for p in mosaic_dir.glob("*_coadd.asdf"))
    if not coadds:
        print(f"no coadds in {mosaic_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"processing {len(coadds)} coadds with {args.workers} workers...")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, r in enumerate(ex.map(_depth_stats, coadds, chunksize=8), 1):
            rows.append(r)
            if n % 200 == 0 or n == len(coadds):
                print(f"  {n}/{len(coadds)}")

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv}")

    medians = np.array([r["dmedian"] for r in rows])
    print("\nper-skycell median pixel depth distribution:")
    bins = np.arange(0, int(medians.max()) + 2)
    counts, _ = np.histogram(medians, bins=bins)
    for v, c in zip(bins[:-1], counts):
        print(f"  median={v:>2}: {c} skycells")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(medians, bins=np.arange(0, medians.max() + 1.5, 1.0), edgecolor="black")
    ax.set_xlabel("Median per-pixel depth")
    ax.set_ylabel("# skycells")
    ax.set_title(f"{cfg.tag}: per-skycell median pixel depth ({len(rows)} coadds)")
    ax.axvline(6, ls="--", color="r", label="design depth = 6")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
