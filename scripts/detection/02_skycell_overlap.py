#!/usr/bin/env python
"""Phase 2: skycell / SCA overlap analysis.

For every pointing (dithered exposure) in Phase 1's
`selected_pointings.ecsv`, compute the on-sky footprint of each of the
18 WFI SCAs and find which skycells the SCA overlaps using
`romancal.skycell.match.find_skycell_matches`.

Two aggregation levels are produced:

- **Pointing-level**: does pointing P contribute *any* SCA to skycell S?
  Per-pair "mosaic depth" = number of pointings (out of 6) that do.

- **SCA-level**: enumerate every (pointing, SCA, skycell) triple, so we
  know exactly which L2 files need to be simulated.

The script is purely geometric — no image data is generated. WCS is built
via `galsim.roman.getWCS` (its internal distortion model is accurate
enough for 4.6′ skycell-overlap decisions).

Outputs
-------
catalogs/detection/skycell_overlap_sca.ecsv
    One row per (PAIR_ID, POINTING_IDX, SCA, skycell) triple. This is
    the complete SCA-level table — lists every L2 file that would be
    needed to fill every overlapping skycell.

catalogs/detection/skycell_overlap_pointing.ecsv
    One row per (PAIR_ID, skycell). Columns include the list of
    pointing indices contributing, `n_pointings` (the depth), and
    `pointing_scas` — the distinct (pointing, SCA) pairs that must be
    simulated for this skycell.

catalogs/detection/selected_skycells.ecsv
    One skycell per pair (the deepest one nearest the pair's
    footprint centre); this is the Phase 3/4 target list.

catalogs/detection/pointings_sca_to_simulate.ecsv
    Per-(pointing, SCA) deduplicated list of what Phase 4 actually
    simulates — 6 pointings per pair but a pair may need multiple SCAs
    per pointing if the selected skycell straddles an SCA boundary.

output/detection/phase2/*.png
    Per-pair diagnostic plots.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import astropy.units as u
import galsim
import galsim.roman as roman
import numpy as np
import pandas as pd
import romanisim.wcs as rwcs
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from romancal.skycell.match import find_skycell_matches
from romancal.skycell.skymap import SKYMAP, SkyCells

REPO = Path(__file__).resolve().parents[2]
POINTINGS_IN = REPO / "catalogs/detection/selected_pointings.ecsv"
OUT_CAT = REPO / "catalogs/detection"
OUT_PLOTS = REPO / "output/detection/phase2"


def sca_sky_corners(wcs,
                    n_pix: int = roman.n_pix) -> np.ndarray:
    """Return the 4 SCA corner positions in (RA_deg, Dec_deg).

    Works for both galsim.CelestialWCS and romanisim.wcs.GWCS (both
    expose `toWorld(galsim.PositionD(x, y))` returning a galsim
    CelestialCoord)."""
    corners_pix = [(0, 0), (n_pix - 1, 0),
                   (n_pix - 1, n_pix - 1), (0, n_pix - 1)]
    out = np.empty((4, 2))
    for i, (xp, yp) in enumerate(corners_pix):
        sp = wcs.toWorld(galsim.PositionD(xp, yp))
        out[i, 0] = sp.ra.deg
        out[i, 1] = sp.dec.deg
    return out


def _build_sca_wcs_crds(ra: float, dec: float, pa: float,
                        date: Time, sca: int):
    """CRDS-distortion-accurate WCS for one (pointing, SCA).

    Matches the code path `romanisim-make-image --usecrds` takes, so
    the resulting WCS corners agree with the real L2 image WCS
    (boresight=False because HLWAS pointings specify the WFI aperture
    centre, not the telescope boresight)."""
    meta = {
        "instrument": {"name": "WFI",
                       "detector": f"WFI{sca:02d}",
                       "optical_element": "F158"},
        "exposure": {"start_time": date,
                     "type": "WFI_IMAGE",
                     "ma_table_number": 1007},
        "observation": {"start_time": date},
        "velocity_aberration": {"scale_factor": 1.0},
        "pointing": {},
        "wcsinfo": {},
    }
    rwcs.fill_in_parameters(
        meta, SkyCoord(ra * u.deg, dec * u.deg),
        pa_aper=pa, boresight=False
    )
    return rwcs.get_wcs(meta, usecrds=True)


def _overlap_for_one_pointing(args):
    """Worker: compute (pointing, SCA) → skycell triples for one pointing.

    Uses romanisim.wcs.get_wcs under CRDS distortion (boresight=False,
    matching `romanisim-make-image --usecrds`) so the corners agree with
    the real L2 WCS — the galsim.roman.getWCS analytical model was off
    by ~arcmin per SCA and flipped near-edge overlap decisions.

    Module-level for pickling into a fork-based process pool so
    module-state (SKYMAP, CRDS cache) is inherited via copy-on-write.
    """
    import time as _t
    pidx, p_dict = args
    t0 = _t.time()
    date = Time(p_dict["SIM_DATE"])
    out = []
    for sca in range(1, 19):
        wcs = _build_sca_wcs_crds(
            p_dict["RA"], p_dict["DEC"], p_dict["PA"], date, sca)
        corners = sca_sky_corners(wcs)
        for sc_idx in find_skycell_matches([tuple(r) for r in corners]):
            out.append({
                "PAIR_ID": int(p_dict["PAIR_ID"]),
                "POINTING_IDX": int(pidx),
                "PASS": int(p_dict["PASS"]),
                "VISIT": int(p_dict["VISIT"]),
                "EXPOSURE": int(p_dict["EXPOSURE"]),
                "SCA": int(sca),
                "skycell_idx": int(sc_idx),
            })
    dt = _t.time() - t0
    return pidx, out, dt


def build_sca_overlap_table(pointings: pd.DataFrame,
                            workers: int = 4) -> pd.DataFrame:
    """For every (pointing_idx, SCA) compute overlapping skycell indices.

    Parallel over pointings using a fork-based process pool so the
    1.3 GB SKYMAP reference (lazy-loaded on first `find_skycell_matches`
    call) is shared via copy-on-write.
    """
    import time as _t
    import multiprocessing as mp
    # Warm up SKYMAP in the parent so fork inherits it (saves ~70 s per
    # worker that would otherwise re-load the 1.3 GB reference).
    print("  warming up SKYMAP in parent (~60 s on first call)…", flush=True)
    t0 = _t.time()
    _ = find_skycell_matches([(0.0, 0.0), (0.01, 0.0),
                               (0.01, 0.01), (0.0, 0.01)])
    print(f"  SKYMAP warm in {_t.time() - t0:.1f} s; "
          f"dispatching {len(pointings)} pointings across "
          f"{workers} workers", flush=True)

    args = [(int(pidx), row.to_dict()) for pidx, row in pointings.iterrows()]
    rows = []
    ctx = mp.get_context("fork")
    t1 = _t.time()
    with ctx.Pool(processes=workers) as pool:
        for i, (pidx, triples, dt) in enumerate(
                pool.imap_unordered(_overlap_for_one_pointing, args), 1):
            rows.extend(triples)
            print(f"  [{i:2d}/{len(args)}] pointing {pidx}: "
                  f"{len(triples)} triples in {dt:.1f} s; "
                  f"total {_t.time() - t1:.1f} s", flush=True)
    return pd.DataFrame(rows)


def aggregate_to_pointing_level(sca_table: pd.DataFrame,
                                pointings: pd.DataFrame) -> pd.DataFrame:
    """Collapse SCA-level table to one row per (PAIR_ID, skycell)."""
    g = sca_table.groupby(["PAIR_ID", "skycell_idx"])
    rows = []
    for (pair_id, sc_idx), sub in g:
        pointing_idxs = sorted(sub["POINTING_IDX"].unique().tolist())
        pointing_scas = sorted(
            set((int(pi), int(sca))
                for pi, sca in zip(sub["POINTING_IDX"], sub["SCA"]))
        )
        rows.append({
            "PAIR_ID": int(pair_id),
            "skycell_idx": int(sc_idx),
            "n_pointings": len(pointing_idxs),
            "n_sca_events": len(pointing_scas),
            "pointings": ";".join(str(x) for x in pointing_idxs),
            "pointing_scas": ";".join(f"{pi}:{sca}" for pi, sca in pointing_scas),
        })
    return pd.DataFrame(rows)


def enrich_with_skycell_meta(pt_table: pd.DataFrame) -> pd.DataFrame:
    """Add skycell names, centres, and per-pair footprint-distance."""
    idxs = pt_table["skycell_idx"].unique().tolist()
    cells = SkyCells(idxs)
    meta = pd.DataFrame({
        "skycell_idx": idxs,
        "skycell_name": cells.names,
        "skycell_ra": cells.radec_centers[:, 0],
        "skycell_dec": cells.radec_centers[:, 1],
    })
    return pt_table.merge(meta, on="skycell_idx")


def select_skycells_per_pair(pt_table: pd.DataFrame,
                              pointings: pd.DataFrame,
                              target_depth: int = 6,
                              n_per_pair: int = 1) -> pd.DataFrame:
    """Pick the `n_per_pair` deepest + most-central skycells per pair.

    Priority order:
    1. Max `n_pointings` (up to `target_depth`) — most dithers contribute.
    2. Min `n_sca_events` at that depth — prefer skycells where each
       pointing contributes via a single SCA. When a pointing's 2 SCAs
       both overlap a skycell (SCA-boundary straddle), they cover
       disjoint regions of the skycell, so per-pixel depth is strictly
       less than the pointing count. Minimising `n_sca_events` gets us
       closer to true per-pixel depth = `n_pointings`.
    3. Nearest to the pair's nominal centre (tiebreak).
    """
    picks = []
    for pair_id, sub in pt_table.groupby("PAIR_ID"):
        p_centre = pointings[pointings["PAIR_ID"] == pair_id].iloc[0]
        ra0, dec0 = p_centre["RA"], p_centre["DEC"]
        max_depth = sub["n_pointings"].max()
        want = sub[sub["n_pointings"] == max(max_depth, target_depth)]
        if want.empty:
            want = sub[sub["n_pointings"] == max_depth]
        # Within that depth, prefer fewer SCA events (no SCA straddling)
        want = want[want["n_sca_events"] == want["n_sca_events"].min()]
        dra = np.minimum(np.abs(want["skycell_ra"] - ra0),
                         360.0 - np.abs(want["skycell_ra"] - ra0))
        ddec = np.abs(want["skycell_dec"] - dec0)
        d2 = (dra * np.cos(np.deg2rad(dec0))) ** 2 + ddec ** 2
        want = want.assign(_dist2=d2.values).sort_values("_dist2")
        picks.append(want.head(n_per_pair).drop(columns="_dist2"))
    return pd.concat(picks).reset_index(drop=True)


def _ra_rel(ra: np.ndarray | float, ra0: float) -> np.ndarray | float:
    """Return RA − ra0 wrapped to [−180°, +180°]. Used to draw near RA=0
    without the 0/360 discontinuity collapsing the axis."""
    return (np.asarray(ra) - ra0 + 180.0) % 360.0 - 180.0


def plot_pair_coverage(pair_id: int, pointings: pd.DataFrame,
                       sca_table: pd.DataFrame, pt_table: pd.DataFrame,
                       selected_skycell_idx: int, out_path: Path) -> None:
    """Per-pair coverage map: SCA footprints + skycell depth heat-map + pick.

    All RA coordinates are plotted as "RA − pair-centre RA", wrapped to
    [−180°, +180°]. Axis ticks are relabelled with the absolute RA so
    this is transparent to the reader. This is essential near RA = 0°:
    without the re-origination, polygon corners at RA 359.x and 0.x
    collapse the x-axis across 360°.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    ptr = pointings[pointings["PAIR_ID"] == pair_id]
    sca_rows = sca_table[sca_table["PAIR_ID"] == pair_id]
    pt_rows = pt_table[pt_table["PAIR_ID"] == pair_id]

    ra0, dec0 = ptr.iloc[0]["RA"], ptr.iloc[0]["DEC"]

    # SCA footprints (CRDS-distortion-accurate). Slower than galsim's
    # analytical WCS but consistent with find_skycell_matches results.
    sca_polygons = {}
    for pidx, p in ptr.iterrows():
        date = Time(p["SIM_DATE"])
        sca_polygons[int(pidx)] = {}
        for sca in range(1, 19):
            wcs = _build_sca_wcs_crds(p["RA"], p["DEC"], p["PA"], date, sca)
            sca_polygons[int(pidx)][sca] = sca_sky_corners(wcs)

    skycell_idxs = pt_rows["skycell_idx"].unique().tolist()
    cells = SkyCells(skycell_idxs)
    skycell_corners = cells.radec_corners
    depth_by_idx = dict(zip(pt_rows["skycell_idx"], pt_rows["n_pointings"]))

    fig, ax = plt.subplots(figsize=(9, 7.5))

    max_d = int(pt_rows["n_pointings"].max())
    cmap = plt.get_cmap("viridis")
    patches, colours = [], []
    for i, idx in enumerate(skycell_idxs):
        rel_corners = skycell_corners[i].copy()
        rel_corners[:, 0] = _ra_rel(rel_corners[:, 0], ra0)
        patches.append(Polygon(rel_corners, closed=True))
        colours.append(depth_by_idx[idx] / max_d)
    pc = PatchCollection(patches, alpha=0.45, edgecolor="white", lw=0.3)
    pc.set_array(np.array(colours))
    pc.set_cmap(cmap)
    pc.set_clim(0, 1)
    ax.add_collection(pc)

    pidx_list = sorted(sca_polygons.keys())
    pcmap = plt.get_cmap("tab10")
    for j, pidx in enumerate(pidx_list):
        for sca, corners in sca_polygons[pidx].items():
            corners_rel = corners.copy()
            corners_rel[:, 0] = _ra_rel(corners_rel[:, 0], ra0)
            xy = np.vstack([corners_rel, corners_rel[:1]])
            ax.plot(xy[:, 0], xy[:, 1], color=pcmap(j % 10),
                    lw=0.7, alpha=0.6)

    sel_i = skycell_idxs.index(int(selected_skycell_idx))
    sel_rel = skycell_corners[sel_i].copy()
    sel_rel[:, 0] = _ra_rel(sel_rel[:, 0], ra0)
    xy = np.vstack([sel_rel, sel_rel[:1]])
    ax.plot(xy[:, 0], xy[:, 1], color="tab:red", lw=2.5, zorder=6,
            label=f"selected skycell (idx {selected_skycell_idx}, depth "
                  f"{depth_by_idx[int(selected_skycell_idx)]})")

    ax.plot(0, dec0, "k+", markersize=14, mew=2, label="pair centre")

    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.invert_xaxis()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Pair {pair_id} — 6 pointings × 18 SCAs; skycells coloured by "
        f"mosaic depth (max = {max_d})"
    )
    # Relabel x-axis with absolute RA
    def _rel_to_abs(x, pos):
        return f"{(ra0 + x) % 360:.2f}"
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(_rel_to_abs))
    fig.colorbar(pc, ax=ax, ticks=np.arange(max_d + 1) / max_d,
                 label="depth fraction of max", shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-depth", type=int, default=6)
    p.add_argument("--n-skycells-per-pair", type=int, default=1,
                   help="how many depth-`target-depth` skycells to select per pair")
    p.add_argument("--force", action="store_true",
                   help="rebuild the SCA-level overlap table even if cached")
    p.add_argument("--plots-only", action="store_true",
                   help="only regenerate per-pair diagnostic plots from existing tables")
    args = p.parse_args()

    OUT_CAT.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    pointings = Table.read(POINTINGS_IN,
                           format="ascii.ecsv").to_pandas()
    pointings = pointings.sort_values(["PAIR_ID", "PASS", "EXPOSURE"]).reset_index(drop=True)
    print(f"Loaded {len(pointings)} pointings across "
          f"{pointings['PAIR_ID'].nunique()} pairs.")

    sca_ecsv = OUT_CAT / "skycell_overlap_sca.ecsv"
    pt_ecsv = OUT_CAT / "skycell_overlap_pointing.ecsv"
    if args.force or args.plots_only:
        use_cache = args.plots_only
    else:
        use_cache = sca_ecsv.exists() and pt_ecsv.exists()

    if use_cache:
        print(f"\nReading cached overlap tables from {OUT_CAT}/ "
              "(use --force to rebuild)…")
        sca_table = Table.read(sca_ecsv, format="ascii.ecsv").to_pandas()
        pt_table = Table.read(pt_ecsv, format="ascii.ecsv").to_pandas()
    else:
        print("\nBuilding SCA-level overlap table "
              f"({len(pointings)} pointings × 18 SCAs)…")
        sca_table = build_sca_overlap_table(pointings)
        print(f"  {len(sca_table):,} (pointing, SCA, skycell) triples")
        print(f"  distinct skycells touched: "
              f"{sca_table['skycell_idx'].nunique():,}")
        Table.from_pandas(sca_table).write(sca_ecsv, format="ascii.ecsv",
                                           overwrite=True)

        print("\nAggregating to pointing level…")
        pt_table = aggregate_to_pointing_level(sca_table, pointings)
        pt_table = enrich_with_skycell_meta(pt_table)
        print(f"  {len(pt_table):,} (pair, skycell) rows")
        depth_counts = (pt_table.groupby("PAIR_ID")["n_pointings"]
                        .apply(lambda s: (s == args.target_depth).sum()))
        print(f"  depth-{args.target_depth} skycells per pair:")
        for pid, cnt in depth_counts.items():
            print(f"    pair {pid:5d}: {cnt} skycells")
        Table.from_pandas(pt_table).write(pt_ecsv, format="ascii.ecsv",
                                          overwrite=True)

    print(f"\nSelecting {args.n_skycells_per_pair} depth-"
          f"{args.target_depth} skycell(s) per pair…")
    selected = select_skycells_per_pair(
        pt_table, pointings, args.target_depth, args.n_skycells_per_pair)
    Table.from_pandas(selected).write(
        OUT_CAT / "selected_skycells.ecsv",
        format="ascii.ecsv", overwrite=True
    )
    print("Selected:")
    print(selected[["PAIR_ID", "skycell_idx", "skycell_name",
                    "skycell_ra", "skycell_dec",
                    "n_pointings", "n_sca_events"]].to_string(index=False))

    # Per-selected-skycell list of (pointing, SCA) pairs to simulate
    need_sim_rows = []
    for _, s in selected.iterrows():
        sub = sca_table[
            (sca_table["PAIR_ID"] == s["PAIR_ID"])
            & (sca_table["skycell_idx"] == s["skycell_idx"])
        ]
        # Dedup (POINTING_IDX, SCA) — a single L2 file per combo
        dedup = sub[["PAIR_ID", "POINTING_IDX", "PASS", "VISIT",
                     "EXPOSURE", "SCA"]].drop_duplicates()
        dedup["skycell_idx"] = int(s["skycell_idx"])
        dedup["skycell_name"] = s["skycell_name"]
        need_sim_rows.append(dedup)
    need_sim = pd.concat(need_sim_rows).reset_index(drop=True)
    Table.from_pandas(need_sim).write(
        OUT_CAT / "pointings_sca_to_simulate.ecsv",
        format="ascii.ecsv", overwrite=True
    )

    # L2-count summary
    n_distinct_l2 = need_sim.drop_duplicates(["POINTING_IDX", "SCA"]).shape[0]
    print(f"\nL2 files to simulate: "
          f"{n_distinct_l2} (from {len(need_sim)} rows after dedup)")
    by_pair = need_sim.groupby("PAIR_ID").apply(
        lambda g: g.drop_duplicates(["POINTING_IDX", "SCA"]).shape[0],
        include_groups=False,
    )
    print("  per-pair counts:")
    for pid, cnt in by_pair.items():
        print(f"    pair {pid}: {cnt} L2 files")

    # Plots
    print("\nWriting per-pair diagnostic plots…")
    for _, s in selected.iterrows():
        out = OUT_PLOTS / f"coverage_pair_{int(s['PAIR_ID'])}.png"
        plot_pair_coverage(int(s["PAIR_ID"]), pointings,
                           sca_table, pt_table,
                           int(s["skycell_idx"]), out)
        print(f"  {out}")


if __name__ == "__main__":
    main()
