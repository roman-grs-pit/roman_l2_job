#!/usr/bin/env python
"""Phase 2: HLWAS expansion for each selected skycell.

For each skycell in `catalogs/detection/selected_skycells.ecsv`
(produced by Phase 1), scan all HLWAS F158 (Wide-Field2) pointings
within `HLWAS_EXPAND_RADIUS_DEG` of the skycell centre and keep every
(pointing, SCA) pair whose CRDS-backed WCS footprint overlaps the
skycell.

All expanded sims use the skycell's chosen SIM_DATE (from Phase 1,
~90° solar elongation) so zodiacal background stays fixed across
every L2 that contributes to the mosaic — the `PASS` an HLWAS visit
originally belonged to no longer drives the sim date.

Outputs
-------
catalogs/detection/pointings_sca_to_simulate.ecsv
    One row per (pointing, SCA) to simulate, carrying RA/Dec/PA/SIM_DATE
    directly so Phase 4a needs no further lookup.

output/detection/phase2/coverage_sky_<ID>.png
    Per-skycell SCA-footprint overlay on the selected skycell.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
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
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from romancal.skycell.match import find_skycell_matches
from romancal.skycell.skymap import SKYMAP, SkyCells

REPO = Path(__file__).resolve().parents[2]
SKYCELLS_IN = REPO / "catalogs/detection/selected_skycells.ecsv"
HLWAS_ECSV = REPO / "catalogs/HLWAS.sim.ecsv"
OUT_CAT = REPO / "catalogs/detection"
OUT_PLOTS = REPO / "output/detection/phase2"

# Coarse HLWAS search radius around each selected skycell. 0.6° covers
# any pointing whose outermost SCA could reach a 4.6′ skycell at the
# centre; actual overlap is checked per-SCA via the CRDS WCS.
HLWAS_EXPAND_RADIUS_DEG = 0.6


def sca_sky_corners(wcs, n_pix: int = roman.n_pix) -> np.ndarray:
    """Return the 4 SCA corner positions in (RA_deg, Dec_deg)."""
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

    Same code path romanisim-make-image --usecrds takes (boresight=False,
    since HLWAS pointings specify the WFI aperture centre)."""
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


def _ra_rel(ra, ra0):
    """RA − ra0 wrapped to [-180, 180]°. Plotting helper for RA~0 cells."""
    return (np.asarray(ra) - ra0 + 180.0) % 360.0 - 180.0


def _hlwas_nearby_exposures(skycell_ra: float, skycell_dec: float,
                             hlwas: pd.DataFrame,
                             radius_deg: float = HLWAS_EXPAND_RADIUS_DEG
                             ) -> pd.DataFrame:
    """HLWAS F158 exposure rows within `radius_deg` of (skycell_ra, skycell_dec)."""
    dra = np.minimum(np.abs(hlwas["RA"] - skycell_ra),
                     360.0 - np.abs(hlwas["RA"] - skycell_ra))
    within = ((dra * np.cos(np.deg2rad(skycell_dec))) < radius_deg) \
             & (np.abs(hlwas["DEC"] - skycell_dec) < radius_deg)
    return hlwas[within].copy()


def _expand_one_skycell(args):
    """Worker: return all (exposure, SCA) rows whose CRDS WCS overlaps
    the target skycell. Module-level for fork-pool pickling."""
    skycell_idx, skycell_name, cand_rows, sim_date, sid = args
    out = []
    date = Time(sim_date)
    for _, row in cand_rows.iterrows():
        for sca in range(1, 19):
            wcs = _build_sca_wcs_crds(row["RA"], row["DEC"], row["PA"],
                                       date, sca)
            corners = sca_sky_corners(wcs)
            matches = find_skycell_matches([tuple(r) for r in corners])
            if skycell_idx in matches:
                out.append({
                    "SKYCELL_ID": int(sid),
                    "PASS": int(row["PASS"]),
                    "SEGMENT": int(row["SEGMENT"]),
                    "OBSERVATION": int(row["OBSERVATION"]),
                    "VISIT": int(row["VISIT"]),
                    "EXPOSURE": int(row["EXPOSURE"]),
                    "SCA": int(sca),
                    "RA": float(row["RA"]),
                    "DEC": float(row["DEC"]),
                    "PA": float(row["PA"]),
                    "SIM_DATE": sim_date,
                    "skycell_idx": int(skycell_idx),
                    "skycell_name": skycell_name,
                })
    return out


def expand_to_full_hlwas(selected: pd.DataFrame, workers: int = 4
                          ) -> pd.DataFrame:
    """Fan out over skycells in a fork pool; return the combined
    (pointing, SCA) → skycell table."""
    hlwas = Table.read(HLWAS_ECSV, format="ascii.ecsv").to_pandas()
    hlwas = hlwas[(hlwas["BANDPASS"] == "F158")
                  & (hlwas["TARGET_NAME"] == "Wide-Field2")].copy()

    tasks = []
    for _, s in selected.iterrows():
        sid = int(s["SKYCELL_ID"])
        cand = _hlwas_nearby_exposures(s["skycell_ra"], s["skycell_dec"], hlwas)
        tasks.append((int(s["skycell_idx"]), str(s["skycell_name"]),
                      cand, str(s["SIM_DATE"]), sid))
        print(f"  skycell {sid} ({s['skycell_name']}): "
              f"{len(cand)} candidate HLWAS exposures")

    print(f"\nExpanding within {HLWAS_EXPAND_RADIUS_DEG}° of each skycell "
          f"(CRDS WCS × 18 SCAs per candidate)…")
    t0 = time.time()
    with mp.get_context("fork").Pool(processes=workers) as pool:
        rows = []
        for i, out in enumerate(
                pool.imap_unordered(_expand_one_skycell, tasks), 1):
            rows.extend(out)
            print(f"  [{i}/{len(tasks)}] +{len(out)} (pointing, SCA) rows; "
                  f"total {time.time()-t0:.1f}s", flush=True)
    df = pd.DataFrame(rows).drop_duplicates(
        ["SKYCELL_ID", "PASS", "SEGMENT", "OBSERVATION", "VISIT",
         "EXPOSURE", "SCA"]
    ).reset_index(drop=True)
    return df


def plot_skycell_coverage(sid: int, selected_row: pd.Series,
                           need_sim: pd.DataFrame, out_path: Path) -> None:
    sub = need_sim[need_sim["SKYCELL_ID"] == sid]
    if sub.empty:
        return
    ra0, dec0 = selected_row["skycell_ra"], selected_row["skycell_dec"]
    date = Time(selected_row["SIM_DATE"])
    fig, ax = plt.subplots(figsize=(8, 8))
    sc_half = 0.077 / 2  # 4.6′ skycell
    ax.add_patch(Rectangle((-sc_half, -sc_half), 2*sc_half, 2*sc_half,
                             fill=False, edgecolor="tab:red", lw=2, zorder=5))
    for _, row in sub.drop_duplicates(
            ["PASS", "SEGMENT", "VISIT", "EXPOSURE", "SCA"]).iterrows():
        wcs = _build_sca_wcs_crds(row["RA"], row["DEC"], row["PA"],
                                    date, int(row["SCA"]))
        c = sca_sky_corners(wcs)
        c[:, 0] = _ra_rel(c[:, 0], ra0)
        c[:, 1] = c[:, 1] - dec0
        xy = np.vstack([c, c[:1]])
        ax.plot(xy[:, 0], xy[:, 1], lw=0.5, alpha=0.4)
    n_l2 = sub.drop_duplicates(
        ["PASS", "SEGMENT", "VISIT", "EXPOSURE", "SCA"]).shape[0]
    ax.set_xlabel("Δ RA × cos(Dec) (deg)")
    ax.set_ylabel("Δ Dec (deg)")
    ax.set_aspect("equal")
    ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6)
    ax.grid(alpha=0.3)
    ax.set_title(f"skycell {sid} — {selected_row['skycell_name']} "
                 f"({n_l2} L2s overlap)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plots-only", action="store_true",
                   help="skip the expansion step; reuse the cached "
                        "pointings_sca_to_simulate.ecsv")
    args = p.parse_args()

    OUT_CAT.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    selected = Table.read(SKYCELLS_IN, format="ascii.ecsv").to_pandas()
    print(f"Loaded {len(selected)} selected skycells.")
    print(selected[["SKYCELL_ID", "skycell_idx", "skycell_name",
                    "skycell_ra", "skycell_dec", "ECL_LAT_DEG",
                    "SIM_DATE"]].to_string(index=False))

    sim_ecsv = OUT_CAT / "pointings_sca_to_simulate.ecsv"
    if args.plots_only and sim_ecsv.exists():
        print(f"\n(plots-only: keeping existing "
              f"{sim_ecsv.relative_to(REPO)})")
        need_sim = Table.read(sim_ecsv, format="ascii.ecsv").to_pandas()
    else:
        need_sim = expand_to_full_hlwas(selected)
        Table.from_pandas(need_sim).write(sim_ecsv, format="ascii.ecsv",
                                           overwrite=True)

    dedup_key = ["PASS", "SEGMENT", "OBSERVATION", "VISIT", "EXPOSURE", "SCA"]
    n_total = need_sim.drop_duplicates(dedup_key).shape[0]
    print(f"\nTotal L2 files to simulate: {n_total}")
    by_sid = need_sim.groupby("SKYCELL_ID").apply(
        lambda g: g.drop_duplicates(dedup_key).shape[0],
        include_groups=False,
    )
    print("  per-skycell counts:")
    for sid, cnt in by_sid.items():
        print(f"    skycell {int(sid)}: {cnt} L2 files")

    print("\nWriting per-skycell coverage plots…")
    for _, s in selected.iterrows():
        sid = int(s["SKYCELL_ID"])
        out = OUT_PLOTS / f"coverage_sky_{sid}.png"
        plot_skycell_coverage(sid, s, need_sim, out)
        print(f"  {out.name}")


if __name__ == "__main__":
    main()
