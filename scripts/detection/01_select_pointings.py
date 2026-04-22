#!/usr/bin/env python
"""Phase 1: select pointing pairs for the L3 detection-efficiency study.

Reads `catalogs/HLWAS.sim.ecsv`, restricts to F158, identifies pairs of
visits at identical (RA, Dec, PA) — these are the survey-design
re-observations (PASS=(15,16) in HLWAS) — computes a solar-elongation
window where the target sits in [90°, 115°] during 2026, and selects a
small well-spread subset.

Output
------
catalogs/detection/selected_pairs.ecsv
    One row per *visit* in the selected pairs. Two visits per pair ×
    ~5 pairs ≈ ~10 rows. Columns include RA, Dec, PA, SEGMENT, PASS,
    VISIT, MA_TABLE_NUMBER, EXPOSURE_TIME, plus the chosen sim date
    and its computed solar elongation + ecliptic coordinates. The
    PAIR_ID column ties the two visits of one pair together.

catalogs/detection/selected_pointings.ecsv
    One row per *pointing* (dithered exposure) = ~30 rows for 5 pairs.
    This is the file Phase 2 consumes. Every column from HLWAS is
    preserved; PAIR_ID, SIM_DATE, SOLAR_ELONG_DEG, ECL_LAT_DEG,
    ECL_LON_DEG are appended.

output/detection/phase1/elongation_windows.png
    For each selected pair, solar elongation vs day-of-year in 2026,
    with the 90°/115° band shaded and the chosen date marked.

output/detection/phase1/sky_map.png
    All candidate (15,16) pairs on the sky (grey), with the selected
    ones highlighted.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, get_sun
from astropy.table import Table
from astropy.time import Time
import astropy.units as u

# get_sun-based separation triggers a benign NonRotationTransformationWarning
# when batched over many coordinates; silence it for this clean-Sun case.
from astropy.coordinates.errors import NonRotationTransformationWarning
warnings.filterwarnings("ignore", category=NonRotationTransformationWarning)


REPO = Path(__file__).resolve().parents[2]
HLWAS = REPO / "catalogs/HLWAS.sim.ecsv"
OUT_CAT = REPO / "catalogs/detection"
OUT_PLOTS = REPO / "output/detection/phase1"

ELONG_MIN = 90.0
ELONG_MAX = 115.0
YEAR = 2026
N_PAIRS = 5  # number of pairs to select
MA_TABLE = 1007  # 107.52 s MA table — canonical HLWAS imaging cadence
DEC_MAX = 0.0  # stay in the main body of the HLWAS survey


def load_pair_candidates() -> pd.DataFrame:
    """Return one row per visit for all PASS=(15,16) F158 pairs."""
    t = Table.read(HLWAS, format="ascii.ecsv")
    df = t[t["BANDPASS"] == "F158"].to_pandas()
    vkey = ["PLAN", "PASS", "SEGMENT", "OBSERVATION", "VISIT"]
    vg = df.groupby(vkey)
    # Visit center = mean of the 3 dither positions
    centers = vg[["RA", "DEC", "PA"]].mean().reset_index()
    meta = vg.first().reset_index()[
        vkey + ["MA_TABLE_NUMBER", "EXPOSURE_TIME", "TARGET_NAME"]
    ]
    visits = centers.merge(meta, on=vkey)
    visits["ra_r"] = visits["RA"].round(5)
    visits["dec_r"] = visits["DEC"].round(5)
    # Keep only positions with exactly two visits
    pos_sizes = visits.groupby(["ra_r", "dec_r"]).size()
    pair_pos = pos_sizes[pos_sizes == 2].index
    pairs = visits.set_index(["ra_r", "dec_r"]).loc[pair_pos].reset_index()
    # Restrict to PASS pair == (15, 16)
    pair_pass = pairs.groupby(["ra_r", "dec_r"])["PASS"].apply(
        lambda s: tuple(sorted(s.tolist()))
    )
    wanted_pos = pair_pass[pair_pass == (15, 16)].index
    pairs = pairs.set_index(["ra_r", "dec_r"]).loc[wanted_pos].reset_index()
    # Restrict to the canonical HLWAS imaging MA table and to the main
    # survey body (south of the Dec~+2 edge).
    pairs = pairs[pairs["MA_TABLE_NUMBER"] == MA_TABLE]
    pairs = pairs[pairs["DEC"] < DEC_MAX]
    # Drop any positions where the Dec filter removed one visit but kept
    # the other (shouldn't happen since a pair shares Dec, but belt+braces):
    ok = pairs.groupby(["ra_r", "dec_r"]).size() == 2
    pairs = pairs.set_index(["ra_r", "dec_r"]).loc[ok[ok].index].reset_index()
    # Numeric PAIR_ID = integer rank over unique (ra_r, dec_r)
    unique_pos = pairs[["ra_r", "dec_r"]].drop_duplicates().reset_index(drop=True)
    unique_pos["PAIR_ID"] = unique_pos.index.astype(int)
    pairs = pairs.merge(unique_pos, on=["ra_r", "dec_r"])
    return pairs


def solar_elongation_curve(ra_deg: float, dec_deg: float, year: int = YEAR) -> pd.DataFrame:
    """Daily solar elongation of (ra, dec) across `year`."""
    start = Time(f"{year}-01-01T00:00:00", scale="utc")
    days = np.arange(0, 366)
    times = start + days * u.day
    sun = get_sun(times)
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    elong = sun.separation(target).deg
    return pd.DataFrame({"doy": days, "time": times.isot, "elong_deg": elong})


def choose_date(curve: pd.DataFrame, minimum: float = ELONG_MIN,
                maximum: float = ELONG_MAX) -> tuple[str, float]:
    """Return (ISO date, elongation) of the day with elongation closest to
    `minimum` while staying inside [minimum, maximum]. Returns (None, nan)
    if no day satisfies the window."""
    ok = curve[(curve["elong_deg"] >= minimum) & (curve["elong_deg"] <= maximum)]
    if ok.empty:
        return None, float("nan")
    # Prefer the day closest to the minimum (=90°) per the plan
    best = ok.iloc[(ok["elong_deg"] - minimum).abs().argmin()]
    return best["time"], float(best["elong_deg"])


def windows(curve: pd.DataFrame, minimum: float = ELONG_MIN,
            maximum: float = ELONG_MAX) -> list[tuple[int, int]]:
    """Contiguous runs of DOYs where elongation is inside the window,
    returned as (start_doy, end_doy) inclusive."""
    mask = (curve["elong_deg"] >= minimum) & (curve["elong_deg"] <= maximum)
    runs = []
    in_run = False
    start = None
    for d, m in zip(curve["doy"].values, mask.values):
        if m and not in_run:
            start = int(d); in_run = True
        elif not m and in_run:
            runs.append((start, int(d) - 1)); in_run = False
    if in_run:
        runs.append((start, int(curve["doy"].iloc[-1])))
    return runs


def select_pair_sample(pair_centers: pd.DataFrame, n: int,
                       rng_seed: int = 20260422) -> pd.DataFrame:
    """Pick `n` pairs that span ecliptic latitude then RA.

    Strategy: sort by ecliptic latitude (the main driver of zodi
    background), split into n roughly equal-count bins, and within each
    bin pick the pair whose RA is most extreme relative to already-chosen
    pairs (greedy max-min-separation in RA after lat binning).
    """
    rng = np.random.default_rng(rng_seed)
    pc = pair_centers.copy().sort_values("ECL_LAT_DEG").reset_index(drop=True)
    # Latitude bins — equal count
    bin_idx = np.linspace(0, len(pc), n + 1).astype(int)
    picked = []
    for i in range(n):
        cand = pc.iloc[bin_idx[i]:bin_idx[i + 1]]
        if cand.empty:
            continue
        if not picked:
            # Pick the median-RA candidate in the bin
            ra_sorted = cand.sort_values("RA")
            picked.append(ra_sorted.iloc[len(ra_sorted) // 2])
        else:
            # Maximise angular RA separation from already-picked pairs
            ras_picked = np.array([p["RA"] for p in picked])
            def dra(row):
                dr = np.minimum(
                    np.abs(row["RA"] - ras_picked),
                    360.0 - np.abs(row["RA"] - ras_picked),
                )
                return dr.min()
            cand = cand.assign(dra=cand.apply(dra, axis=1))
            picked.append(cand.sort_values("dra", ascending=False).iloc[0])
    return pd.DataFrame(picked).reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=N_PAIRS)
    p.add_argument("--year", type=int, default=YEAR)
    args = p.parse_args()

    OUT_CAT.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading HLWAS pair candidates…")
    pairs = load_pair_candidates()
    print(f"  {pairs['PAIR_ID'].nunique()} PASS=(15,16) F158 pairs")

    # Pair-level centers (identical for both visits), with ecliptic lat/lon
    centers = pairs.groupby("PAIR_ID").agg(
        RA=("RA", "first"), DEC=("DEC", "first"), PA=("PA", "first"),
    ).reset_index()
    sc = SkyCoord(ra=centers["RA"].values * u.deg, dec=centers["DEC"].values * u.deg)
    ecl = sc.transform_to("barycentrictrueecliptic")
    centers["ECL_LON_DEG"] = ecl.lon.deg
    centers["ECL_LAT_DEG"] = ecl.lat.deg

    # Compute elongation window existence for every candidate.
    # To keep this fast across 11k pairs, compute on a coarse 40-day grid
    # first; only candidates that ever enter [90, 115] survive.
    print("Screening candidates with elongation window…")
    from astropy.time import Time
    start = Time(f"{args.year}-01-01T00:00:00", scale="utc")
    doy_coarse = np.arange(0, 366, 10)
    times_coarse = start + doy_coarse * u.day
    sun_coarse = get_sun(times_coarse)
    elong_coarse = sun_coarse[None, :].separation(sc[:, None]).deg
    has_window = ((elong_coarse >= ELONG_MIN) & (elong_coarse <= ELONG_MAX)).any(axis=1)
    centers = centers[has_window].reset_index(drop=True)
    print(f"  {len(centers)} candidates have a valid date in {args.year}")

    # Select spread
    print(f"Selecting {args.n_pairs} well-spread pairs…")
    selected = select_pair_sample(centers, n=args.n_pairs)

    # For each selected pair, compute the exact elongation curve and pick a date
    sel_rows = []
    for _, row in selected.iterrows():
        curve = solar_elongation_curve(row["RA"], row["DEC"], year=args.year)
        date_iso, elong_deg = choose_date(curve)
        wins = windows(curve)
        sel_rows.append({
            "PAIR_ID": int(row["PAIR_ID"]),
            "RA": row["RA"], "DEC": row["DEC"], "PA": row["PA"],
            "ECL_LON_DEG": row["ECL_LON_DEG"], "ECL_LAT_DEG": row["ECL_LAT_DEG"],
            "SIM_DATE": date_iso,
            "SOLAR_ELONG_DEG": elong_deg,
            "WINDOWS_DOY": ";".join(f"{a}-{b}" for a, b in wins),
        })
    selected_full = pd.DataFrame(sel_rows).sort_values("PAIR_ID").reset_index(drop=True)

    # Write pair-level summary — one row per PAIR_ID
    Table.from_pandas(selected_full).write(
        OUT_CAT / "selected_pairs.ecsv", format="ascii.ecsv", overwrite=True
    )

    # Visit-level: PAIR_ID × (PASS, SEGMENT, OBSERVATION, VISIT) — used internally
    vis_rows = []
    for _, prow in selected_full.iterrows():
        pid = prow["PAIR_ID"]
        pair_visits = pairs[pairs["PAIR_ID"] == pid].sort_values("PASS")
        for _, v in pair_visits.iterrows():
            vis_rows.append({
                "PAIR_ID": pid,
                "PASS": int(v["PASS"]),
                "SEGMENT": int(v["SEGMENT"]),
                "OBSERVATION": int(v["OBSERVATION"]),
                "VISIT": int(v["VISIT"]),
                "RA": v["RA"], "DEC": v["DEC"], "PA": v["PA"],
                "MA_TABLE_NUMBER": int(v["MA_TABLE_NUMBER"]),
                "EXPOSURE_TIME": float(v["EXPOSURE_TIME"]),
                "TARGET_NAME": v["TARGET_NAME"],
                "SIM_DATE": prow["SIM_DATE"],
                "SOLAR_ELONG_DEG": prow["SOLAR_ELONG_DEG"],
                "ECL_LON_DEG": prow["ECL_LON_DEG"],
                "ECL_LAT_DEG": prow["ECL_LAT_DEG"],
            })
    selected_visits = pd.DataFrame(vis_rows)

    # Write pointing-level (per exposure) file — the one Phase 2 will consume
    t = Table.read(HLWAS, format="ascii.ecsv")
    full_f158 = t[t["BANDPASS"] == "F158"].to_pandas()
    keys = ["PLAN", "PASS", "SEGMENT", "OBSERVATION", "VISIT"]
    # Merge to get all 3 dithers per selected visit
    want = selected_visits[["PAIR_ID", "PASS", "SEGMENT", "OBSERVATION", "VISIT",
                            "SIM_DATE", "SOLAR_ELONG_DEG",
                            "ECL_LON_DEG", "ECL_LAT_DEG"]]
    pointings = full_f158.merge(want, on=["PASS", "SEGMENT", "OBSERVATION", "VISIT"])
    # PLAN is a merge dup key — we only merged on the 4-key, but if PLAN differs,
    # there would be two rows. Verify there isn't:
    assert pointings.groupby(keys + ["EXPOSURE"]).size().max() == 1, "pointing dup"
    pointings = pointings.sort_values(["PAIR_ID", "PASS", "EXPOSURE"]).reset_index(drop=True)
    Table.from_pandas(pointings).write(
        OUT_CAT / "selected_pointings.ecsv", format="ascii.ecsv", overwrite=True
    )

    # Summary printout
    print("\nSelected pairs:")
    print(selected_full.to_string(index=False))
    print(f"\n{len(selected_visits)} visits × 3 dithers = {len(pointings)} pointings")

    # ----- Plots -----
    import matplotlib.pyplot as plt

    # Elongation curves
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    for i, (_, prow) in enumerate(selected_full.iterrows()):
        curve = solar_elongation_curve(prow["RA"], prow["DEC"], year=args.year)
        ax.plot(curve["doy"], curve["elong_deg"], color=cmap(i % 10),
                label=f"pair {int(prow['PAIR_ID'])} "
                      f"(RA={prow['RA']:.1f}, Dec={prow['DEC']:.1f}, "
                      f"ecl_lat={prow['ECL_LAT_DEG']:+.1f}°)")
        # Mark chosen date
        if prow["SIM_DATE"]:
            doy_chosen = (Time(prow["SIM_DATE"]) - Time(f"{args.year}-01-01")).jd
            ax.plot(doy_chosen, prow["SOLAR_ELONG_DEG"], "o",
                    color=cmap(i % 10), markersize=10, markeredgecolor="k")
    ax.axhspan(ELONG_MIN, ELONG_MAX, color="k", alpha=0.08,
               label=f"[{ELONG_MIN:.0f}°, {ELONG_MAX:.0f}°]")
    ax.axhline(ELONG_MIN, color="k", ls="--", lw=0.7)
    ax.axhline(ELONG_MAX, color="k", ls="--", lw=0.7)
    ax.set_xlabel(f"Day of year {args.year}")
    ax.set_ylabel("Solar elongation (deg)")
    ax.set_ylim(0, 180)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Solar elongation vs date for the selected pairs")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "elongation_windows.png", dpi=140)
    plt.close(fig)

    # Sky map — all candidates (grey) + selected (red)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(centers["RA"], centers["DEC"], s=2, color="lightgray",
               label=f"{len(centers)} (15,16) pairs with valid date window")
    for _, prow in selected_full.iterrows():
        ax.scatter(prow["RA"], prow["DEC"], s=90, marker="*",
                   color="tab:red", edgecolor="k", lw=0.5, zorder=5)
        ax.annotate(f" {int(prow['PAIR_ID'])}",
                    (prow["RA"], prow["DEC"]),
                    fontsize=9)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title(f"HLWAS F158 (15,16) pairs — {args.n_pairs} selected")
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "sky_map.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote:\n  {OUT_CAT / 'selected_pairs.ecsv'}")
    print(f"  {OUT_CAT / 'selected_pointings.ecsv'}")
    print(f"  {OUT_PLOTS / 'elongation_windows.png'}")
    print(f"  {OUT_PLOTS / 'sky_map.png'}")


if __name__ == "__main__":
    main()
