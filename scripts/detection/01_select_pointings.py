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
BANDPASS = "F158"

# Pixel scale for converting e/arcsec²/s → e/pix/s (Roman WFI native).
WFI_PIXEL_SCALE_ARCSEC = 0.11

# Main-survey RA/Dec box for the interior-pool "spread in RA" picks.
# RA wraps through 0°, expressed as [RA_MIN_A, 360] ∪ [0, RA_MAX_B].
# Anchors (see `ANCHORS`) can live outside this box to broaden the
# ecliptic-latitude coverage of the final selection.
RA_MIN_A = 330.0
RA_MAX_B = 50.0
DEC_MIN = -25.0
DEC_MAX = -10.0

# Two anchor specs: each is a target (RA, Dec) plus a permissive window.
# The selector picks the candidate in each window closest to the target.
# These sit above/below the main box to reach lower/higher ecliptic
# latitude than the Dec-interior picks can.
ANCHORS = [
    # Southern edge: Dec ≈ -35
    {"name": "south",
     "ra_target": 20.0, "dec_target": -35.0,
     "ra_lo": 10.0, "ra_hi": 30.0,
     "dec_lo": -40.0, "dec_hi": -30.0},
    # Northern edge: Dec ≈ -2
    {"name": "north",
     "ra_target": 20.0, "dec_target": -2.0,
     "ra_lo": 10.0, "ra_hi": 30.0,
     "dec_lo": -5.0, "dec_hi": 0.0},
]

# "Interior" check: a pair is considered interior to the survey if there
# are at least INTERIOR_MIN F158 exposures within INTERIOR_RADIUS_DEG of
# its (wrap-aware) centre. Filters phantom centres and near-edge pairs.
INTERIOR_RADIUS_DEG = 2.0
INTERIOR_MIN = 200


def _circular_mean_deg(angles_deg: np.ndarray) -> float:
    """Wrap-aware mean of angles in degrees in [0, 360)."""
    rad = np.deg2rad(angles_deg)
    x = np.cos(rad).mean()
    y = np.sin(rad).mean()
    return (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0


def _in_ra_box(ra_deg: np.ndarray | pd.Series) -> np.ndarray:
    """True where RA is inside [RA_MIN_A, 360] ∪ [0, RA_MAX_B]."""
    ra = np.asarray(ra_deg)
    return (ra >= RA_MIN_A) | (ra <= RA_MAX_B)


def load_pair_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pair-visit rows, raw F158 rows).

    Uses a wrap-aware mean for the visit centre. Restricts to
    PASS=(15,16) F158 pairs with MA_TABLE_NUMBER=MA_TABLE. No RA/Dec
    box is applied here — selection windows are enforced at the pick
    stage. The interior filter (≥ INTERIOR_MIN F158 exposures within
    INTERIOR_RADIUS_DEG) is applied in `apply_interior_filter`.
    """
    t = Table.read(HLWAS, format="ascii.ecsv")
    df = t[t["BANDPASS"] == "F158"].to_pandas()
    vkey = ["PLAN", "PASS", "SEGMENT", "OBSERVATION", "VISIT"]
    # Wrap-aware visit centre (RA via circular mean; Dec is linear)
    vg = df.groupby(vkey)
    centers = vg["DEC"].mean().reset_index(name="DEC")
    centers["RA"] = vg["RA"].apply(
        lambda s: _circular_mean_deg(s.to_numpy())
    ).values
    centers["PA"] = vg["PA"].first().values
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
    # Canonical MA table
    pairs = pairs[pairs["MA_TABLE_NUMBER"] == MA_TABLE]
    # Drop positions where one visit survived but not both
    ok = pairs.groupby(["ra_r", "dec_r"]).size() == 2
    pairs = pairs.set_index(["ra_r", "dec_r"]).loc[ok[ok].index].reset_index()
    # Numeric PAIR_ID = integer rank over unique (ra_r, dec_r) within
    # the surviving set (so IDs are stable across reruns at a given
    # config but not across config changes)
    unique_pos = pairs[["ra_r", "dec_r"]].drop_duplicates().reset_index(drop=True)
    unique_pos["PAIR_ID"] = unique_pos.index.astype(int)
    pairs = pairs.merge(unique_pos, on=["ra_r", "dec_r"])
    return pairs, df


def apply_interior_filter(pairs: pd.DataFrame, f158: pd.DataFrame) -> pd.DataFrame:
    """Drop pairs whose centres sit too close to the survey boundary.

    Criterion: the number of F158 exposures within INTERIOR_RADIUS_DEG
    of the (wrap-aware) pair centre must be at least INTERIOR_MIN.
    """
    centres = pairs.groupby("PAIR_ID")[["RA", "DEC"]].first()
    ra0 = centres["RA"].to_numpy()
    dec0 = centres["DEC"].to_numpy()
    ra = f158["RA"].to_numpy()
    dec = f158["DEC"].to_numpy()
    # Small-angle Euclidean with RA wrap — OK within the box since
    # |dec| < 25° and radii are ~2°.
    cos_dec = np.cos(np.deg2rad(dec0))
    counts = np.zeros(len(centres), dtype=int)
    for i in range(len(centres)):
        dra = np.abs(ra - ra0[i])
        dra = np.minimum(dra, 360.0 - dra)
        ddec = np.abs(dec - dec0[i])
        near = (dra * cos_dec[i]) ** 2 + ddec ** 2 < INTERIOR_RADIUS_DEG ** 2
        counts[i] = int(near.sum())
    centres["n_f158_within_r"] = counts
    keep = centres[centres["n_f158_within_r"] >= INTERIOR_MIN].index
    return pairs[pairs["PAIR_ID"].isin(keep)].reset_index(drop=True)


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


def _pick_closest(df: pd.DataFrame, ra_target: float,
                  dec_target: float) -> pd.Series:
    """From `df`, pick the row minimising great-circle distance to the
    target position. Falls back to Euclidean since windows are narrow."""
    dra = np.minimum(np.abs(df["RA"] - ra_target),
                     360.0 - np.abs(df["RA"] - ra_target))
    ddec = np.abs(df["DEC"] - dec_target)
    cos_dec = np.cos(np.deg2rad(dec_target))
    d2 = (dra * cos_dec) ** 2 + ddec ** 2
    return df.iloc[int(np.argmin(d2.values))]


def select_pair_sample(pair_centers: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick n pairs: two anchors (edge Dec), then n-2 RA-spread picks in
    the main box at median Dec.

    The two anchors come from ANCHORS — a southern (Dec ≈ -35) and a
    northern (Dec ≈ -2) position at RA ≈ 20°, to give the final
    selection ecliptic-latitude diversity beyond the main-box Dec
    range. The remaining n-2 picks are drawn from the interior pool
    inside the main RA/Dec box, binned evenly along the wrap-aware
    RA axis; in each bin we take the median-Dec candidate.
    """
    pc = pair_centers.copy()

    picked_ids = []
    picked_rows = []

    # Anchors
    for anchor in ANCHORS:
        in_ra = (pc["RA"] >= anchor["ra_lo"]) & (pc["RA"] <= anchor["ra_hi"])
        in_dec = (pc["DEC"] >= anchor["dec_lo"]) & (pc["DEC"] <= anchor["dec_hi"])
        window = pc[in_ra & in_dec & ~pc["PAIR_ID"].isin(picked_ids)]
        if window.empty:
            raise SystemExit(
                f"No candidate in anchor window {anchor['name']}: "
                f"RA [{anchor['ra_lo']}, {anchor['ra_hi']}], "
                f"Dec [{anchor['dec_lo']}, {anchor['dec_hi']}]"
            )
        row = _pick_closest(window, anchor["ra_target"], anchor["dec_target"])
        picked_rows.append(row); picked_ids.append(int(row["PAIR_ID"]))

    # Remaining: RA-spread in main box at median Dec
    remaining = n - len(picked_rows)
    in_box = (
        ((pc["RA"] >= RA_MIN_A) | (pc["RA"] <= RA_MAX_B))
        & (pc["DEC"] >= DEC_MIN) & (pc["DEC"] <= DEC_MAX)
        & ~pc["PAIR_ID"].isin(picked_ids)
    )
    box = pc[in_box].copy()
    box["RA_LIN"] = np.where(
        box["RA"] >= RA_MIN_A,
        box["RA"] - RA_MIN_A,
        box["RA"] + (360.0 - RA_MIN_A),
    )
    box = box.sort_values(["RA_LIN", "PAIR_ID"]).reset_index(drop=True)
    bin_idx = np.linspace(0, len(box), remaining + 1).astype(int)
    for i in range(remaining):
        cand = box.iloc[bin_idx[i]:bin_idx[i + 1]]
        if cand.empty:
            continue
        mid_dec = cand["DEC"].median()
        row = cand.iloc[(cand["DEC"] - mid_dec).abs().values.argmin()]
        picked_rows.append(row); picked_ids.append(int(row["PAIR_ID"]))

    return pd.DataFrame(picked_rows).reset_index(drop=True)


def sky_level_per_pix_e_per_s(ra_deg: float, dec_deg: float,
                              date_iso: str) -> float:
    """Return the romanisim-predicted F158 sky level in e⁻/pix/s at the
    given (RA, Dec, date). Uses `galsim.roman.getSkyLevel`, scales by
    stray-light fraction (as romanisim.image does at line ~699), adds
    the F158 thermal background, and multiplies by the WFI pixel area.
    """
    import galsim
    import galsim.roman as roman
    import romanisim.bandpass

    bp_name = romanisim.bandpass.roman2galsim_bandpass[BANDPASS]
    bp = roman.getBandpasses(AB_zeropoint=True)[bp_name]
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    world = galsim.CelestialCoord(
        c.ra.rad * galsim.radians, c.dec.rad * galsim.radians)
    sky = roman.getSkyLevel(bp, world_pos=world,
                            date=Time(date_iso).datetime, exptime=1)
    sky *= (1.0 + roman.stray_light_fraction)
    sky_per_pix = sky * WFI_PIXEL_SCALE_ARCSEC ** 2
    sky_per_pix += roman.thermal_backgrounds[bp_name]
    return float(sky_per_pix)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-pairs", type=int, default=N_PAIRS)
    p.add_argument("--year", type=int, default=YEAR)
    args = p.parse_args()

    OUT_CAT.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    print("Loading HLWAS pair candidates…")
    pairs, f158 = load_pair_candidates()
    print(f"  {pairs['PAIR_ID'].nunique()} PASS=(15,16) F158 pairs (global)")
    pairs = apply_interior_filter(pairs, f158)
    print(f"  {pairs['PAIR_ID'].nunique()} remain after the interior filter "
          f"(>= {INTERIOR_MIN} F158 exposures within {INTERIOR_RADIUS_DEG}°)")

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

    # For each selected pair, compute the exact elongation curve, pick a
    # date, and predict the romanisim-F158 sky level at that date.
    sel_rows = []
    for _, row in selected.iterrows():
        curve = solar_elongation_curve(row["RA"], row["DEC"], year=args.year)
        date_iso, elong_deg = choose_date(curve)
        wins = windows(curve)
        sky_e_pix_s = sky_level_per_pix_e_per_s(row["RA"], row["DEC"], date_iso)
        sel_rows.append({
            "PAIR_ID": int(row["PAIR_ID"]),
            "RA": row["RA"], "DEC": row["DEC"], "PA": row["PA"],
            "ECL_LON_DEG": row["ECL_LON_DEG"], "ECL_LAT_DEG": row["ECL_LAT_DEG"],
            "SIM_DATE": date_iso,
            "SOLAR_ELONG_DEG": elong_deg,
            "SKY_E_PER_PIX_PER_S": sky_e_pix_s,
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
    sky_map = selected_full.set_index("PAIR_ID")["SKY_E_PER_PIX_PER_S"]
    selected_visits["SKY_E_PER_PIX_PER_S"] = selected_visits["PAIR_ID"].map(sky_map)
    want = selected_visits[["PAIR_ID", "PASS", "SEGMENT", "OBSERVATION", "VISIT",
                            "SIM_DATE", "SOLAR_ELONG_DEG",
                            "SKY_E_PER_PIX_PER_S",
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

    # Sky map — show the full HLWAS F158 cloud + our selection box + picks
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(f158["RA"], f158["DEC"], s=0.5, color="lightgray", alpha=0.5,
               label=f"{len(f158):,} HLWAS F158 exposures (all)")
    ax.scatter(centers["RA"], centers["DEC"], s=6, color="tab:blue",
               label=f"{len(centers)} (15,16) interior pairs in selection box",
               alpha=0.5)
    for _, prow in selected_full.iterrows():
        ax.scatter(prow["RA"], prow["DEC"], s=140, marker="*",
                   color="tab:red", edgecolor="k", lw=0.6, zorder=5)
        ax.annotate(f" {int(prow['PAIR_ID'])}",
                    (prow["RA"], prow["DEC"]), fontsize=9)
    # Main box (RA wraps)
    for ra_lo, ra_hi in [(RA_MIN_A, 360.0), (0.0, RA_MAX_B)]:
        ax.plot([ra_lo, ra_hi, ra_hi, ra_lo, ra_lo],
                [DEC_MIN, DEC_MIN, DEC_MAX, DEC_MAX, DEC_MIN],
                color="k", ls="--", lw=1.0, alpha=0.6,
                label="main RA-spread box" if ra_lo == RA_MIN_A else None)
    # Anchor windows
    for i, anchor in enumerate(ANCHORS):
        ax.plot([anchor["ra_lo"], anchor["ra_hi"], anchor["ra_hi"],
                 anchor["ra_lo"], anchor["ra_lo"]],
                [anchor["dec_lo"], anchor["dec_lo"], anchor["dec_hi"],
                 anchor["dec_hi"], anchor["dec_lo"]],
                color="tab:orange", ls=":", lw=1.0,
                label="anchor window" if i == 0 else None)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_xlim(360, 0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", markerscale=2)
    ax.set_title(f"HLWAS F158 — {args.n_pairs} selected pairs inside main-survey box")
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "sky_map.png", dpi=140)
    plt.close(fig)

    print(f"\nWrote:\n  {OUT_CAT / 'selected_pairs.ecsv'}")
    print(f"  {OUT_CAT / 'selected_pointings.ecsv'}")
    print(f"  {OUT_PLOTS / 'elongation_windows.png'}")
    print(f"  {OUT_PLOTS / 'sky_map.png'}")


if __name__ == "__main__":
    main()
