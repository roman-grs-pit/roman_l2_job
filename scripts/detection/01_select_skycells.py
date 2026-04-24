#!/usr/bin/env python
"""Phase 1 (skycell-driven): pick N target skycells in the HLWAS F158
footprint, stratified by ecliptic latitude (zodi proxy) and local
HLWAS coverage density (depth proxy).

Replaces the pair-driven Phase 1. The pair concept was vestigial once
Phase 2 learned to expand to the full HLWAS pointing list: all that's
really needed is a list of sky positions + a chosen sim date for each.

Sampling strategy
-----------------
1. **Candidate pool**: take every skycell whose centre is within
   `HLWAS_EXPAND_RADIUS_DEG` of at least one HLWAS F158 pointing
   centre (so the skycell has at least one nearby exposure and can
   plausibly be imaged by the survey). Gives tens of thousands.
2. **Metrics**:
   - `ecl_lat_deg` via astropy (zodi proxy — higher |lat| = lower zodi)
   - `count_0p25` = # HLWAS F158 exposures within 0.25° of the skycell
     centre (depth proxy — inner SCAs reach this skycell)
   - `count_0p6` = same at 0.6° (upper bound on SCA reach)
3. **Stratification**:
   - ecl_lat bins: |lat| < 15° / 15-30° / > 30°
   - coverage bins: bottom 25th pct of `count_0p25` (low depth) vs
     top 25th pct (high depth), computed from the candidate pool.
   - 2 per cell × 6 cells = 12 skycells. Picks are spread within
     each cell by greedy max-min distance in (ecl_lat, count_0p25).

Output
------
catalogs/detection/selected_skycells.ecsv
    One row per selected skycell. Columns include
    PAIR_ID (integer tag 1..N, retained for downstream script
    compatibility), skycell_idx, skycell_name, skycell_ra, skycell_dec,
    ECL_LAT_DEG, ECL_LON_DEG, SIM_DATE, SOLAR_ELONG_DEG,
    count_0p25, count_0p6, zodi_bin, coverage_bin, SKY_E_PER_PIX_PER_S.

output/detection/phase1/*.png
    Sky map, selection-matrix scatter, solar-elongation curves.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord, get_sun
from astropy.coordinates.errors import NonRotationTransformationWarning
from astropy.table import Table
from astropy.time import Time
import astropy.units as u
import galsim
import galsim.roman as roman
import romanisim.bandpass
from romancal.skycell.skymap import SKYMAP

warnings.filterwarnings("ignore", category=NonRotationTransformationWarning)

REPO = Path(__file__).resolve().parents[2]
HLWAS = REPO / "catalogs/HLWAS.sim.ecsv"
OUT_CAT = REPO / "catalogs/detection"
OUT_PLOTS = REPO / "output/detection/phase1"

# Coverage neighbour-search radii
COVERAGE_RADIUS_TIGHT = 0.25  # deg — inner-SCA reach proxy
COVERAGE_RADIUS_WIDE = 0.6    # deg — outer-SCA reach proxy
N_PER_CELL = 2
ELONG_MIN = 90.0
ELONG_MAX = 115.0
YEAR = 2026

ECL_LAT_BINS = [(0, 10), (15, 30), (40, 90)]  # |ecl_lat| bands with 5-10° gaps
ALL_PICK_MIN_SEP_DEG = 5.0  # great-circle separation between any 2 picks
COVERAGE_LOW_PCTILE = 25
COVERAGE_HIGH_PCTILE = 75
BANDPASS = "F158"
WFI_PIXEL_SCALE_ARCSEC = 0.11


def hlwas_f158() -> pd.DataFrame:
    """F158 exposures in the main HLWAS Wide-Field2 survey only.

    The HLWAS.sim.ecsv table also contains deep-field targets
    (COSMOS, XMM-LSS) whose coverage density is ~10× typical HLWAS.
    Including them in the candidate pool can leak deep-field skycells
    into the "high coverage" tier, which isn't what the study wants.
    """
    t = Table.read(HLWAS, format="ascii.ecsv")
    df = t[t["BANDPASS"] == "F158"].to_pandas()
    return df[df["TARGET_NAME"] == "Wide-Field2"].copy()


def build_coverage_tree(f158: pd.DataFrame):
    """3D unit-vector cKDTree on HLWAS F158 exposure positions."""
    from scipy.spatial import cKDTree
    sc = SkyCoord(ra=f158["RA"].values * u.deg, dec=f158["DEC"].values * u.deg)
    xyz = np.column_stack([sc.cartesian.x.value,
                            sc.cartesian.y.value,
                            sc.cartesian.z.value])
    return cKDTree(xyz)


def query_counts(tree, ra_deg: np.ndarray, dec_deg: np.ndarray,
                 radius_deg: float) -> np.ndarray:
    """# HLWAS F158 exposures within `radius_deg` of each (ra, dec)."""
    sc = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    xyz = np.column_stack([sc.cartesian.x.value,
                            sc.cartesian.y.value,
                            sc.cartesian.z.value])
    r_rad = np.deg2rad(radius_deg)
    chord = 2 * np.sin(r_rad / 2)
    return np.array(tree.query_ball_point(xyz, r=chord, return_length=True))


def candidate_skycells(f158: pd.DataFrame,
                        radius_deg: float = COVERAGE_RADIUS_WIDE) -> pd.DataFrame:
    """Return one row per skycell whose centre is within `radius_deg` of
    any HLWAS F158 pointing (the HLWAS-imageable candidate pool).
    """
    print(f"Loading skymap centres ({len(SKYMAP.model.skycells):,} skycells)…")
    centres = np.column_stack(
        [SKYMAP.model.skycells["ra_center"],
         SKYMAP.model.skycells["dec_center"]]
    )
    # RA/Dec bounding box of HLWAS F158 to pre-filter
    # (RA wraps through 0, so we filter on |Δra|<some)
    hlwas_dec_min, hlwas_dec_max = f158["DEC"].min() - 1.0, f158["DEC"].max() + 1.0
    dec_mask = (centres[:, 1] >= hlwas_dec_min) & (centres[:, 1] <= hlwas_dec_max)
    # RA: HLWAS sits in [~330, 360] ∪ [0, 80] plus small islands. Use all
    # RA inside the Dec band — RA reject happens in the tree query.
    candidates_idx = np.where(dec_mask)[0]
    print(f"  {len(candidates_idx):,} skycells in HLWAS Dec range")

    cand_ra = centres[candidates_idx, 0]
    cand_dec = centres[candidates_idx, 1]
    tree = build_coverage_tree(f158)
    # Interior = ≥ some_exposures within 0.6°. A single nearby exposure
    # could be a stray isolated visit at the edge; require ≥ 5 to mark
    # the skycell as inside a well-sampled patch of HLWAS.
    MIN_WIDE_COUNT = 5
    print(f"Filtering to HLWAS-interior (≥ {MIN_WIDE_COUNT} exposures within {radius_deg}°)…")
    counts_wide = query_counts(tree, cand_ra, cand_dec, radius_deg)
    interior = counts_wide >= MIN_WIDE_COUNT
    print(f"  {interior.sum():,} candidate skycells in HLWAS interior")

    # Compute tight-radius count for every interior candidate
    idx = candidates_idx[interior]
    ra = cand_ra[interior]; dec = cand_dec[interior]
    counts_tight = query_counts(tree, ra, dec, COVERAGE_RADIUS_TIGHT)
    counts_wide_i = counts_wide[interior]
    names = np.array(SKYMAP.model.skycells["name"])[idx]

    df = pd.DataFrame({
        "skycell_idx": idx,
        "skycell_name": names,
        "skycell_ra": ra,
        "skycell_dec": dec,
        f"count_0p{int(COVERAGE_RADIUS_TIGHT*100):02d}": counts_tight,
        f"count_0p{int(COVERAGE_RADIUS_WIDE*10):01d}": counts_wide_i,
    })

    # Ecliptic latitude
    sc = SkyCoord(ra=df["skycell_ra"].values * u.deg,
                   dec=df["skycell_dec"].values * u.deg)
    ecl = sc.transform_to("barycentrictrueecliptic")
    df["ECL_LAT_DEG"] = ecl.lat.deg
    df["ECL_LON_DEG"] = ecl.lon.deg
    return df


def stratify_and_pick(df: pd.DataFrame, n_per_cell: int = N_PER_CELL,
                      rng_seed: int = 2026) -> pd.DataFrame:
    """Apply the 3 (ecl_lat) × 2 (coverage) × N matrix."""
    tight = f"count_0p{int(COVERAGE_RADIUS_TIGHT*100):02d}"
    lo = np.percentile(df[tight], COVERAGE_LOW_PCTILE)
    hi = np.percentile(df[tight], COVERAGE_HIGH_PCTILE)
    print(f"\nCoverage percentiles (all candidates, n={len(df):,}):")
    print(f"  {COVERAGE_LOW_PCTILE}th pct = {lo:.1f} exposures within 0.25°")
    print(f"  {COVERAGE_HIGH_PCTILE}th pct = {hi:.1f}")

    df["coverage_bin"] = np.where(df[tight] <= lo, "low",
                                    np.where(df[tight] >= hi, "high", "mid"))
    abs_lat = np.abs(df["ECL_LAT_DEG"])
    # Use gap-separated bands so cross-bin picks don't land at adjacent
    # |ecl_lat| values (e.g., 15.1° and 14.9°).
    zodi_bin = pd.Series(["none"] * len(df), index=df.index)
    zodi_bin[abs_lat <= ECL_LAT_BINS[0][1]] = "high_zodi"
    zodi_bin[(abs_lat >= ECL_LAT_BINS[1][0])
              & (abs_lat <= ECL_LAT_BINS[1][1])] = "mid_zodi"
    zodi_bin[abs_lat >= ECL_LAT_BINS[2][0]] = "low_zodi"
    df["zodi_bin"] = zodi_bin

    rng = np.random.default_rng(rng_seed)
    picks = []
    pair_id = 1
    all_chosen_radec = []  # (ra, dec) of every pick made so far
    for zb in ["low_zodi", "mid_zodi", "high_zodi"]:
        for cb in ["low", "high"]:
            sub = df[(df["zodi_bin"] == zb) & (df["coverage_bin"] == cb)]
            # Exclude candidates within ALL_PICK_MIN_SEP_DEG of any earlier pick
            if all_chosen_radec:
                arr = np.array(all_chosen_radec)
                dra = np.abs(sub["skycell_ra"].values[:, None] - arr[:, 0][None, :])
                dra = np.minimum(dra, 360 - dra)
                ddec = np.abs(sub["skycell_dec"].values[:, None] - arr[:, 1][None, :])
                gc = np.sqrt((dra * np.cos(np.deg2rad(arr[:, 1][None, :])))**2
                             + ddec**2).min(axis=1)
                sub = sub[gc >= ALL_PICK_MIN_SEP_DEG]
            if len(sub) == 0:
                print(f"WARNING: no candidates in ({zb}, {cb})")
                continue
            # Greedy pick n_per_cell spread in (ecl_lat, count_0p25).
            # Start with the candidate closest to the cell's median on
            # both axes; subsequent picks maximise the minimum distance
            # (in normalized (ecl_lat, count) space) from picks so far.
            lat_med = sub["ECL_LAT_DEG"].median()
            cov_med = sub[tight].median()
            sub = sub.copy()
            sub["_lat_n"] = ((sub["ECL_LAT_DEG"] - lat_med)
                              / (sub["ECL_LAT_DEG"].std() or 1))
            sub["_cov_n"] = ((sub[tight] - cov_med)
                              / (sub[tight].std() or 1))
            chosen = []
            dist2 = np.sqrt(sub["_lat_n"]**2 + sub["_cov_n"]**2).values
            # Also enforce a minimum great-circle separation so the
            # two picks in a cell don't land next to each other in HLWAS.
            while len(chosen) < n_per_cell and len(chosen) < len(sub):
                if not chosen:
                    # First: median-closest
                    i = int(np.argmin(dist2))
                else:
                    chosen_coords = np.array([[sub.iloc[j]["_lat_n"],
                                                 sub.iloc[j]["_cov_n"]]
                                                for j in chosen])
                    pts = np.column_stack([sub["_lat_n"].values,
                                             sub["_cov_n"].values])
                    d_to_chosen = np.min(
                        np.linalg.norm(pts[:, None, :]
                                         - chosen_coords[None, :, :], axis=-1),
                        axis=1,
                    )
                    # Enforce 1° great-circle minimum separation
                    chosen_radec = np.array([[sub.iloc[j]["skycell_ra"],
                                                sub.iloc[j]["skycell_dec"]]
                                                for j in chosen])
                    dra = np.abs(sub["skycell_ra"].values[:, None]
                                  - chosen_radec[:, 0][None, :])
                    dra = np.minimum(dra, 360 - dra)
                    ddec = np.abs(sub["skycell_dec"].values[:, None]
                                    - chosen_radec[:, 1][None, :])
                    gc = np.sqrt(
                        (dra * np.cos(np.deg2rad(chosen_radec[:, 1][None, :])))**2
                        + ddec**2
                    ).min(axis=1)
                    d_to_chosen[gc < 1.0] = -1
                    d_to_chosen[chosen] = -1
                    if (d_to_chosen > 0).sum() == 0:
                        break
                    i = int(np.argmax(d_to_chosen))
                chosen.append(i)
            for j in chosen:
                r = sub.iloc[j].copy()
                r["PAIR_ID"] = pair_id; pair_id += 1
                r = r.drop(labels=["_lat_n", "_cov_n"])
                picks.append(r)
                all_chosen_radec.append((r["skycell_ra"], r["skycell_dec"]))

    picks_df = pd.DataFrame(picks).reset_index(drop=True)
    print(f"\nSelected {len(picks_df)} skycells:")
    print(picks_df[["PAIR_ID", "skycell_name", "skycell_ra", "skycell_dec",
                      "ECL_LAT_DEG", tight, "zodi_bin",
                      "coverage_bin"]].to_string(index=False))
    return picks_df


def solar_elongation_curve(ra_deg: float, dec_deg: float,
                             year: int = YEAR) -> pd.DataFrame:
    start = Time(f"{year}-01-01T00:00:00", scale="utc")
    days = np.arange(0, 366)
    times = start + days * u.day
    sun = get_sun(times)
    tgt = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    elong = sun.separation(tgt).deg
    return pd.DataFrame({"doy": days, "time": times.isot, "elong_deg": elong})


def choose_date(curve: pd.DataFrame) -> tuple[str, float]:
    ok = curve[(curve["elong_deg"] >= ELONG_MIN)
                & (curve["elong_deg"] <= ELONG_MAX)]
    if ok.empty:
        return None, float("nan")
    best = ok.iloc[(ok["elong_deg"] - ELONG_MIN).abs().argmin()]
    return best["time"], float(best["elong_deg"])


def sky_level_per_pix_e_per_s(ra_deg: float, dec_deg: float,
                                date_iso: str) -> float:
    bp_name = romanisim.bandpass.roman2galsim_bandpass[BANDPASS]
    bp = roman.getBandpasses(AB_zeropoint=True)[bp_name]
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)
    world = galsim.CelestialCoord(c.ra.rad * galsim.radians,
                                    c.dec.rad * galsim.radians)
    sky = roman.getSkyLevel(bp, world_pos=world,
                              date=Time(date_iso).datetime, exptime=1)
    sky *= (1.0 + roman.stray_light_fraction)
    sky_per_pix = sky * WFI_PIXEL_SCALE_ARCSEC ** 2
    sky_per_pix += roman.thermal_backgrounds[bp_name]
    return float(sky_per_pix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cell", type=int, default=N_PER_CELL)
    ap.add_argument("--year", type=int, default=YEAR)
    args = ap.parse_args()

    OUT_CAT.mkdir(parents=True, exist_ok=True)
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    f158 = hlwas_f158()
    print(f"HLWAS F158 exposures: {len(f158):,}")

    df = candidate_skycells(f158)
    picks = stratify_and_pick(df, n_per_cell=args.n_per_cell)

    # Per-skycell date + sky level
    dates = []; elongs = []; skies = []
    for _, r in picks.iterrows():
        curve = solar_elongation_curve(r["skycell_ra"], r["skycell_dec"],
                                         year=args.year)
        d, e = choose_date(curve)
        sky = sky_level_per_pix_e_per_s(r["skycell_ra"], r["skycell_dec"], d) if d else float("nan")
        dates.append(d); elongs.append(e); skies.append(sky)
    picks["SIM_DATE"] = dates
    picks["SOLAR_ELONG_DEG"] = elongs
    picks["SKY_E_PER_PIX_PER_S"] = skies
    # Rename for consistency with the pair_id schema used by 04a
    picks["n_pointings"] = 0  # unused now but the phase2 expansion step needs column
    picks["n_sca_events"] = 0

    out = OUT_CAT / "selected_skycells.ecsv"
    Table.from_pandas(picks).write(out, format="ascii.ecsv", overwrite=True)
    print(f"\nWrote {out.relative_to(REPO)}")

    # Plots
    import matplotlib.pyplot as plt
    tight = f"count_0p{int(COVERAGE_RADIUS_TIGHT*100):02d}"

    # 1) Coverage stats across the HLWAS-interior
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[tight], bins=40, color="tab:blue", alpha=0.7,
            label=f"{len(df):,} HLWAS-interior candidates")
    lo = np.percentile(df[tight], COVERAGE_LOW_PCTILE)
    hi = np.percentile(df[tight], COVERAGE_HIGH_PCTILE)
    ax.axvline(lo, color="tab:red", ls="--",
                 label=f"{COVERAGE_LOW_PCTILE}th pct ({lo:.0f})")
    ax.axvline(hi, color="tab:green", ls="--",
                 label=f"{COVERAGE_HIGH_PCTILE}th pct ({hi:.0f})")
    for _, r in picks.iterrows():
        ax.axvline(r[tight], color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel(f"# HLWAS F158 exposures within {COVERAGE_RADIUS_TIGHT}° of skycell")
    ax.set_ylabel("# skycells")
    ax.set_title("HLWAS-interior coverage distribution; selected skycells marked")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "coverage_distribution.png", dpi=140)
    plt.close(fig)

    # 2) Selection matrix — scatter in (ecl_lat, coverage)
    fig, ax = plt.subplots(figsize=(10, 6))
    sample = df.sample(min(len(df), 20000), random_state=0)
    ax.scatter(sample["ECL_LAT_DEG"], sample[tight], s=1, color="lightgray",
                 alpha=0.3, label="HLWAS-interior candidates")
    cmap = plt.get_cmap("tab10")
    for i, (_, r) in enumerate(picks.iterrows()):
        ax.scatter(r["ECL_LAT_DEG"], r[tight], s=120, marker="*",
                     color=cmap(i % 10), edgecolor="k", zorder=5)
        ax.annotate(f" {int(r['PAIR_ID'])}",
                     (r["ECL_LAT_DEG"], r[tight]), fontsize=8)
    ax.set_xlabel("ecliptic latitude (deg)")
    ax.set_ylabel(f"# exposures within {COVERAGE_RADIUS_TIGHT}°")
    ax.axhline(lo, color="tab:red", ls="--", alpha=0.3)
    ax.axhline(hi, color="tab:green", ls="--", alpha=0.3)
    for b in [-30, -15, 15, 30]:
        ax.axvline(b, color="k", ls=":", alpha=0.3, lw=0.5)
    ax.grid(alpha=0.3)
    ax.set_title("Stratification matrix — selected skycells (stars) in (zodi × coverage) space")
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "selection_matrix.png", dpi=140)
    plt.close(fig)

    # 3) Sky map
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(f158["RA"], f158["DEC"], s=0.3, color="lightgray", alpha=0.3,
                 label=f"{len(f158):,} HLWAS F158 exposures")
    for i, (_, r) in enumerate(picks.iterrows()):
        ax.scatter(r["skycell_ra"], r["skycell_dec"], s=140, marker="*",
                     color=cmap(i % 10), edgecolor="k", zorder=5)
        ax.annotate(f" {int(r['PAIR_ID'])}",
                     (r["skycell_ra"], r["skycell_dec"]), fontsize=9)
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_xlim(360, 0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_PLOTS / "sky_map.png", dpi=140)
    plt.close(fig)

    print(f"Wrote {OUT_PLOTS / 'coverage_distribution.png'}")
    print(f"Wrote {OUT_PLOTS / 'selection_matrix.png'}")
    print(f"Wrote {OUT_PLOTS / 'sky_map.png'}")


if __name__ == "__main__":
    main()
