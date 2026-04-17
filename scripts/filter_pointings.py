#!/usr/bin/env python3
"""Filter HLWAS.sim.ecsv down to a pointings file for one sub-region.

Reads pointings-filter parameters from the YAML config (region cone or
box, bandpass, optional pass/segment/visit restrictions), applies them
to catalogs/HLWAS.sim.ecsv, and writes `pointings_<tag>.ecsv`.

Selection logic:

1. Keep rows whose BANDPASS matches `pointings.bandpass`.
2. Identify visits (PLAN+PASS+SEGMENT+OBSERVATION+VISIT) in which at
   least one exposure falls inside the region:
       cone: separation from (ra, dec) < radius_deg
       box:  ra in [ra_min, ra_max]  AND  dec in [dec_min, dec_max]
3. Emit every exposure of every matching visit (all dithers, not only
   the one that triggered the match).
4. If any of `only_pass/segment/visit` are non-null, restrict further.

Usage:
    pixi run python scripts/filter_pointings.py configs/smoke.yaml
    # writes pointings_smoke.ecsv
"""
import argparse
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u

from _config import BoxRegion, ConeRegion, load_config


VISIT_KEYS = ("PLAN", "PASS", "SEGMENT", "OBSERVATION", "VISIT")


def _region_mask(candidates, region):
    if isinstance(region, ConeRegion):
        center = SkyCoord(ra=region.ra * u.deg, dec=region.dec * u.deg)
        sep = SkyCoord(ra=candidates["RA"] * u.deg,
                       dec=candidates["DEC"] * u.deg).separation(center)
        return sep < region.radius_deg * u.deg
    if isinstance(region, BoxRegion):
        ra = candidates["RA"]
        dec = candidates["DEC"]
        return ((ra >= region.ra_min) & (ra <= region.ra_max)
                & (dec >= region.dec_min) & (dec <= region.dec_max))
    raise TypeError(f"unsupported region: {type(region).__name__}")


def filter_pointings(tbl, region, bandpass):
    band_mask = tbl["BANDPASS"] == bandpass
    if not band_mask.any():
        raise SystemExit(f"no rows with BANDPASS={bandpass!r}")
    candidates = tbl[band_mask]

    within = candidates[_region_mask(candidates, region)]
    if len(within) == 0:
        raise SystemExit(
            f"no {bandpass} exposures inside the configured region "
            f"({type(region).__name__})")

    matching_visits = {tuple(int(row[k]) for k in VISIT_KEYS) for row in within}
    keep = [tuple(int(row[k]) for k in VISIT_KEYS) in matching_visits
            for row in candidates]
    return candidates[keep]


def restrict_visits(tbl, only_pass, only_segment, only_visit):
    mask = [True] * len(tbl)
    for col, val in (("PASS", only_pass), ("SEGMENT", only_segment),
                     ("VISIT", only_visit)):
        if val is not None:
            mask = [m and (row[col] == val) for m, row in zip(mask, tbl)]
    return tbl[mask]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to YAML config file")
    ap.add_argument("-i", "--input", default="catalogs/HLWAS.sim.ecsv", type=Path,
                    help="HLWAS.sim.ecsv (default: catalogs/HLWAS.sim.ecsv)")
    ap.add_argument("-o", "--output", default=None, type=Path,
                    help="output ECSV (default: pointings_<tag>.ecsv)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_path = args.output or Path(f"pointings_{cfg.tag}.ecsv")

    tbl = Table.read(args.input, format="ascii.ecsv")
    out = filter_pointings(tbl, cfg.pointings.region, cfg.pointings.bandpass)
    out = restrict_visits(out,
                          cfg.pointings.only_pass,
                          cfg.pointings.only_segment,
                          cfg.pointings.only_visit)
    if len(out) == 0:
        raise SystemExit("filter produced 0 rows")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.write(out_path, format="ascii.ecsv", overwrite=True)
    print(f"wrote {len(out)} rows -> {out_path}")


if __name__ == "__main__":
    main()
