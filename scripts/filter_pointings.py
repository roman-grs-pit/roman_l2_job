#!/usr/bin/env python3
"""
Filter HLWAS.sim.ecsv down to a pointings file for one sub-region of the
Roman High-Latitude Wide-Area Survey footprint.

The selection is:
    1. Keep only rows matching --bandpass.
    2. Identify visits (PASS, SEGMENT, OBSERVATION, VISIT) in which at least one
       exposure falls within --radius degrees of (--ra, --dec).
    3. Emit every exposure of every matching visit (i.e. all dithers, not just
       the one that triggered the match).

This reproduces the selection used to build pointings_full.ecsv:
    filter_pointings.py --ra 10 --dec 0 --radius 0.5 --bandpass F158 \
        -i catalogs/HLWAS.sim.ecsv -o pointings_full.ecsv

To scale up, widen --radius or move (--ra, --dec), or swap --bandpass for
another filter. Add --pass/--segment/--visit to carve out a smoke subset.
"""
import argparse
from pathlib import Path

from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u


VISIT_KEYS = ("PLAN", "PASS", "SEGMENT", "OBSERVATION", "VISIT")


def filter_pointings(tbl, ra, dec, radius, bandpass):
    band_mask = tbl["BANDPASS"] == bandpass
    if not band_mask.any():
        raise SystemExit(f"no rows with BANDPASS={bandpass!r}")
    candidates = tbl[band_mask]

    center = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    sep = SkyCoord(ra=candidates["RA"] * u.deg,
                   dec=candidates["DEC"] * u.deg).separation(center)
    within = candidates[sep < radius * u.deg]
    if len(within) == 0:
        raise SystemExit(
            f"no {bandpass} exposures within {radius} deg of "
            f"(RA={ra}, Dec={dec})")

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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", required=True, type=Path,
                   help="HLWAS.sim.ecsv")
    p.add_argument("-o", "--output", required=True, type=Path,
                   help="output ECSV pointings file")
    p.add_argument("--ra", type=float, required=True, help="center RA (deg)")
    p.add_argument("--dec", type=float, required=True, help="center Dec (deg)")
    p.add_argument("--radius", type=float, default=0.5,
                   help="selection radius (deg), default 0.5")
    p.add_argument("--bandpass", default="F158",
                   help="BANDPASS to keep, default F158")
    p.add_argument("--pass", dest="only_pass", type=int, default=None,
                   help="restrict to a single PASS after visit expansion")
    p.add_argument("--segment", dest="only_segment", type=int, default=None,
                   help="restrict to a single SEGMENT after visit expansion")
    p.add_argument("--visit", dest="only_visit", type=int, default=None,
                   help="restrict to a single VISIT after visit expansion")
    args = p.parse_args()

    tbl = Table.read(args.input, format="ascii.ecsv")
    out = filter_pointings(tbl, args.ra, args.dec, args.radius, args.bandpass)
    out = restrict_visits(out, args.only_pass, args.only_segment,
                          args.only_visit)
    if len(out) == 0:
        raise SystemExit("filter produced 0 rows")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write(args.output, format="ascii.ecsv", overwrite=True)
    print(f"wrote {len(out)} rows -> {args.output}")


if __name__ == "__main__":
    main()
