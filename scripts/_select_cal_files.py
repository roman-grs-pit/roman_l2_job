#!/usr/bin/env python3
"""Print L2 _cal.asdf paths matching the pointings file.

Filename convention from make_stack:
  r{program}{plan:02d}{passno:03d}{segment:03d}{observation:03d}{visit:03d}_
  {exposure:04d}_wfi{sca:02d}_{bandpass.lower()}_cal.asdf
"""
import argparse
import sys
from pathlib import Path

from astropy.table import Table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pointings")
    ap.add_argument("--cal-dir", default="output/cal")
    ap.add_argument("--program", default="00001")
    ap.add_argument("--require-exists", action="store_true",
                    help="Only print files that actually exist on disk.")
    args = ap.parse_args()

    cal = Path(args.cal_dir)
    t = Table.read(args.pointings)
    out = []
    missing = []
    for row in t:
        prefix = (
            f"r{args.program}{row['PLAN']:02d}{row['PASS']:03d}"
            f"{row['SEGMENT']:03d}{row['OBSERVATION']:03d}{row['VISIT']:03d}"
            f"_{row['EXPOSURE']:04d}_wfi"
        )
        suffix = f"_{row['BANDPASS'].lower()}_cal.asdf"
        for sca in range(1, 19):
            fname = cal / f"{prefix}{sca:02d}{suffix}"
            if args.require_exists and not fname.exists():
                missing.append(str(fname))
                continue
            out.append(str(fname))

    for p in out:
        print(p)
    if missing:
        print(f"# missing: {len(missing)}", file=sys.stderr)


if __name__ == "__main__":
    main()
