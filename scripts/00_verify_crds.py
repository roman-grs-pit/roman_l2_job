#!/usr/bin/env python3
"""Scan the CRDS reference cache for truncated / corrupted files.

Groups files by reference type (e.g. `roman_wfi_readnoise_*`), computes the
median size within each group, and flags any file whose size deviates more
than --tolerance from that median. Exits 0 if clean, 1 if any outliers found.

Motivation: when a stage-02 worker is OOM-killed mid-CRDS-download, the
partial reference file is left on disk. CRDS doesn't re-fetch files it can
see, so every subsequent sim that needs that reference crashes with an
opaque `TypeError: buffer is too small for requested array` inside
`asdf/tags/core/ndarray.py`. The debugging recipe (compare file sizes
within each ref-type class, delete the outlier) is cheap to codify.

Usage:
    pixi run python scripts/00_verify_crds.py
    pixi run python scripts/00_verify_crds.py --tolerance 0.05
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median


TYPE_RE = re.compile(r"^(roman_wfi_[a-z]+)_\d+\.asdf$")


def scan(ref_dir, tolerance):
    groups = defaultdict(list)
    for p in ref_dir.glob("*.asdf"):
        m = TYPE_RE.match(p.name)
        if not m:
            continue
        groups[m.group(1)].append(p)

    outliers = []
    for ref_type, paths in sorted(groups.items()):
        if len(paths) < 2:
            continue  # can't form a meaningful median with one file
        sizes = [p.stat().st_size for p in paths]
        med = median(sizes)
        if med == 0:
            continue
        for p, sz in zip(paths, sizes):
            dev = abs(sz - med) / med
            if dev > tolerance:
                outliers.append((ref_type, p, sz, med, dev))
    return outliers, groups


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", default="crds_cache/references/roman/wfi",
                    type=Path,
                    help="CRDS reference directory (default: "
                         "crds_cache/references/roman/wfi)")
    ap.add_argument("--tolerance", type=float, default=0.10,
                    help="flag files >this fraction off their ref-type "
                         "group median (default 0.10)")
    args = ap.parse_args()

    if not args.ref_dir.is_dir():
        print(f"{args.ref_dir} not found (CRDS cache empty — nothing to verify)")
        sys.exit(0)

    outliers, groups = scan(args.ref_dir, args.tolerance)
    n_files = sum(len(paths) for paths in groups.values())

    if not outliers:
        print(f"OK: {n_files} files across {len(groups)} ref types, all "
              f"within {args.tolerance:.0%} of their group median.")
        sys.exit(0)

    print(f"FAIL: {len(outliers)} suspicious file(s) in {args.ref_dir}:")
    for _, p, sz, med, dev in outliers:
        print(f"  {p.name}  ({sz/1e6:.1f} MB, {dev*100:.1f}% off group "
              f"median of {med/1e6:.1f} MB)")
    print()
    print("These were likely left by an interrupted download (OOM, network drop).")
    print("CRDS will not re-fetch files that exist on disk. To recover:")
    print()
    print("  rm <the files listed above>")
    print("  pixi run bash scripts/00_hydrate_crds.sh   # or re-run the failing stage")
    sys.exit(1)


if __name__ == "__main__":
    main()
