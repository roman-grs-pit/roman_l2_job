#!/usr/bin/env python3
"""Filter skycell asn JSONs by distinct-pointing count.

Each L2 cal filename matches:
  r{program:05d}{plan:02d}{passno:03d}{segment:03d}{observation:03d}{visit:03d}_
  {exposure:04d}_wfi{sca:02d}_{bandpass}_cal.asdf

For each asn JSON in --asn-dir, count distinct pointings -- the unique
(passno, segment, observation, visit, exposure) tuples among its members.
This counts pointings, NOT SCAs: 18 SCAs of one pointing yield n_pointings=1,
not 18.

Asn JSONs whose pointing count is below --min-pointings are flagged for
rejection. Default action is dry-run; with --apply, rejected JSONs are
moved to <asn-dir>/rejected/ (not deleted -- restore with mv).

Usage:
  pixi run python scripts/filter_asn_skycells.py \
      --asn-dir output/<tag>/asn --min-pointings 6
  pixi run python scripts/filter_asn_skycells.py \
      --asn-dir output/<tag>/asn --min-pointings 6 --apply
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# Strip _wfi<NN>_<bandpass>_cal.asdf to get the per-pointing prefix.
SCA_SUFFIX_RX = re.compile(r"_wfi\d{2}_[a-zA-Z0-9]+_cal\.asdf$")


def pointing_key(expname: str) -> str | None:
    """Return the per-pointing prefix from a cal filename, or None on mismatch."""
    base = Path(expname).name
    m = SCA_SUFFIX_RX.search(base)
    if not m:
        return None
    return base[: m.start()]


def count_pointings(asn_path: Path) -> tuple[int, int]:
    """Return (n_distinct_pointings, n_total_members) for an asn JSON."""
    with asn_path.open() as f:
        d = json.load(f)
    keys: set[str] = set()
    n_members = 0
    for prod in d.get("products", []):
        for mem in prod.get("members", []):
            n_members += 1
            k = pointing_key(mem.get("expname", ""))
            if k is not None:
                keys.add(k)
    return len(keys), n_members


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--asn-dir", required=True, type=Path,
                    help="Directory containing *_asn.json files (top-level only)")
    ap.add_argument("--min-pointings", type=int, default=6,
                    help="Reject asn JSONs with fewer distinct pointings than this (default: 6)")
    ap.add_argument("--apply", action="store_true",
                    help="Actually move rejected JSONs to <asn-dir>/rejected/. "
                         "Default is dry-run (just print summary).")
    ap.add_argument("--delete", action="store_true",
                    help="With --apply: delete rejected JSONs instead of moving them")
    args = ap.parse_args()

    asn_dir: Path = args.asn_dir
    if not asn_dir.is_dir():
        print(f"asn-dir not a directory: {asn_dir}", file=sys.stderr)
        return 1

    asns = sorted(p for p in asn_dir.glob("*_asn.json") if p.is_file())
    if not asns:
        print(f"no *_asn.json files under {asn_dir}", file=sys.stderr)
        return 1

    counts: dict[Path, tuple[int, int]] = {}
    for a in asns:
        try:
            counts[a] = count_pointings(a)
        except Exception as e:
            print(f"WARN: failed to parse {a.name}: {e}", file=sys.stderr)
            counts[a] = (0, 0)

    pointing_hist = Counter(c[0] for c in counts.values())
    rejected = [a for a, (n, _) in counts.items() if n < args.min_pointings]
    kept = [a for a in asns if a not in set(rejected)]

    print(f"asn-dir: {asn_dir}")
    print(f"total skycells: {len(asns)}")
    print(f"distribution (distinct-pointings/skycell -> count):")
    for n in sorted(pointing_hist):
        print(f"  {n:>3}: {pointing_hist[n]}")
    print(f"min-pointings threshold: {args.min_pointings}")
    print(f"kept: {len(kept)}    rejected: {len(rejected)}")

    show_count = 50
    show = rejected if len(rejected) <= show_count else rejected[:show_count // 2]
    for r in show:
        n, total = counts[r]
        print(f"  reject: {r.name}  ({n} pointings, {total} cal members)")
    if len(rejected) > show_count:
        print(f"  ... and {len(rejected) - len(show)} more")

    if not args.apply:
        print("(dry run -- re-run with --apply to actually move/delete)")
        return 0

    if args.delete:
        for r in rejected:
            r.unlink()
        print(f"applied: deleted {len(rejected)} skycell asn JSONs")
    else:
        rej_dir = asn_dir / "rejected"
        rej_dir.mkdir(exist_ok=True)
        for r in rejected:
            shutil.move(str(r), str(rej_dir / r.name))
        print(f"applied: moved {len(rejected)} skycell asn JSONs to {rej_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
