#!/usr/bin/env python3
"""Convert input metadata.parquet (F158 in mag) to catalogs/sources.parquet (F158 in maggies).

Keeps stars and galaxies. No spatial filter applied. Writes parquet so
romanisim's Table.read picks it up directly.
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/metadata.parquet",
                    help="Input parquet path (F158 column in AB magnitudes).")
    ap.add_argument("--output", default="catalogs/sources.parquet",
                    help="Output parquet path (F158 column in maggies).")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Reading {in_path} ...")
    t = Table.read(in_path)
    print(f"  rows: {len(t):,}")
    print(f"  F158 range (mag): {t['F158'].min():.2f} .. {t['F158'].max():.2f}")

    # mag → maggies (AB): flux = 10**(-mag / 2.5)
    t["F158"] = np.power(10.0, -np.asarray(t["F158"], dtype=np.float64) / 2.5)
    print(f"  F158 range (maggies): {t['F158'].min():.3e} .. {t['F158'].max():.3e}")

    t.write(out_path, overwrite=True)
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
