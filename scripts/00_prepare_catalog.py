#!/usr/bin/env python3
"""Prepare catalogs/sources.parquet from the config's input catalog.

Reads the catalog path, bandpass column name, and input units from the
YAML config passed on the command line. If `catalog.input_units == "mag"`,
converts that column from AB magnitudes to maggies (romanisim's expected
unit); if `== "maggies"`, passes through unchanged. The output is written
to catalogs/sources.parquet regardless, so downstream stages are unaware
of the upstream unit.

Usage:
    pixi run python scripts/00_prepare_catalog.py configs/smoke.yaml
"""
import argparse
from pathlib import Path

import numpy as np
from astropy.table import Table

from _config import load_config


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to YAML config file")
    ap.add_argument("--output", default="catalogs/sources.parquet", type=Path,
                    help="output parquet path (default: catalogs/sources.parquet)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    col = cfg.catalog.bandpass_col

    print(f"Reading {cfg.catalog.input} (input_units={cfg.catalog.input_units}) ...")
    t = Table.read(cfg.catalog.input)
    if col not in t.colnames:
        raise SystemExit(f"column {col!r} not found in {cfg.catalog.input} "
                         f"(have: {t.colnames})")
    print(f"  rows: {len(t):,}")

    values = np.asarray(t[col], dtype=np.float64)
    if cfg.catalog.input_units == "mag":
        print(f"  {col} range (mag): {values.min():.2f} .. {values.max():.2f}")
        t[col] = np.power(10.0, -values / 2.5)
        print(f"  {col} range (maggies): {t[col].min():.3e} .. {t[col].max():.3e}")
    else:  # "maggies"
        print(f"  {col} range (maggies): {values.min():.3e} .. {values.max():.3e}")
        print("  no unit conversion (input already in maggies)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    t.write(args.output, overwrite=True)
    print(f"Wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
