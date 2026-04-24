#!/usr/bin/env python
"""Phase 4a: build the combined catalog + sims.script for one skycell.

For a given SKYCELL_ID, this script

1. Reads the Phase-3 stars and galaxies catalogs for the selected
   skycell and concatenates them into a single combined parquet
   (romanisim accepts mixed `type="PSF"` / `type="SER"` rows in one
   file — the plan's "combined input" choice).
2. Reads `pointings_sca_to_simulate.ecsv` (Phase 2) and
   `selected_pointings.ecsv` (Phase 1) to enumerate the (pointing, SCA)
   pairs for the given SKYCELL_ID. Emits one `romanisim-make-image` line
   per L2 file, each guarded with a skip-if-exists check, and writes
   the lines to `output/detection/sims_<cell>.script`.

The per-line seed is `zlib.crc32(basename)` — same convention as the
full run (`scripts/_postprocess_sims.py`), so every (visit, exposure,
SCA) draws an independent noise realisation.

Filenames follow the canonical HLWAS convention:
    r{program:05d}{plan:02d}{pass:03d}{segment:03d}{observation:03d}{visit:03d}
    _{exposure:04d}_wfi{sca:02d}_{bandpass}_cal.asdf

Usage:
    pixi run python scripts/detection/04a_build_sims.py --cell 1
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import pandas as pd
from astropy.table import Table

REPO = Path(__file__).resolve().parents[2]
CATS_IN = REPO / "catalogs/detection/catalogs"
OUT_CAT = REPO / "output/detection/catalogs"
L2_DIR = REPO / "output/detection/l2"
LOG_DIR = REPO / "output/detection/logs"
BANDPASS = "F158"
MA_TABLE = 1007
PROGRAM = 1       # hard-coded, as in the existing full run
PLAN = 1          # ditto


def canonical_l2_filename(row: pd.Series) -> str:
    """Return the `r.._cal.asdf` filename for a (pointing, SCA) row."""
    return (
        f"r{PROGRAM:05d}{PLAN:02d}{int(row['PASS']):03d}"
        f"{int(row['SEGMENT']):03d}{int(row['OBSERVATION']):03d}"
        f"{int(row['VISIT']):03d}_{int(row['EXPOSURE']):04d}"
        f"_wfi{int(row['SCA']):02d}_{BANDPASS.lower()}_cal.asdf"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", type=int, required=True,
                    help="SKYCELL_ID to build sims for (integer tag from "
                         "selected_skycells.ecsv)")
    args = ap.parse_args()

    OUT_CAT.mkdir(parents=True, exist_ok=True)
    L2_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --- catalogs merge ---
    pts_sim = Table.read(
        REPO / "catalogs/detection/pointings_sca_to_simulate.ecsv",
        format="ascii.ecsv"
    ).to_pandas()
    pts_sim = pts_sim[pts_sim["SKYCELL_ID"] == args.cell].copy()
    if pts_sim.empty:
        raise SystemExit(f"No rows in pointings_sca_to_simulate.ecsv for cell {args.cell}")
    skycell_name = str(pts_sim["skycell_name"].iloc[0])
    print(f"Skycell {args.cell} → {skycell_name}, "
          f"{len(pts_sim)} (pointing, SCA) rows")

    stars_path = CATS_IN / f"skycell_{skycell_name}_stars.parquet"
    gal_path = CATS_IN / f"skycell_{skycell_name}_galaxies.parquet"
    stars = Table.read(stars_path).to_pandas()
    gals = Table.read(gal_path).to_pandas()
    # Re-index the galaxies so (stars, galaxies) have disjoint src_index
    gals["src_index"] = gals["src_index"] + len(stars)
    combined = pd.concat([stars, gals], ignore_index=True)
    combined_path = OUT_CAT / f"skycell_{skycell_name}_combined.parquet"
    Table.from_pandas(combined).write(combined_path, overwrite=True)
    print(f"  wrote combined catalog: {combined_path.relative_to(REPO)} "
          f"({len(combined)} sources)")

    # --- emit sims.script ---
    # pointings_sca_to_simulate.ecsv carries RA/DEC/PA/SIM_DATE directly
    # (from Phase 2's HLWAS expansion), so no merge needed here.
    if pts_sim["RA"].isna().any():
        bad = pts_sim[pts_sim["RA"].isna()]
        raise SystemExit(f"Missing RA/Dec for {len(bad)} rows:\n{bad}")
    script_path = REPO / f"output/detection/sims_{args.cell}.script"
    lines = []
    for _, r in pts_sim.sort_values(
            ["PASS", "SEGMENT", "VISIT", "EXPOSURE", "SCA"]).iterrows():
        fname = canonical_l2_filename(r)
        l2 = L2_DIR / fname
        log = LOG_DIR / f"{fname}.log"
        seed = zlib.crc32(fname.encode()) & 0x7FFFFFFF
        date = r["SIM_DATE"]
        cmd = (
            f"[ -f {l2} ] || {{ romanisim-make-image {l2} "
            f"--date {date} --level 2 --rng_seed {seed} "
            f"--sca {int(r['SCA'])} --usecrds --psftype stpsf "
            f"--bandpass {BANDPASS} --radec {r['RA']:.6f} {r['DEC']:.6f} "
            f"--roll {r['PA']} --ma_table_number {MA_TABLE} "
            f"--catalog {combined_path.relative_to(REPO)} "
            f"--meta visit.nexposures=3 "
            f"> {log} 2>&1; }}"
        )
        lines.append(cmd)

    script_path.write_text("\n".join(lines) + "\n")
    print(f"  wrote sims script: {script_path.relative_to(REPO)} "
          f"({len(lines)} L2 sims)")
    print(f"  L2 output dir: {L2_DIR.relative_to(REPO)}")
    print(f"  log output dir: {LOG_DIR.relative_to(REPO)}")


if __name__ == "__main__":
    main()
