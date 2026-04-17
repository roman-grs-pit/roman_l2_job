#!/usr/bin/env python3
"""Rewrite sims_raw.script lines:
- prepend cal-dir to the output filename
- replace --rng_seed N with a deterministic per-line seed (CRC32 of basename)
- wrap with [ -f <out> ] || ... > log 2>&1
"""
import argparse
import re
import zlib
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cal-dir", default="output/cal")
    ap.add_argument("--log-dir", default="output/logs")
    args = ap.parse_args()

    rx_seed = re.compile(r"--rng_seed\s+\d+")
    out_lines = []

    with open(args.input) as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            tokens = raw.split()
            # romanisim-make-image <FILENAME> ...
            assert tokens[0] == "romanisim-make-image", tokens[0]
            base = tokens[1]
            new_path = f"{args.cal_dir}/{base}"
            tokens[1] = new_path
            line = " ".join(tokens)
            seed = zlib.crc32(base.encode()) & 0x7FFFFFFF
            line = rx_seed.sub(f"--rng_seed {seed}", line)
            log = f"{args.log_dir}/{base}.log"
            wrapped = f"[ -f {new_path} ] || {{ {line} > {log} 2>&1; }}"
            out_lines.append(wrapped)

    Path(args.output).write_text("\n".join(out_lines) + "\n")
    print(f"Wrote {args.output} ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
