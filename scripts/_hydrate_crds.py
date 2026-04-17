#!/usr/bin/env python3
"""Pre-hydrate the CRDS reference cache by calling crds.getreferences()
once per (SCA, bandpass) pair, for the reference types romanisim fetches
when invoked with --usecrds.

This replaces an earlier "run 18 throwaway sims" approach that was
correct but spent ~80 min on simulation work we don't need; CRDS cache
population is a pure metadata + download operation and finishes in ~5–15 s
against a warm cache, or in however long it takes to download the refs
otherwise.

The list of reference types mirrors `romanisim.parameters.reference_data`
(all keys are set to None when `--usecrds` is passed, forcing a CRDS
fetch). If romanisim grows or shrinks that list, update REFTYPES below.
"""
import argparse
import sys
import time

import crds


# From romanisim.parameters.reference_data: keys romanisim hands to
# crds.getreferences() when the user passes --usecrds.
REFTYPES = [
    "dark",
    "darkdecaysignal",
    "distortion",
    "flat",
    "gain",
    "inverselinearity",
    "linearity",
    "integralnonlinearity",
    "readnoise",
    "saturation",
]


def crds_params(sca, bandpass, date="2026-01-01T00:00:00.000"):
    """Minimal Roman WFI header dict sufficient for CRDS ref-type lookup."""
    return {
        "roman.meta.instrument.name": "WFI",
        "roman.meta.instrument.detector": f"WFI{sca:02d}",
        "roman.meta.instrument.optical_element": bandpass,
        "roman.meta.exposure.type": "WFI_IMAGE",
        "roman.meta.exposure.ma_table_number": 1007,
        "roman.meta.exposure.start_time": date,
        "roman.meta.observation.start_time": date,
    }


def hydrate(bandpass, scas, reftypes):
    total_start = time.time()
    seen = set()
    print(f"Hydrating CRDS refs: bandpass={bandpass}, {len(scas)} SCAs, "
          f"{len(reftypes)} reftypes")
    print(f"{'SCA':<5} {'seconds':>9}  {'new refs':>9}  status")
    print("-" * 40)
    for sca in scas:
        start = time.time()
        try:
            result = crds.getreferences(
                crds_params(sca, bandpass),
                reftypes=reftypes,
                observatory="roman",
            )
        except Exception as e:
            print(f"{sca:<5} {'-':>9}  {'-':>9}  FAILED: {e}")
            return 1
        fetched = {v.split("/")[-1] for v in result.values()}
        new = fetched - seen
        seen |= fetched
        print(f"{sca:<5} {time.time() - start:>9.1f}  {len(new):>9}  ok")
    print("-" * 40)
    total = time.time() - total_start
    print(f"Total: {total:.1f}s for {len(scas)} SCAs × {len(reftypes)} reftypes, "
          f"{len(seen)} unique ref files cached")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bandpass", default="F158")
    ap.add_argument("--scas", nargs="+", type=int, default=list(range(1, 19)),
                    help="SCA indices (default 1..18)")
    args = ap.parse_args()
    sys.exit(hydrate(args.bandpass, args.scas, REFTYPES))


if __name__ == "__main__":
    main()
