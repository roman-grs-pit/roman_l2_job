#!/usr/bin/env python3
"""Parallel drop-in for romancal's skycell_asn.

Upstream `skycell_asn` is single-threaded; the hot path
(`_create_intersecting_skycell_index`) is a pure-function loop over input
cal files (open + WCS-vs-skycell-tessellation match per file) and dominates
wall time at >1 second per file. Measured on cpun-2xlg cal files: ~10 s/file
serial, so 2700 files ≈ 7.5 hours. This script swaps that loop for a
ProcessPoolExecutor; the rest of the pipeline (group → emit JSONs) is
imported unchanged from upstream.

CLI is identical to upstream `skycell_asn` plus `--workers`, so this can
substitute one-for-one in `scripts/03_build_asn.sh`.

Why a vendored script instead of patching upstream: the parallelism is
trivial here, but romancal's logging configuration (NullHandler + propagate
to root) and import side effects make process-spawn semantics worth pinning
down per-tool rather than per-package.
"""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor

from roman_datamodels import datamodels as rdm

import romancal.skycell.match as sm
from romancal.associations.skycell_asn import (
    FileRecord,
    _create_groups,
    _process_groups,
)

logger = logging.getLogger(__name__)


def _worker(file_name: str) -> tuple[str, list[int], str]:
    """Pure-function per-file work. Runs in a subprocess.

    Returns a tuple suitable for FileRecord(*tup); FileRecord itself can't
    be returned directly because its constructor lives in the parent module
    and pickling across the process boundary is cleaner with plain types.
    """
    try:
        cal = rdm.open(file_name)
        filter_id = cal.meta.instrument.optical_element.lower()
        indices = [int(i) for i in sm.find_skycell_matches(cal.meta.wcs)]
        cal.close()
    except Exception as e:
        logger.warning("Unable to read %s: %s; defaulting to unknown", file_name, e)
        filter_id = "unknown"
        indices = []
    return (file_name, indices, filter_id)


def skycell_asn_parallel(
    filelist: list[str],
    output_file_root: str,
    product_type: str,
    data_release_id: str,
    workers: int,
) -> None:
    """Drop-in replacement for upstream skycell_asn() with a parallel phase 1."""
    product_type = (product_type or "full").lower()
    groups = _create_groups(filelist, product_type)
    print(f"phase 1: opening {len(filelist)} cal files with {workers} workers...")

    file_index: list[FileRecord] = []
    n_done = 0
    # ex.map preserves input order, which keeps the resulting `file_index`
    # deterministic (matters for asn JSON contents). Per-file cost is
    # roughly uniform so head-of-line blocking is not material here.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for tup in ex.map(_worker, filelist, chunksize=4):
            file_index.append(FileRecord(*tup))
            n_done += 1
            if n_done % 100 == 0 or n_done == len(filelist):
                print(f"  phase 1: {n_done}/{len(filelist)}")

    print(f"phase 2: building {product_type} associations for {len(groups)} groups...")
    _process_groups(groups, file_index, output_file_root, data_release_id, product_type)
    print("done.")


def _cli(args=None) -> int:
    parser = argparse.ArgumentParser(
        description="Parallel skycell_asn (drop-in for romancal's skycell_asn).",
    )
    parser.add_argument("-o", "--output-file-root", required=True)
    parser.add_argument("--product-type", default="full")
    parser.add_argument("--data-release-id", default="p")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("filelist", nargs="+")
    parsed = parser.parse_args(args=args)

    skycell_asn_parallel(
        parsed.filelist,
        parsed.output_file_root,
        parsed.product_type,
        parsed.data_release_id,
        parsed.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
