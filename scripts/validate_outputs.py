#!/usr/bin/env python3
"""Stage-output validation for the pipeline.

Two rungs of checks per file:
  - Rung 1 (structural): roman_datamodels.open + validate. Catches truncated
    files, schema violations, missing required meta.
  - Rung 2 (array sanity): NaN-fraction, DQ-bad-fraction, data-array median
    sanity, error-array positivity. Thresholds are deliberately liberal on
    first pass — the JSON summary emits the observed distribution so we can
    tighten later.

Currently covers stage 02 (cal asdf) only. Stage 04 mosaic and stage 05
catalog/segm checks are TODO; design accommodates them via dispatch on
filename pattern + datamodel type.

Usage:
    pixi run python scripts/validate_outputs.py configs/<tag>.yaml
    pixi run python scripts/validate_outputs.py configs/<tag>.yaml --workers 16

Exit codes: 0 if all PASS, 2 if any FAIL, 1 on usage error.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# Liberal first-pass thresholds. Refine after seeing the distribution.
NAN_FRAC_MAX = 0.50      # data array; NaNs may flag bad pixels but >50% is suspicious
DQ_BAD_FRAC_MAX = 0.50   # dq != 0 fraction
EXPECTED_SHAPE = (4088, 4088)


@dataclass
class Result:
    path: str
    passed: bool
    reasons: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)


def _validate_cal(path: str) -> Result:
    """Validate one stage-02 cal asdf. Runs in a worker process."""
    reasons: list[str] = []
    stats: dict = {}
    try:
        import roman_datamodels as rdm

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with rdm.open(path) as model:
                # Rung 1: schema validate
                try:
                    model.validate()
                except Exception as e:
                    reasons.append(f"schema validate failed: {type(e).__name__}: {e}")

                # Required meta fields
                for field_name in ("exposure", "observation", "instrument", "wcsinfo"):
                    if not hasattr(model.meta, field_name):
                        reasons.append(f"meta.{field_name} missing")

                # Rung 2: data array
                data = np.asarray(model.data)
                stats["shape"] = list(data.shape)
                if tuple(data.shape) != EXPECTED_SHAPE:
                    reasons.append(f"data shape {tuple(data.shape)} != {EXPECTED_SHAPE}")

                nan_mask = np.isnan(data)
                nan_frac = float(nan_mask.mean())
                stats["nan_frac"] = nan_frac
                if nan_frac > NAN_FRAC_MAX:
                    reasons.append(f"NaN fraction {nan_frac:.4f} > {NAN_FRAC_MAX}")

                valid = data[~nan_mask]
                if valid.size > 0:
                    stats["data_median"] = float(np.median(valid))
                    stats["data_p99"] = float(np.percentile(valid, 99))
                    if not np.isfinite(stats["data_median"]):
                        reasons.append(f"data median is non-finite: {stats['data_median']}")
                else:
                    reasons.append("data is all-NaN")

                # DQ array
                dq = np.asarray(model.dq)
                bad_frac = float((dq != 0).mean())
                stats["dq_bad_frac"] = bad_frac
                if bad_frac > DQ_BAD_FRAC_MAX:
                    reasons.append(f"DQ-bad fraction {bad_frac:.4f} > {DQ_BAD_FRAC_MAX}")

                # ERR array — must be positive where DQ is good. Bad pixels
                # legitimately get err=0 from upstream; only flag non-positive
                # err where DQ also says the pixel is good.
                err = np.asarray(model.err)
                good = (dq == 0) & np.isfinite(err)
                err_good = err[good]
                stats["err_median"] = float(np.median(err_good)) if err_good.size else float("nan")
                if err_good.size and (err_good <= 0).any():
                    n_nonpos = int((err_good <= 0).sum())
                    reasons.append(f"err has {n_nonpos} non-positive values where DQ==good")

    except Exception as e:
        reasons.append(f"open failed: {type(e).__name__}: {e}")

    return Result(path=path, passed=len(reasons) == 0, reasons=reasons, stats=stats)


def _summarize_stats(results: list[Result]) -> dict:
    """Aggregate per-file numeric stats so thresholds can be calibrated."""
    keys: set[str] = set()
    for r in results:
        keys.update(k for k, v in r.stats.items() if isinstance(v, (int, float)))
    summary: dict = {}
    for k in sorted(keys):
        vals = np.array(
            [r.stats[k] for r in results if k in r.stats and isinstance(r.stats[k], (int, float))]
        )
        if vals.size:
            summary[k] = {
                "n": int(vals.size),
                "min": float(vals.min()),
                "median": float(np.median(vals)),
                "p99": float(np.percentile(vals, 99)),
                "max": float(vals.max()),
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("config", help="path to configs/<tag>.yaml")
    parser.add_argument("--workers", type=int, default=8, help="parallel processes (default 8)")
    parser.add_argument(
        "--out",
        default=None,
        help="JSON summary output path; default ${output_base}/<tag>/validation_summary.json",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="validate only the first N files (debug)"
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from _config import load_config

    cfg = load_config(args.config)
    cal_dir = Path(cfg.output_base) / "cal"
    files = sorted(str(p) for p in cal_dir.glob("*.asdf"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"no cal files in {cal_dir}", file=sys.stderr)
        return 1

    print(f"validating {len(files)} cal files with {args.workers} workers...")
    results: list[Result] = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_validate_cal, p): p for p in files}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            n_done += 1
            if not r.passed:
                print(f"  FAIL {Path(r.path).name}: {'; '.join(r.reasons)}")
            elif n_done % 200 == 0:
                print(f"  ...{n_done}/{len(files)} ok so far")

    n_pass = sum(r.passed for r in results)
    n_fail = len(results) - n_pass
    print(f"\nDone: {n_pass}/{len(results)} pass, {n_fail} fail")

    out_path = (
        Path(args.out)
        if args.out
        else Path(cfg.output_base) / cfg.tag / "validation_summary.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "config": str(args.config),
        "tag": cfg.tag,
        "output_base": str(cfg.output_base),
        "stage": "cal",
        "thresholds": {
            "nan_frac_max": NAN_FRAC_MAX,
            "dq_bad_frac_max": DQ_BAD_FRAC_MAX,
            "expected_shape": list(EXPECTED_SHAPE),
        },
        "n_files": len(results),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "stats_distribution": _summarize_stats(results),
        "fails": [asdict(r) for r in sorted(results, key=lambda r: r.path) if not r.passed],
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"summary: {out_path}")

    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
