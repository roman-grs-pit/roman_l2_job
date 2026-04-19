#!/usr/bin/env python3
"""Config loader + validator for roman_l2_job pipeline runs.

Each pipeline stage reads one YAML config (`configs/<tag>.yaml`) via this
module. Config-file semantics:

    - Every field is required. Values cannot be silently defaulted in code.
      Fields that are semantically optional (e.g. `only_pass`) take the
      literal YAML value `null` to mean "not set"; the key must still be
      present.
    - `pointings.region` is a discriminated union: `type: cone` expects
      `ra`, `dec`, `radius_deg`; `type: box` expects `ra_min`, `ra_max`,
      `dec_min`, `dec_max`.

Usage from Python:

    from _config import load_config
    cfg = load_config("configs/smoke.yaml")
    cfg.tag                        # "smoke"
    cfg.catalog.input_units        # "mag" or "maggies"
    cfg.pointings.region.radius_deg

Usage from bash (one invocation per stage; cheap on warm pixi env):

    eval "$(pixi run python scripts/_config.py configs/smoke.yaml)"
    # now TAG, CATALOG_INPUT, CATALOG_INPUT_UNITS, CATALOG_BANDPASS_COL,
    # POINTINGS_BANDPASS, POINTINGS_ONLY_{PASS,SEGMENT,VISIT},
    # POINTINGS_REGION_TYPE, RUN_PARALLELISM, CONFIG_PATH are exported.
    # Region params expose under POINTINGS_REGION_<FIELD> depending on type.

When in doubt, prefer `load_config` over shell-eval — Python keeps field
names honest.
"""
from __future__ import annotations

import argparse
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml


@dataclass
class ConeRegion:
    ra: float
    dec: float
    radius_deg: float

    type: str = "cone"


@dataclass
class BoxRegion:
    ra_min: float
    ra_max: float
    dec_min: float
    dec_max: float

    type: str = "box"


Region = Union[ConeRegion, BoxRegion]


@dataclass
class Pointings:
    region: Region
    bandpass: str
    only_pass: int | None
    only_segment: int | None
    only_visit: int | None
    design_depth: int  # per-pixel coverage count intended by the filter choice


@dataclass
class Catalog:
    input: Path
    input_units: str
    bandpass_col: str


@dataclass
class Run:
    parallelism: int


@dataclass
class Config:
    tag: str
    catalog: Catalog
    pointings: Pointings
    run: Run
    path: Path  # back-reference to the loaded config file


REQUIRED = {
    "tag": str,
    "catalog": dict,
    "pointings": dict,
    "run": dict,
}
REQUIRED_CATALOG = {
    "input": str,
    "input_units": str,
    "bandpass_col": str,
}
REQUIRED_POINTINGS = {
    "region": dict,
    "bandpass": str,
    "only_pass": object,       # int | None
    "only_segment": object,
    "only_visit": object,
    "design_depth": int,
}
REQUIRED_RUN = {
    "parallelism": int,
}
REQUIRED_REGION_CONE = {
    "type": str,
    "ra": (int, float),
    "dec": (int, float),
    "radius_deg": (int, float),
}
REQUIRED_REGION_BOX = {
    "type": str,
    "ra_min": (int, float),
    "ra_max": (int, float),
    "dec_min": (int, float),
    "dec_max": (int, float),
}


def _require(section: str, got: dict, required: dict):
    missing = sorted(set(required) - set(got))
    if missing:
        raise ValueError(f"config[{section}] missing required keys: {missing}")
    extra = sorted(set(got) - set(required))
    if extra:
        raise ValueError(f"config[{section}] has unknown keys: {extra}")
    for k, t in required.items():
        if t is object:
            continue  # caller handles
        if not isinstance(got[k], t):
            raise ValueError(
                f"config[{section}].{k} has type {type(got[k]).__name__}, "
                f"expected {t if isinstance(t, type) else t}")


def _region(got: dict) -> Region:
    kind = got.get("type")
    if kind == "cone":
        _require("pointings.region[cone]", got, REQUIRED_REGION_CONE)
        return ConeRegion(ra=float(got["ra"]), dec=float(got["dec"]),
                          radius_deg=float(got["radius_deg"]))
    if kind == "box":
        _require("pointings.region[box]", got, REQUIRED_REGION_BOX)
        return BoxRegion(ra_min=float(got["ra_min"]),
                         ra_max=float(got["ra_max"]),
                         dec_min=float(got["dec_min"]),
                         dec_max=float(got["dec_max"]))
    raise ValueError(f"pointings.region.type must be 'cone' or 'box', got {kind!r}")


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at top level")

    _require("<top>", raw, REQUIRED)
    _require("catalog", raw["catalog"], REQUIRED_CATALOG)
    _require("pointings", raw["pointings"], REQUIRED_POINTINGS)
    _require("run", raw["run"], REQUIRED_RUN)

    pts = raw["pointings"]
    for k in ("only_pass", "only_segment", "only_visit"):
        v = pts[k]
        if v is not None and not isinstance(v, int):
            raise ValueError(
                f"config[pointings].{k} must be int or null, got {type(v).__name__}")

    cat = raw["catalog"]
    if cat["input_units"] not in {"mag", "maggies"}:
        raise ValueError(
            f"config[catalog].input_units must be 'mag' or 'maggies', got "
            f"{cat['input_units']!r}")

    return Config(
        tag=raw["tag"],
        catalog=Catalog(input=Path(cat["input"]),
                        input_units=cat["input_units"],
                        bandpass_col=cat["bandpass_col"]),
        pointings=Pointings(region=_region(pts["region"]),
                            bandpass=pts["bandpass"],
                            only_pass=pts["only_pass"],
                            only_segment=pts["only_segment"],
                            only_visit=pts["only_visit"],
                            design_depth=pts["design_depth"]),
        run=Run(parallelism=raw["run"]["parallelism"]),
        path=path,
    )


def _shell_quote(v) -> str:
    if v is None:
        return ""
    return shlex.quote(str(v))


def _export_for_shell(cfg: Config) -> str:
    lines = [
        f"export CONFIG_PATH={_shell_quote(cfg.path)}",
        f"export TAG={_shell_quote(cfg.tag)}",
        f"export CATALOG_INPUT={_shell_quote(cfg.catalog.input)}",
        f"export CATALOG_INPUT_UNITS={_shell_quote(cfg.catalog.input_units)}",
        f"export CATALOG_BANDPASS_COL={_shell_quote(cfg.catalog.bandpass_col)}",
        f"export POINTINGS_BANDPASS={_shell_quote(cfg.pointings.bandpass)}",
        f"export POINTINGS_ONLY_PASS={_shell_quote(cfg.pointings.only_pass)}",
        f"export POINTINGS_ONLY_SEGMENT={_shell_quote(cfg.pointings.only_segment)}",
        f"export POINTINGS_ONLY_VISIT={_shell_quote(cfg.pointings.only_visit)}",
        f"export POINTINGS_DESIGN_DEPTH={_shell_quote(cfg.pointings.design_depth)}",
        f"export POINTINGS_REGION_TYPE={_shell_quote(cfg.pointings.region.type)}",
        f"export RUN_PARALLELISM={_shell_quote(cfg.run.parallelism)}",
    ]
    r = cfg.pointings.region
    if isinstance(r, ConeRegion):
        lines += [
            f"export POINTINGS_REGION_RA={_shell_quote(r.ra)}",
            f"export POINTINGS_REGION_DEC={_shell_quote(r.dec)}",
            f"export POINTINGS_REGION_RADIUS_DEG={_shell_quote(r.radius_deg)}",
        ]
    else:  # BoxRegion
        lines += [
            f"export POINTINGS_REGION_RA_MIN={_shell_quote(r.ra_min)}",
            f"export POINTINGS_REGION_RA_MAX={_shell_quote(r.ra_max)}",
            f"export POINTINGS_REGION_DEC_MIN={_shell_quote(r.dec_min)}",
            f"export POINTINGS_REGION_DEC_MAX={_shell_quote(r.dec_max)}",
        ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to YAML config file")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        print(f"_config.py: {e}", file=sys.stderr)
        sys.exit(1)

    print(_export_for_shell(cfg))


if __name__ == "__main__":
    main()
