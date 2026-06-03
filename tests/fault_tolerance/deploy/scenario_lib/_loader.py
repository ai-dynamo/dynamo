# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# YAML → Scenario dataclass loader. Strict: unknown keys at any nesting
# level raise a ValueError so typos surface at pytest collection time.

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml

from ._schema import (
    Admission,
    CheckSpec,
    Deployment,
    EventSpec,
    ExpectedRange,
    Load,
    LoadCommon,
    ReportSpec,
    Router,
    Rung,
    Scenario,
    Shape,
)

SCENARIOS_DIR = Path(__file__).parent

VALID_SHAPE_TYPES = {
    "no_prefix",
    "same_prefix",
    "partial_prefix",
    "long_isl",
    "high_qps",
    "custom",
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def load_scenario(path: Path) -> Scenario:
    """Load a scenario YAML and validate. Raises ValueError on bad input."""
    text = path.read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")
    parent_kind = path.parent.name
    if raw.get("kind") != parent_kind:
        raise ValueError(
            f"{path}: kind={raw.get('kind')!r} but lives under "
            f"{parent_kind!r}/. Move the file, or fix the kind field."
        )
    s = _scenario_from_dict(raw, path)
    s.path = str(path)
    return s


def discover_scenarios(kind: str) -> list[Path]:
    """Return sorted list of scenario YAML paths for the given kind
    (= subdirectory name). Returns empty list if the subdir doesn't
    exist (so a kind that hasn't shipped its first scenario yet
    collects zero tests instead of erroring)."""
    subdir = SCENARIOS_DIR / kind
    if not subdir.is_dir():
        return []
    return sorted(p for p in subdir.glob("*.yaml") if not p.name.startswith("_"))


# --------------------------------------------------------------------------- #
# Internal: strict dict → dataclass conversion
# --------------------------------------------------------------------------- #


def _scenario_from_dict(raw: dict, path: Path) -> Scenario:
    return Scenario(
        kind=_require_str(raw, "kind", path),
        name=_require_str(raw, "name", path),
        description=raw.get("description", ""),
        labels=_as_str_dict(raw.get("labels", {}), path, "labels"),
        deployment=_deployment(raw.get("deployment", {}), path),
        router=_router(raw.get("router", {}), path),
        admission=_admission(raw["admission"], path) if "admission" in raw else None,
        load=_load(raw["load"], path) if "load" in raw else None,
        events=[_event(e, path) for e in raw.get("events", [])],
        reports=[_report(r, path) for r in raw.get("reports", [])],
        checks=[_check(c, path) for c in raw.get("checks", [])],
        expectations=_expectations(raw.get("expectations", {}), path),
    )


def _deployment(d: dict, path: Path) -> Deployment:
    _reject_unknown(
        d, {f.name for f in dataclasses.fields(Deployment)}, path, "deployment"
    )
    if "backend" not in d:
        raise ValueError(f"{path}: deployment.backend is required")
    return Deployment(**d)


def _router(d: dict, path: Path) -> Router:
    _reject_unknown(d, {f.name for f in dataclasses.fields(Router)}, path, "router")
    knobs = _as_str_dict(d.get("knobs", {}), path, "router.knobs")
    return Router(mode=d.get("mode", "kv"), knobs=knobs)


def _admission(d: dict, path: Path) -> Admission:
    _reject_unknown(
        d, {f.name for f in dataclasses.fields(Admission)}, path, "admission"
    )
    knobs = _as_str_dict(d.get("knobs", {}), path, "admission.knobs")
    return Admission(knobs=knobs)


def _load(d: dict, path: Path) -> Load:
    _reject_unknown(d, {f.name for f in dataclasses.fields(Load)}, path, "load")
    if "shape" not in d:
        raise ValueError(f"{path}: load.shape is required")
    shape = _shape(d["shape"], path, "load.shape")
    rungs = [_rung(r, path, i) for i, r in enumerate(d.get("rungs", []))]
    common = _load_common(d.get("common", {}), path)
    return Load(shape=shape, rungs=rungs, common=common)


def _shape(d: dict, path: Path, where: str) -> Shape:
    _reject_unknown(d, {f.name for f in dataclasses.fields(Shape)}, path, where)
    if "type" not in d:
        raise ValueError(f"{path}: {where}.type is required")
    t = d["type"]
    if t not in VALID_SHAPE_TYPES:
        raise ValueError(
            f"{path}: {where}.type={t!r} not in {sorted(VALID_SHAPE_TYPES)}"
        )
    return Shape(**d)


def _rung(d: dict, path: Path, idx: int) -> Rung:
    where = f"load.rungs[{idx}]"
    _reject_unknown(d, {f.name for f in dataclasses.fields(Rung)}, path, where)
    for f in ("name", "concurrency", "duration_minutes"):
        if f not in d:
            raise ValueError(f"{path}: {where}.{f} is required")
    shape = _shape(d["shape"], path, f"{where}.shape") if "shape" in d else None
    rr = d.get("request_rate")
    return Rung(
        name=d["name"],
        concurrency=int(d["concurrency"]),
        duration_minutes=float(d["duration_minutes"]),
        request_rate=float(rr) if rr is not None else None,
        shape=shape,
    )


def _load_common(d: dict, path: Path) -> LoadCommon:
    _reject_unknown(
        d, {f.name for f in dataclasses.fields(LoadCommon)}, path, "load.common"
    )
    return LoadCommon(**d)


def _event(d: dict, path: Path) -> EventSpec:
    if not isinstance(d, dict) or "kind" not in d:
        raise ValueError(f"{path}: each event must be a mapping with a 'kind' field")
    return EventSpec(kind=d["kind"], params={k: v for k, v in d.items() if k != "kind"})


def _report(d: dict, path: Path) -> ReportSpec:
    if isinstance(d, str):
        # Shorthand: just the kind name, no params.
        return ReportSpec(kind=d, params={})
    if not isinstance(d, dict) or "kind" not in d:
        raise ValueError(
            f"{path}: each report must be a kind string or {{kind, ...}} mapping"
        )
    return ReportSpec(
        kind=d["kind"], params={k: v for k, v in d.items() if k != "kind"}
    )


def _check(d: dict, path: Path) -> CheckSpec:
    if not isinstance(d, dict) or "kind" not in d:
        raise ValueError(f"{path}: each check must be a mapping with a 'kind' field")
    return CheckSpec(kind=d["kind"], params={k: v for k, v in d.items() if k != "kind"})


def _expectations(d: dict, path: Path) -> dict[str, ExpectedRange]:
    out: dict[str, ExpectedRange] = {}
    for k, v in d.items():
        if not isinstance(v, dict):
            raise ValueError(f"{path}: expectations.{k} must be a mapping")
        _reject_unknown(
            v,
            {f.name for f in dataclasses.fields(ExpectedRange)},
            path,
            f"expectations.{k}",
        )
        out[k] = ExpectedRange(**v)
    return out


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _require_str(d: dict, key: str, path: Path) -> str:
    if key not in d or not isinstance(d[key], str):
        raise ValueError(f"{path}: required string field {key!r} missing")
    return d[key]


def _as_str_dict(v: Any, path: Path, where: str) -> dict[str, str]:
    if not isinstance(v, dict):
        raise ValueError(f"{path}: {where} must be a mapping")
    out: dict[str, str] = {}
    for k, val in v.items():
        if not isinstance(k, str):
            raise ValueError(f"{path}: {where} keys must be strings, got {k!r}")
        # Coerce non-string values to strings (YAML may parse "false" as bool, etc.)
        out[k] = str(val)
    return out


def _reject_unknown(d: dict, allowed: set[str], path: Path, where: str) -> None:
    unknown = set(d.keys()) - allowed
    if unknown:
        raise ValueError(
            f"{path}: unknown fields in {where}: {sorted(unknown)}. "
            f"Allowed: {sorted(allowed)}"
        )
