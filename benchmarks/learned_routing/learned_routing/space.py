# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``space.yaml``: the free and fixed parameters ``lr-train`` searches, and their transforms.

Example::

    base:                          # a harness policy spec (see learned_routing.policy)
      name: learned-choice-M1
      type: learned-choice
      parameters: {feature_set: v1, temperature: 0.0}
    fixed:                         # dotted path -> value, applied on top of base
      parameters.context: {p: [], q: []}
    params:
      - path: router_config.router_temperature
        kind: float                # float | int | vector | matrix | lowrank
        bounds: [0.0, 2.0]
        scale: linear              # linear | log (log needs 0 < low)
        init: 0.0
      - path: parameters.theta
        kind: vector
        dim: 8
        bounds: [-10.0, 10.0]      # one pair, or one pair (or null) per element
        init: [-1, 0, 0, 0, 0, 0, 0, 0]
        fixed: {0: -1.0}           # pinned elements, e.g. the LR-05 scale anchor
        clamp: [null, [null, null], [null, 0.0], null, null, null, null, null]  # theta[2] <= 0
      - path: parameters.context   # {p: [[dim]] * rank, q: [[dim]] * rank}
        kind: lowrank
        dim: 8
        rank: 1
        bounds: [-5.0, 5.0]
        init: 0.0
    cma: {sigma0: 0.2, tolflatfitness: 10}   # optional pycma options

**Internal coordinates.** CMA-ES searches internal coordinates, not physical values:

- bounded linear: ``u in [0, 1]``, ``x = low + u (high - low)``;
- bounded log: ``u in [0, 1]``, ``x = exp(ln low + u (ln high - ln low))``;
- unbounded (``bounds: null``): ``u = x / unit`` (``unit`` defaults to 1);
- ``int``: the linear or log value rounded to the nearest integer.

**Clamp (physical clip).** ``clamp`` (one ``[low, high]`` pair, or one pair or null per element;
either end may be null) clips the physical value after the transform:
``x = min(max(x, low), high)``. The internal box and its geometry are unchanged, so a clamped
space searches exactly the same internal coordinates, step sizes and start as the unclamped one,
and only the decoded policy differs. Use it for sign constraints whose natural start sits on the
constraint (e.g. ``theta_j <= 0`` from ``theta_j = 0``): moving the box bound there instead puts the
start on pycma's boundary transform, whose slope is zero at the bound, so the first generations
barely move that coordinate. ``init`` must satisfy the clamp.

Bounded coordinates get the box ``[0, 1]`` (pycma's ``BoundTransform``). ``std`` gives a
per-parameter initial standard deviation in internal units (pycma ``CMA_stds``, relative to
``sigma0``); it defaults to ``sigma0``. Pinned elements are not searched.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from learned_routing.canon import canonical_json, sha256_text
from learned_routing.policy import PolicySpec, spec_from_dict

KINDS = ("float", "int", "vector", "matrix", "lowrank")
DEFAULT_SIGMA0 = 0.2


class SpaceError(ValueError):
    pass


@dataclass(frozen=True)
class Coord:
    """One searched scalar: where it goes and how internal u maps to physical x."""

    param: str
    index: tuple  # position inside the parameter's value
    low: float | None
    high: float | None
    scale: str
    integer: bool
    unit: float
    std: float | None
    clip_low: float | None = None
    clip_high: float | None = None

    @property
    def bounded(self) -> bool:
        return self.low is not None and self.high is not None

    def to_physical(self, u: float) -> float | int:
        if not self.bounded:
            x = u * self.unit
        else:
            u = min(max(u, 0.0), 1.0)
            if self.scale == "log":
                x = math.exp(
                    math.log(self.low) + u * (math.log(self.high) - math.log(self.low))
                )
            else:
                x = self.low + u * (self.high - self.low)
            x = min(max(x, self.low), self.high)
        if self.clip_low is not None:
            x = max(x, self.clip_low)
        if self.clip_high is not None:
            x = min(x, self.clip_high)
        return int(round(x)) if self.integer else float(x)

    def to_internal(self, x: float) -> float:
        if not self.bounded:
            return float(x) / self.unit
        if not self.low <= x <= self.high:
            raise SpaceError(
                f"{self.param}{list(self.index)}: init {x} outside [{self.low}, {self.high}]"
            )
        if self.high == self.low:
            return 0.5
        if self.scale == "log":
            return (math.log(x) - math.log(self.low)) / (
                math.log(self.high) - math.log(self.low)
            )
        return (x - self.low) / (self.high - self.low)


def _set_path(target: dict, path: str, value: Any) -> None:
    keys = path.split(".")
    node = target
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise SpaceError(f"path {path!r} crosses a non-mapping at {key!r}")
    node[keys[-1]] = value


def _element_bounds(spec: dict, count: int, path: str) -> list[tuple]:
    bounds = spec.get("bounds")
    if bounds is None:
        return [(None, None)] * count
    if len(bounds) == 2 and all(
        isinstance(b, (int, float)) or b is None for b in bounds
    ):
        pair = tuple(None if b is None else float(b) for b in bounds)
        return [pair] * count
    if len(bounds) != count:
        raise SpaceError(f"{path}: bounds needs one pair or {count} per-element pairs")
    out = []
    for pair in bounds:
        out.append((None, None) if pair is None else tuple(float(b) for b in pair))
    return out


def _element_clamps(spec: dict, count: int, path: str) -> list[tuple]:
    clamp = spec.get("clamp")
    if clamp is None:
        return [(None, None)] * count

    def pair(value) -> tuple:
        if value is None:
            return (None, None)
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise SpaceError(
                f"{path}: clamp entries must be [low, high] (null for open)"
            )
        lo, hi = (None if b is None else float(b) for b in value)
        if lo is not None and hi is not None and lo > hi:
            raise SpaceError(f"{path}: clamp must have low <= high")
        return (lo, hi)

    if len(clamp) == 2 and all(isinstance(b, (int, float)) or b is None for b in clamp):
        return [pair(clamp)] * count
    if len(clamp) != count:
        raise SpaceError(f"{path}: clamp needs one pair or {count} per-element entries")
    return [pair(value) for value in clamp]


def _flat_init(spec: dict, shape: tuple[int, ...], path: str) -> list[float]:
    count = math.prod(shape)
    init = spec.get("init", 0.0)
    if isinstance(init, (int, float)):
        return [float(init)] * count
    flat: list = []

    def walk(value):
        if isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        else:
            flat.append(float(value))

    walk(init)
    if len(flat) != count:
        raise SpaceError(f"{path}: init has {len(flat)} values, expected {count}")
    return flat


@dataclass
class Param:
    path: str
    kind: str
    shape: tuple[int, ...]
    init: list[float]
    pinned: dict[int, float]
    coords: list[Coord]
    free_index: list[int]  # flat indices that are searched
    fields: tuple[str, str] = ("p", "q")

    def assemble(self, values: list[float | int]) -> Any:
        flat = list(self.init)
        for flat_index, value in self.pinned.items():
            flat[flat_index] = value
        for flat_index, value in zip(self.free_index, values):
            flat[flat_index] = value
        if self.kind in ("float", "int"):
            return flat[0]
        if self.kind == "vector":
            return flat
        if self.kind == "matrix":
            rows, cols = self.shape
            return [flat[r * cols : (r + 1) * cols] for r in range(rows)]
        rank, dim = self.shape[1], self.shape[2]
        half = rank * dim
        p = [flat[r * dim : (r + 1) * dim] for r in range(rank)]
        q = [flat[half + r * dim : half + (r + 1) * dim] for r in range(rank)]
        return {self.fields[0]: p, self.fields[1]: q}


def _identity_view(raw: dict) -> dict:
    """``raw`` with each param's pinned-element keys as decimal strings.

    YAML reads ``fixed: {0: -1.0}`` with integer keys, which canonical JSON rejects; ``0`` and
    ``"0"`` pin the same element, so they hash the same.
    """
    params = []
    for spec in raw.get("params") or []:
        if isinstance(spec, dict) and isinstance(spec.get("fixed"), dict):
            spec = {
                **spec,
                "fixed": {str(int(k)): v for k, v in spec["fixed"].items()},
            }
        params.append(spec)
    return {**raw, "params": params} if "params" in raw else raw


class Space:
    def __init__(self, raw: dict, source: str = "<space>"):
        if not isinstance(raw, dict):
            raise SpaceError(f"{source}: space must be a mapping")
        unknown = sorted(set(raw) - {"base", "fixed", "params", "cma", "notes"})
        if unknown:
            raise SpaceError(f"{source}: unknown space keys {unknown}")
        self.raw = raw
        self.base = dict(raw.get("base") or {})
        if not self.base:
            raise SpaceError(f"{source}: base policy spec is required")
        self.fixed = dict(raw.get("fixed") or {})
        self.cma_options = dict(raw.get("cma") or {})
        self.sigma0 = float(self.cma_options.pop("sigma0", DEFAULT_SIGMA0))
        self.params: list[Param] = [self._param(p) for p in raw.get("params") or []]
        paths = [p.path for p in self.params]
        if len(set(paths)) != len(paths):
            raise SpaceError(f"{source}: duplicate param paths")
        self.coords: list[Coord] = [c for p in self.params for c in p.coords]
        # Validate that base + fixed + init decode to a loadable spec.
        self.spec(self.x0())

    @classmethod
    def load(cls, path: str | Path) -> "Space":
        return cls(yaml.safe_load(Path(path).read_text()), str(path))

    @property
    def sha(self) -> str:
        return sha256_text(canonical_json(_identity_view(self.raw)))

    @property
    def dimension(self) -> int:
        return len(self.coords)

    def _param(self, spec: dict) -> Param:
        allowed = {
            "path",
            "kind",
            "bounds",
            "scale",
            "init",
            "std",
            "unit",
            "dim",
            "rank",
            "shape",
            "fixed",
            "fields",
            "clamp",
        }
        unknown = sorted(set(spec) - allowed)
        path = spec.get("path")
        if not path:
            raise SpaceError("every param needs a path")
        if unknown:
            raise SpaceError(f"{path}: unknown param keys {unknown}")
        kind = spec.get("kind", "float")
        if kind not in KINDS:
            raise SpaceError(f"{path}: kind must be one of {KINDS}")
        if kind in ("float", "int"):
            shape: tuple[int, ...] = (1,)
        elif kind == "vector":
            shape = (int(spec["dim"]),)
        elif kind == "matrix":
            shape = tuple(int(n) for n in spec["shape"])
            if len(shape) != 2:
                raise SpaceError(f"{path}: matrix shape must be [rows, cols]")
        else:
            shape = (2, int(spec.get("rank", 1)), int(spec["dim"]))
        count = math.prod(shape)
        scale = spec.get("scale", "linear")
        if scale not in ("linear", "log"):
            raise SpaceError(f"{path}: scale must be linear or log")
        bounds = _element_bounds(spec, count, path)
        if scale == "log" and any(lo is None or lo <= 0 for lo, _ in bounds):
            raise SpaceError(f"{path}: log scale needs bounds with 0 < low")
        for lo, hi in bounds:
            if (lo is None) != (hi is None) or (lo is not None and lo > hi):
                raise SpaceError(
                    f"{path}: bounds must be [low, high] with low <= high, or null"
                )
        clamps = _element_clamps(spec, count, path)
        init = _flat_init(spec, shape, path)
        pinned = {int(k): float(v) for k, v in (spec.get("fixed") or {}).items()}
        if any(not 0 <= k < count for k in pinned):
            raise SpaceError(f"{path}: pinned index out of range 0..{count - 1}")
        unit = float(spec.get("unit", 1.0))
        std = spec.get("std")
        coords, free = [], []
        for flat_index in range(count):
            if flat_index in pinned:
                continue
            lo, hi = bounds[flat_index]
            clip_lo, clip_hi = clamps[flat_index]
            value = init[flat_index]
            if (clip_lo is not None and value < clip_lo) or (
                clip_hi is not None and value > clip_hi
            ):
                raise SpaceError(
                    f"{path}[{flat_index}]: init {value} outside clamp [{clip_lo}, {clip_hi}]"
                )
            coords.append(
                Coord(
                    param=path,
                    index=(flat_index,),
                    low=lo,
                    high=hi,
                    scale=scale,
                    integer=kind == "int",
                    unit=unit,
                    std=None if std is None else float(std),
                    clip_low=clip_lo,
                    clip_high=clip_hi,
                )
            )
            free.append(flat_index)
        fields = tuple(spec.get("fields", ("p", "q")))
        if len(fields) != 2:
            raise SpaceError(f"{path}: fields must name two outputs")
        return Param(path, kind, shape, init, pinned, coords, free, fields)

    def x0(self) -> list[float]:
        out = []
        for param in self.params:
            for coord, flat_index in zip(param.coords, param.free_index):
                out.append(coord.to_internal(param.init[flat_index]))
        return out

    def decode(self, z: list[float]) -> dict[str, Any]:
        if len(z) != self.dimension:
            raise SpaceError(f"expected {self.dimension} coordinates, got {len(z)}")
        values: dict[str, Any] = {}
        offset = 0
        for param in self.params:
            n = len(param.coords)
            physical = [
                c.to_physical(u) for c, u in zip(param.coords, z[offset : offset + n])
            ]
            values[param.path] = param.assemble(physical)
            offset += n
        return values

    def spec_dict(self, z: list[float]) -> dict:
        spec = copy.deepcopy(self.base)
        for path, value in self.fixed.items():
            _set_path(spec, path, copy.deepcopy(value))
        for path, value in self.decode(z).items():
            _set_path(spec, path, value)
        return spec

    def spec(self, z: list[float], name: str | None = None) -> PolicySpec:
        raw = self.spec_dict(z)
        if name:
            raw["name"] = name
        return spec_from_dict(raw)

    def cma_bounds(self) -> list[list[float | None]]:
        lower = [0.0 if c.bounded else None for c in self.coords]
        upper = [1.0 if c.bounded else None for c in self.coords]
        return [lower, upper]

    def cma_stds(self) -> list[float] | None:
        if all(c.std is None for c in self.coords):
            return None
        return [
            (c.std if c.std is not None else self.sigma0) / self.sigma0
            for c in self.coords
        ]
