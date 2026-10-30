# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live retune sweep executor: one window through DGDR v1beta2 Requests.

``run_dgdr_sweep`` is the live ``sweep_fn`` of ``RetuneController`` (design
§4.5). It runs on the controller's single sweep thread and never raises:

1. Shard the router grid (``plan_shards``): balanced Cartesian sub-grids of
   at most 100 points (a Request's ``maxCandidates`` bound), one
   single-point shard for an off-grid baseline or startup point, and a CRN
   probe that evaluates one fixed grid point a second time.
2. Adopt the Requests a previous leader created for this window, matched by
   ``(dgd-uid, window, shard, spec hash)``; others of the window are queued
   for deletion. A window whose every shard is adopted as Succeeded is
   harvested without cutting it again. An adopted attempt whose Run failed
   or ran another input is replaced by attempt+1 while the shard still
   fits before the deadline.
3. Cut the window file (a subprocess, ``context.cut_window``), then create
   one Request per remaining shard with Strict field validation and check
   that the API server kept every submitted field.
4. Wait for each Request's Run, check that ``Run.status.aisimulate.input``
   is what the Request asked for, and read the Run's Candidates.
5. Map each Candidate's router adapter parameters back to live names,
   require identical non-router parameters across the window, score each
   with ``score_summary`` on ``retune.objective``, and check the CRN probe.

The executor never deletes anything itself: Requests to remove go to the
``SweepRegistry`` deletion queue, which the reconciler drains while this
process leads. Shutdown (``cancel``) and loss of leadership (``abort``)
return at once with ``failure_reason`` ``shutdown``/``lost_leadership`` and
leave every Request in place for the next leader to adopt. Every mutating
call first checks ``abort`` and ``context.is_leader()``.

A shard that holds the baseline, or the probe, is critical: when it fails
the window fails. Any other failed shard only leaves its points unscored.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import itertools
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, NoReturn, Optional

from dynamo.planner.config.planner_config import DgdrExecutorConfig, RetuneConfig
from dynamo.planner.core.types import RetuneParams
from dynamo.planner.retune.live.contracts import (
    DGDR_GROUP,
    DGDR_ROUTER_KNOBS,
    DGDR_VERSION,
    LABEL_ATTEMPT,
    LABEL_DGD,
    LABEL_DGD_UID,
    LABEL_PREFIX,
    LABEL_SHARD,
    LABEL_SPEC,
    LABEL_WINDOW,
    RUN_UID_LABEL,
    DgdRef,
    KubeAlreadyExists,
    KubeError,
    KubeInvalid,
    KubeNotFound,
    RetuneKube,
    SweepRegistry,
    WindowCutResult,
    WindowCutter,
)
from dynamo.planner.retune.sweep import (
    SweepRequest,
    SweepResult,
    SweepStatus,
    score_summary,
)

logger = logging.getLogger(__name__)

ShardKind = Literal["grid", "point", "probe"]

# A Request publishes at most recommendation.maxCandidates (<= 100) points.
MAX_SHARD_POINTS = 100
# Seconds a new Request may go without an Accepted condition before the
# window fails as operator_not_reconciling.
ACCEPT_TIMEOUT_S = 60.0
# Relative score difference the CRN probe tolerates.
CRN_REL_TOLERANCE = 1e-9
# Oldest Dynamo runtime whose trace replay runs with common random numbers.
# None: the version is not checked and the CRN probe alone decides.
CRN_MIN_VERSION: Optional[str] = None
# Annotation holding the cut window's row counts, so a window adopted as
# Succeeded reports them without cutting the window again.
ROWS_ANNOTATION = LABEL_PREFIX + "rows"
# Operator constants (deploy/operator at #17483).
REQUEST_GENERATION_ANNOTATION = "nvidia.com/dgdr-request-generation"
TRACE_MOUNT_PATH = "/var/run/aisimulate/trace"
ROUTER_ADAPTER = "dynamo.router"
_ROLES = ("aggregated", "prefill", "decode")
_TEMPERATURE = DGDR_ROUTER_KNOBS["temperature"]
# spec.objective.metric -> AISimulate optimization.target (operator lowering).
_METRIC_TARGETS = {
    "Throughput": "throughput",
    "ThroughputPerGPU": "throughput_per_gpu",
    "ThroughputPerUser": "throughput_per_user",
    "TTFT": "ttft",
    "E2ELatency": "e2e_latency",
    "Goodput": "goodput",
    "GoodputPerGPU": "goodput_per_gpu",
}
# Statuses a retry cannot fix (bad request, auth, missing API).
_PERMANENT_STATUSES = frozenset({400, 401, 403, 404, 405, 422})
# How the API server words a rejection by an admission webhook.
_ADMISSION_DENIED = re.compile(r'admission webhook "[^"]*" denied the request')
_K8S_TIMEOUT_S = 10.0
_SNAP_TOLERANCE = 1e-9
_MIN_K8S_TIMEOUT_S = 1.0
_ABSENT = object()


# ------------------------------------------------------------------ shards


@dataclass(frozen=True)
class Shard:
    """One Request's slice of the window's candidates.

    ``choices`` maps each live router knob to the values the shard spans;
    ``points`` is their Cartesian product (live names). A ``point`` or
    ``probe`` shard has one value per knob.
    """

    id: str
    kind: ShardKind
    choices: dict[str, tuple[float, ...]]
    points: tuple[dict[str, float], ...]

    @property
    def size(self) -> int:
        return len(self.points)

    def contains(self, point: Mapping[str, float]) -> bool:
        return all(point.get(key) in values for key, values in self.choices.items())


def plan_shards(
    router_domain: Mapping[str, Sequence[float]],
    *,
    baseline: Mapping[str, float],
    startup: Mapping[str, float],
    max_shards: int,
    probe: bool = True,
    parallelism: int | None = None,
) -> list[Shard]:
    """Split ``router_domain`` (live knob -> values) into Request shards.

    The grid becomes at most ``max_shards`` Cartesian sub-grids of at most
    ``MAX_SHARD_POINTS`` points: each knob's sorted values are cut into
    contiguous, near-equal chunks. The split minimizes the largest shard's
    wall time, then the shard count: with the Run ``parallelism`` the time
    is its evaluation rounds (``1 + ceil((n - 1) / parallelism)``, as in
    ``need_seconds``), so a shard that would not shorten the window is not
    added; without it, the largest shard's size. ``baseline`` and ``startup``
    (live knob -> value) each get a single-point shard when off the grid.
    With ``probe``, a ``probe`` shard repeats one fixed grid point (the
    middle value of each knob, a positive temperature when the grid has
    one) to check common random numbers across Runs. Raises ``ValueError``
    when no split fits or a point does not cover the domain's knobs.
    """

    if not router_domain:
        raise ValueError("router domain is empty; there is nothing to shard")
    if max_shards < 1:
        raise ValueError(f"max_shards must be at least 1, got {max_shards}")
    keys = tuple(sorted(router_domain))
    axes: dict[str, tuple[float, ...]] = {}
    for key in keys:
        values = tuple(sorted({float(value) for value in router_domain[key]}))
        if not values:
            raise ValueError(f"router domain {key!r} has no values")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"router domain {key!r} has a non-finite value")
        axes[key] = values
    for name, point in (("baseline", baseline), ("startup", startup)):
        if set(point) != set(keys):
            raise ValueError(
                f"{name} router point keys {sorted(point)} do not match the "
                f"router domain {list(keys)}"
            )

    if parallelism is not None and parallelism < 1:
        raise ValueError(f"parallelism must be at least 1, got {parallelism}")
    sizes = [len(axes[key]) for key in keys]
    best: tuple[float, int, int, tuple[int, ...]] | None = None
    for splits in _splits(sizes, max_shards):
        largest = math.prod(
            math.ceil(size / split) for size, split in zip(sizes, splits)
        )
        # An oversized shard can never be chosen over one that fits.
        cost = (
            math.inf
            if largest > MAX_SHARD_POINTS
            else largest
            if parallelism is None
            else _rounds(largest, parallelism)
        )
        candidate = (cost, math.prod(splits), largest, splits)
        if best is None or candidate < best:
            best = candidate
    assert best is not None  # all-ones always fits max_shards >= 1
    _, _, largest, splits = best
    if largest > MAX_SHARD_POINTS:
        raise ValueError(
            f"the router grid of {math.prod(sizes)} points cannot be split into "
            f"at most {max_shards} Cartesian shards of at most "
            f"{MAX_SHARD_POINTS} points (the smallest largest shard has "
            f"{largest}); raise retune.dgdr.run.max_shards or shrink the domain"
        )

    shards: list[Shard] = []
    chunked = [_chunks(axes[key], split) for key, split in zip(keys, splits)]
    for index, combo in enumerate(itertools.product(*chunked)):
        choices = dict(zip(keys, combo))
        shards.append(
            Shard(
                id=f"grid-{index}",
                kind="grid",
                choices=choices,
                points=_product(choices),
            )
        )

    def in_grid(point: Mapping[str, float]) -> bool:
        return all(float(point[key]) in axes[key] for key in keys)

    def single(shard_id: str, kind: ShardKind, point: Mapping[str, float]) -> Shard:
        values = {key: float(point[key]) for key in keys}
        return Shard(
            id=shard_id,
            kind=kind,
            choices={key: (value,) for key, value in values.items()},
            points=(values,),
        )

    baseline_point = {key: float(baseline[key]) for key in keys}
    startup_point = {key: float(startup[key]) for key in keys}
    if not in_grid(baseline_point):
        shards.append(single("point-baseline", "point", baseline_point))
    if not in_grid(startup_point) and startup_point != baseline_point:
        shards.append(single("point-startup", "point", startup_point))
    if probe:
        shards.append(single("probe", "probe", _probe_point(axes)))
    return shards


def _rounds(points: int, parallelism: int) -> int:
    """Evaluation rounds of a Run: the first candidate alone, then batches."""

    return 1 + math.ceil((points - 1) / parallelism)


def _splits(sizes: Sequence[int], max_shards: int) -> Iterable[tuple[int, ...]]:
    """Every per-knob split count whose product is at most ``max_shards``."""

    def walk(index: int, budget: int) -> Iterable[tuple[int, ...]]:
        if index == len(sizes):
            yield ()
            return
        for split in range(1, min(sizes[index], budget) + 1):
            for rest in walk(index + 1, budget // split):
                yield (split, *rest)

    return walk(0, max_shards)


def _chunks(values: tuple[float, ...], count: int) -> list[tuple[float, ...]]:
    """``values`` cut into ``count`` contiguous chunks whose sizes differ by <= 1."""

    base, extra = divmod(len(values), count)
    chunks: list[tuple[float, ...]] = []
    start = 0
    for index in range(count):
        end = start + base + (1 if index < extra else 0)
        chunks.append(values[start:end])
        start = end
    return chunks


def _product(choices: Mapping[str, tuple[float, ...]]) -> tuple[dict[str, float], ...]:
    keys = sorted(choices)
    return tuple(
        dict(zip(keys, values))
        for values in itertools.product(*(choices[key] for key in keys))
    )


def _probe_point(axes: Mapping[str, tuple[float, ...]]) -> dict[str, float]:
    """The fixed CRN probe point: each knob's middle value; a positive
    temperature when the grid has one, so the probe exercises sampling."""

    point: dict[str, float] = {}
    for key, values in axes.items():
        pool = values
        if key == _TEMPERATURE:
            positive = tuple(value for value in values if value > 0)
            pool = positive or values
        point[key] = pool[len(pool) // 2]
    return point


def need_seconds(shard: Shard, cfg: DgdrExecutorConfig, *, needs_cut: bool) -> float:
    """Wall time a shard needs from now to finish (design §4.1).

    ``(cut_budget if needs_cut) + job_startup + (1 + ceil((n - 1) /
    parallelism)) * candidate_timeout``: AISimulate evaluates the first
    candidate alone, then ``parallelism`` at a time.
    """

    rounds = _rounds(shard.size, cfg.run.parallelism)
    return (
        (cfg.cut_budget_seconds if needs_cut else 0.0)
        + cfg.job_startup_seconds
        + rounds * cfg.run.candidate_timeout_seconds
    )


# ----------------------------------------------------------------- Request


def spec_hash(spec: Mapping[str, Any]) -> str:
    """First 16 hex digits of the SHA-256 of the canonical JSON of ``spec``."""

    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def window_claim_path(cfg: DgdrExecutorConfig, dgd: DgdRef, window_end_ms: int) -> str:
    """Claim-relative path of a window file: ``<windows_dir>/<dgd>/<end_ms>.jsonl.gz``."""

    return f"{cfg.windows_dir.strip('/')}/{dgd.name}/{int(window_end_ms)}.jsonl.gz"


def request_name(
    dgd: DgdRef, window_end_ms: int, shard_id: str, digest: str, attempt: int
) -> str:
    """``<dgd prefix>-rt-<sha256(uid|window|shard|spec|attempt)[:10]>``."""

    key = f"{dgd.uid}|{int(window_end_ms)}|{shard_id}|{digest}|{attempt}"
    suffix = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    prefix = dgd.name[:40].rstrip("-.") or "dgd"
    return f"{prefix}-rt-{suffix}"


def _label_value(value: str) -> str:
    """``value`` when it is a valid label value, else a hashed 63-char form."""

    if len(value) <= 63 and re.fullmatch(
        r"([A-Za-z0-9]([-A-Za-z0-9_.]*[A-Za-z0-9])?)?", value
    ):
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    head = re.sub(r"[^-A-Za-z0-9_.]", "-", value)[:52].strip("-_.")
    return f"{head}-{digest}" if head else digest


def build_request(
    *,
    dgd: DgdRef,
    cfg: DgdrExecutorConfig,
    retune: RetuneConfig,
    sla_ttft_ms: float,
    sla_itl_ms: float,
    window_end_ms: int,
    shard: Shard,
    spec_replicas: Mapping[str, int],
    backend: str,
    attempt: int,
) -> dict:
    """The DGDR v1beta2 Request that replays one shard of a window (§4.5.2).

    The engine is ``cfg.engine`` with every leaf a scalar and each used
    role's ``parallelism.preset[0].replicas`` set from ``spec_replicas``
    (unused roles are deleted with ``null``); the GPU budget is the sum of
    replicas times GPUs per worker (tensor x pipeline x attention_data).
    Router knobs use their AISimulate public names: ``{choices: [...]}``
    for a grid shard, a scalar for a point or probe shard. Raises
    ``ValueError`` for an engine or shard the Request cannot express.
    """

    engine = _pinned_engine(cfg.engine, spec_replicas, backend)
    size = shard.size
    if not 1 <= size <= MAX_SHARD_POINTS:
        raise ValueError(f"shard {shard.id} has {size} points; 1..100 fit a Request")
    spec: dict[str, Any] = {
        "model": {"name": cfg.model},
        "backends": [backend],
        "hardware": {
            "gpu": {"sku": cfg.sku, "budget": _gpu_budget(engine, spec_replicas)}
        },
        "workload": {
            "trace": {
                "format": "dynamo",
                "source": {
                    "pvc": {
                        "claimName": cfg.claim_name,
                        "path": window_claim_path(cfg, dgd, window_end_ms),
                    }
                },
            }
        },
        "objective": {
            "mode": "Optimize",
            "metric": retune.objective,
            "sla": {"ttftMs": float(sla_ttft_ms), "itlMs": float(sla_itl_ms)},
        },
        "recommendation": {"maxCandidates": size},
        "aisimulate": {
            "inputOverrides": {
                "engine": engine,
                "router": _router_section(shard),
                "optimizer": {
                    "algorithm": "random",
                    "max_trials": 2 * size,
                    "parallelism": cfg.run.parallelism,
                    "candidate_timeout_seconds": cfg.run.candidate_timeout_seconds,
                },
                "optimization": {"strict_sla": False},
                "dgdr_run": {"generator": "direct"},
            }
        },
    }
    digest = spec_hash(spec)
    return {
        "apiVersion": f"{DGDR_GROUP}/{DGDR_VERSION}",
        "kind": "DynamoGraphDeploymentRequest",
        "metadata": {
            "name": request_name(dgd, window_end_ms, shard.id, digest, attempt),
            "namespace": dgd.namespace,
            "labels": {
                LABEL_DGD: _label_value(dgd.name),
                LABEL_DGD_UID: dgd.uid,
                LABEL_WINDOW: str(int(window_end_ms)),
                LABEL_SHARD: shard.id,
                LABEL_SPEC: digest,
                LABEL_ATTEMPT: str(attempt),
            },
            "ownerReferences": [
                {
                    "apiVersion": "nvidia.com/v1beta1",
                    "kind": "DynamoGraphDeployment",
                    "name": dgd.name,
                    "uid": dgd.uid,
                    "blockOwnerDeletion": False,
                }
            ],
        },
        "spec": spec,
    }


def _pinned_engine(
    engine: Mapping[str, Any], spec_replicas: Mapping[str, int], backend: str
) -> dict[str, Any]:
    pinned = copy.deepcopy(dict(engine))
    roles = set(spec_replicas)
    if roles not in ({"aggregated"}, {"prefill", "decode"}):
        raise ValueError(
            "spec_replicas must name 'aggregated', or 'prefill' and 'decode'; "
            f"got {sorted(roles)}"
        )
    written = pinned.get("backend")
    if written is None:
        pinned["backend"] = backend
    elif written != backend:
        raise ValueError(
            f"retune.dgdr.engine.backend {written!r} differs from the deployed "
            f"backend {backend!r}"
        )
    workers = pinned.get("workers")
    if not isinstance(workers, dict):
        raise ValueError("retune.dgdr.engine.workers must be a mapping")
    for role in _ROLES:
        if role not in roles:
            workers[role] = None
            continue
        worker = workers.get(role)
        preset = (
            worker.get("parallelism", {}).get("preset")
            if isinstance(worker, dict) and isinstance(worker.get("parallelism"), dict)
            else None
        )
        if (
            not isinstance(preset, list)
            or len(preset) != 1
            or not isinstance(preset[0], dict)
        ):
            raise ValueError(
                f"retune.dgdr.engine.workers.{role}.parallelism.preset must be a "
                "list of exactly one entry"
            )
        replicas = int(spec_replicas[role])
        if replicas < 1:
            raise ValueError(
                f"{role} has {replicas} spec replicas; at least 1 is needed"
            )
        preset[0]["replicas"] = replicas
    return pinned


def _gpu_budget(engine: Mapping[str, Any], spec_replicas: Mapping[str, int]) -> int:
    budget = 0
    for role, replicas in spec_replicas.items():
        entry = engine["workers"][role]["parallelism"]["preset"][0]
        per_worker = 1
        for knob in ("tensor", "pipeline", "attention_data"):
            value = entry.get(knob, 1)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"retune.dgdr.engine.workers.{role}.parallelism.preset[0]."
                    f"{knob} must be a positive integer, got {value!r}"
                )
            per_worker *= value
        budget += int(replicas) * per_worker
    return budget


def _router_section(shard: Shard) -> dict[str, Any]:
    live_knobs = set(DGDR_ROUTER_KNOBS.values())
    if set(shard.choices) != live_knobs:
        raise ValueError(
            f"shard {shard.id} router knobs {sorted(shard.choices)} must be exactly "
            f"{sorted(live_knobs)}, the knobs a DGDR Request can express"
        )
    section: dict[str, Any] = {
        "policy": "kv_router",
        "prefill_load_model": {"type": "none"},
    }
    for public, live in DGDR_ROUTER_KNOBS.items():
        values = [float(value) for value in shard.choices[live]]
        section[public] = {"choices": values} if shard.kind == "grid" else values[0]
    return section


# --------------------------------------------------------- input read-back


def merge_patch(target: Any, patch: Any) -> Any:
    """RFC 7386 JSON merge patch (the operator's lowering); never mutates."""

    if not isinstance(patch, Mapping):
        return copy.deepcopy(patch)
    result = dict(target) if isinstance(target, Mapping) else {}
    for key, value in patch.items():
        if value is None:
            result.pop(key, None)
        else:
            result[key] = merge_patch(result.get(key), value)
    return result


def _get(obj: Any, path: Sequence[str], default: Any = None) -> Any:
    for key in path:
        if not isinstance(obj, Mapping) or key not in obj:
            return default
        obj = obj[key]
    return obj


def _same_json(left: Any, right: Any) -> bool:
    """JSON equality: numbers by value (2 == 2.0), booleans never equal numbers."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _same_json(left[key], right[key]) for key in left
        )
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_json(a, b) for a, b in zip(left, right)
        )
    return type(left) is type(right) and left == right


def missing_fields(stored: Any, submitted: Any, path: str = "") -> list[str]:
    """Submitted leaves the stored object lacks or changed (pruned or rejected).

    ``stored`` may hold more (admission defaults); a submitted ``null`` only
    requires the stored value to be absent or ``null``.
    """

    if isinstance(submitted, Mapping):
        if not isinstance(stored, Mapping):
            return [path or "."]
        problems: list[str] = []
        for key, value in submitted.items():
            child = f"{path}.{key}" if path else key
            if value is None:
                if stored.get(key) is not None:
                    problems.append(child)
                continue
            if key not in stored:
                problems.append(child)
                continue
            problems.extend(missing_fields(stored[key], value, child))
        return problems
    return [] if _same_json(stored, submitted) else [path or "."]


def input_mismatches(
    spec: Mapping[str, Any], effective: Mapping[str, Any]
) -> list[str]:
    """How ``Run.status.aisimulate.input`` differs from what ``spec`` asks for.

    The expected input follows the operator's lowering at #17483: the
    public role underlay, ``spec.model``/``backends``/``hardware`` into
    ``engine``, then ``inputOverrides`` as a merge patch. ``engine``,
    ``router`` and ``optimizer`` must match exactly; the objective, SLA,
    budget, trace source and the direct generator are checked by value.
    """

    overrides = _get(spec, ("aisimulate", "inputOverrides"), {}) or {}
    typed_engine: dict[str, Any] = {
        "workers": {role: {} for role in _ROLES},
        "model": _get(spec, ("model", "name")),
        "backend": {"choices": list(spec.get("backends") or [])},
        "hardware": _get(spec, ("hardware", "gpu", "sku")),
    }
    expected_trace = _get(spec, ("workload", "trace", "source", "pvc", "path"), "")
    mounted = TRACE_MOUNT_PATH + (
        "/trace.jsonl.gz" if str(expected_trace).endswith(".gz") else "/trace.jsonl"
    )
    sla = _get(spec, ("objective", "sla"), {}) or {}
    checks: list[tuple[tuple[str, ...], Any]] = [
        (("engine",), merge_patch(typed_engine, overrides.get("engine") or {})),
        (("router",), merge_patch({}, overrides.get("router") or {})),
        (("optimizer",), merge_patch({}, overrides.get("optimizer") or {})),
        (
            ("optimization", "target"),
            _METRIC_TARGETS.get(_get(spec, ("objective", "metric"), ""), _ABSENT),
        ),
        (
            ("optimization", "strict_sla"),
            _get(overrides, ("optimization", "strict_sla")),
        ),
        (
            ("optimization", "constraints", "max_candidate_gpus"),
            _get(spec, ("hardware", "gpu", "budget")),
        ),
        (("evaluation", "sla", "ttft_ms"), sla.get("ttftMs", _ABSENT)),
        (("evaluation", "sla", "itl_ms"), sla.get("itlMs", _ABSENT)),
        (("evaluation", "sla", "e2e_ms"), sla.get("e2eMs", _ABSENT)),
        (
            ("traffic", "source"),
            {"type": "trace", "format": "dynamo", "paths": [mounted]},
        ),
        (("traffic", "load"), {"type": "trace_timestamps"}),
        (("dgdr_run", "generator"), _get(overrides, ("dgdr_run", "generator"))),
        (
            ("dgdr_run", "max_candidates"),
            _get(spec, ("recommendation", "maxCandidates")),
        ),
    ]
    # Normalize like the API server (JSON numbers, no tuples).
    effective = json.loads(json.dumps(effective))
    problems: list[str] = []
    for path, expected in checks:
        expected = expected if expected is _ABSENT else json.loads(json.dumps(expected))
        actual = _get(effective, path, _ABSENT)
        if expected is _ABSENT:
            if actual is not _ABSENT and actual is not None:
                problems.append(f"{'.'.join(path)}: expected unset, got {actual!r}")
        elif actual is _ABSENT or not _same_json(actual, expected):
            shown = "unset" if actual is _ABSENT else repr(actual)
            problems.append(f"{'.'.join(path)}: expected {expected!r}, got {shown}")
    return problems


def _version_tuple(text: str | None) -> tuple[int, ...] | None:
    if not text:
        return None
    found = re.match(r"^v?(\d+)\.(\d+)(?:\.(\d+))?", text.strip())
    if found is None:
        return None
    return tuple(int(part) for part in found.groups(default="0"))


def crn_unsupported(
    runtime_version: str | None, runtime_image: str | None, minimum: str | None
) -> bool:
    """Whether a parseable runtime version is below ``minimum`` (None: unknown)."""

    floor = _version_tuple(minimum)
    if floor is None:
        return False
    version = _version_tuple(runtime_version)
    if version is None and runtime_image:
        # registry/repo:tag@digest -> tag
        last = runtime_image.split("@", 1)[0].rsplit("/", 1)[-1]
        if ":" in last:
            version = _version_tuple(last.split(":", 1)[1])
    return version is not None and version < floor


# ---------------------------------------------------------------- executor


@dataclass(frozen=True)
class DgdrContext:
    """Everything ``run_dgdr_sweep`` needs besides the ``SweepRequest``.

    ``backend`` is the deployed backend (``vllm``, ``sglang``, ``trtllm``).
    ``is_leader`` is checked before every mutating call. ``now``/``sleep``
    are wall-clock (``deadline_s`` is epoch seconds). ``crn_min_version`` is
    the oldest runtime that replays traces with common random numbers.
    """

    dgd: DgdRef
    cfg: DgdrExecutorConfig
    retune: RetuneConfig
    backend: str
    kube: RetuneKube
    registry: SweepRegistry
    cut_window: WindowCutter
    is_leader: Callable[[], bool]
    now: Callable[[], float] = time.time
    sleep: Callable[[float], None] = time.sleep
    poll_interval_s: float = 5.0
    crn_min_version: Optional[str] = CRN_MIN_VERSION


def run_dgdr_sweep(
    request: SweepRequest,
    *,
    cancel: threading.Event,
    abort: Optional[threading.Event],
    context: DgdrContext,
) -> SweepResult:
    """Sweep one window through DGDR v1beta2 Requests; never raises.

    Scores are keyed by ``candidate.key()`` for every candidate of
    ``request`` (live router names, no planner part); a candidate no Run
    published scores ``None``. See the module docstring for the protocol.
    """

    started = context.now()
    try:
        return _DgdrSweep(request, cancel, abort, context, started).run()
    except Exception as error:  # defensive: the executor must never raise
        logger.exception("retune DGDR sweep of window %s failed", request.window_index)
        return SweepResult(
            status="failed",
            wall_s=max(0.0, context.now() - started),
            error=f"{type(error).__name__}: {error}",
            failure_reason="internal_error",
            details={},
        )


class _Interrupted(Exception):
    """Shutdown or loss of leadership: return without a decision."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _WindowFailed(Exception):
    """The window ends without scores (``failed``, ``timeout`` or ``empty``)."""

    def __init__(
        self, status: SweepStatus, reason: Optional[str], message: str
    ) -> None:
        super().__init__(message)
        self.status = status
        self.reason = reason
        self.message = message


class _EitherEvent(threading.Event):
    """An event that reads as set once any of its sources is set.

    Hands the window cutter one ``cancel`` for both shutdown and abort.
    """

    def __init__(self, *sources: Optional[threading.Event]) -> None:
        super().__init__()
        self._sources = [source for source in sources if source is not None]

    def is_set(self) -> bool:
        return super().is_set() or any(source.is_set() for source in self._sources)

    isSet = is_set

    def wait(self, timeout: Optional[float] = None) -> bool:
        end = None if timeout is None else time.monotonic() + timeout
        while not self.is_set():
            remaining = None if end is None else end - time.monotonic()
            if remaining is not None and remaining <= 0:
                return False
            super().wait(0.05 if remaining is None else min(0.05, remaining))
        return True


ShardState = Literal["create", "unknown", "wait", "done", "missing"]


@dataclass
class _ShardRun:
    """Progress of one shard's Request within this sweep."""

    shard: Shard
    critical: bool
    spec: dict
    digest: str
    attempt: int = 1
    state: ShardState = "create"
    name: Optional[str] = None
    adopted: bool = False
    # This sweep sent a create for the current attempt that may have landed;
    # False for an attempt a previous leader created.
    created: bool = False
    reason: Optional[str] = None
    message: Optional[str] = None
    run_name: Optional[str] = None
    run_uid: Optional[str] = None
    checked_run_uid: Optional[str] = None
    unaccepted_since: Optional[float] = None
    pending_since: Optional[float] = None
    run_pending: bool = False
    rows: Optional[dict] = None
    published: int = 0

    def require_name(self) -> str:
        """The current attempt's Request name.

        It is set once a create is sent or a Request is adopted, so a shard in
        the ``unknown`` or ``wait`` state, where it is read, always has it.
        """

        if self.name is None:
            raise RuntimeError(
                f"shard {self.shard.id} has no Request name in state {self.state}"
            )
        return self.name

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.shard.kind,
            "points": self.shard.size,
            "critical": self.critical,
            "request": self.name,
            "attempt": self.attempt,
            "adopted": self.adopted,
            "run": self.run_name,
            "state": self.state,
            "reason": self.reason,
            "message": self.message,
            "published": self.published,
            "run_pending": self.run_pending,
        }


@dataclass
class _Scored:
    score: Optional[float]
    entry: dict[str, Any] = field(default_factory=dict)


class _DgdrSweep:
    def __init__(
        self,
        request: SweepRequest,
        cancel: threading.Event,
        abort: Optional[threading.Event],
        context: DgdrContext,
        started: float,
    ) -> None:
        self.request = request
        self.cancel = cancel
        self.abort = abort
        self.ctx = context
        self.cfg = context.cfg
        self.started = started
        deadline = getattr(request, "deadline_s", None)
        self.deadline = (
            float(deadline) if deadline is not None else started + request.timeout_s
        )
        self.window_end_ms = int(round(request.window_end_ms))
        # Registry references this sweep took; each is released at the end.
        self._protections = 0
        self.claim_path = window_claim_path(self.cfg, context.dgd, self.window_end_ms)
        self.local_path = os.path.join(self.cfg.mount_path, self.claim_path)
        self.shards: dict[str, _ShardRun] = {}
        self.cut_done = False
        self.num_requests = 0
        self.row_counts: dict[str, int] = {}
        self.details: dict[str, Any] = {
            "window": {"path": self.claim_path, "cut": False},
            "deadline_s": self.deadline,
            "generation": getattr(request, "generation", 0),
            "replicas_verified": getattr(request, "replicas_verified", True),
        }
        self.k8s_errors = 0
        self.runtime: dict[str, Any] = {}

    # ------------------------------------------------------------ driver

    def run(self) -> SweepResult:
        try:
            self._prepare()
            self._check_interrupt()
            self._adopt()
            self._create_all()
            self._poll()
            return self._harvest()
        except _Interrupted as stop:
            logger.info(
                "retune window %s sweep stopped (%s); its Requests stay for the "
                "next leader",
                self.request.window_index,
                stop.reason,
            )
            return self._result(
                "failed",
                failure_reason=stop.reason,
                error=f"sweep stopped: {stop.reason}",
            )
        except _WindowFailed as failed:
            self._abandon_in_flight()
            log = logger.info if failed.status == "empty" else logger.warning
            log(
                "retune window %s %s (%s): %s",
                self.request.window_index,
                failed.status,
                failed.reason,
                failed.message,
            )
            return self._result(
                failed.status,
                failure_reason=failed.reason,
                error=failed.message if failed.status != "empty" else None,
            )
        finally:
            # One release per protect: the registry counts references per
            # window, and another generation's sweep may hold its own.
            try:
                for _ in range(self._protections):
                    self.ctx.registry.release(self.window_end_ms)
            except Exception:  # pragma: no cover - registry is in-process
                logger.exception("retune registry release failed")
            self._protections = 0

    def _protect(self, names: list[str]) -> None:
        self.ctx.registry.protect(self.window_end_ms, names)
        self._protections += 1

    def _result(
        self,
        status: SweepStatus,
        *,
        failure_reason: Optional[str] = None,
        error: Optional[str] = None,
        scores: Optional[dict[str, Optional[float]]] = None,
        metrics: Optional[dict[str, dict[str, Any]]] = None,
    ) -> SweepResult:
        details = dict(self.details)
        details["shards"] = {key: run.summary() for key, run in self.shards.items()}
        details["k8s_errors"] = self.k8s_errors
        details.update(self.runtime)
        return SweepResult(
            status=status,
            num_requests=self.num_requests,
            row_counts=dict(self.row_counts),
            scores=scores or {},
            metrics=metrics or {},
            wall_s=max(0.0, self._now() - self.started),
            error=error,
            failure_reason=failure_reason,
            details=details,
        )

    # ----------------------------------------------------------- helpers

    def _now(self) -> float:
        return self.ctx.now()

    def _timeout(self) -> float:
        return max(_MIN_K8S_TIMEOUT_S, min(_K8S_TIMEOUT_S, self.deadline - self._now()))

    def _check_interrupt(self) -> None:
        if self.cancel.is_set():
            raise _Interrupted("shutdown")
        if self.abort is not None and self.abort.is_set():
            raise _Interrupted("lost_leadership")

    def _check_mutation(self) -> None:
        self._check_interrupt()
        if not self.ctx.is_leader():
            raise _Interrupted("lost_leadership")

    def _fail(
        self, reason: Optional[str], message: str, status: SweepStatus = "failed"
    ) -> NoReturn:
        raise _WindowFailed(status, reason, message)

    def _retrying(self, what: str, call: Callable[[], Any]) -> Any:
        """``call`` until it succeeds; a permanent error or the deadline fails."""

        while True:
            self._check_interrupt()
            try:
                return call()
            except KubeError as error:
                self.k8s_errors += 1
                if _permanent(error):
                    self._fail("k8s_error", f"{what}: {error}")
                if self._now() + self.ctx.poll_interval_s >= self.deadline:
                    self._fail("k8s_error", f"{what}: {error}", "timeout")
                logger.warning("retune %s failed, retrying: %s", what, error)
                self.ctx.sleep(self.ctx.poll_interval_s)

    def _read_failed(self, what: str, error: KubeError) -> None:
        """A GET failed: a permanent error fails the window; else poll again."""

        self.k8s_errors += 1
        if _permanent(error):
            self._fail("k8s_error", f"{what}: {error}")
        logger.warning("retune %s failed: %s", what, error)

    def _enqueue_delete(self, names: Iterable[Optional[str]]) -> None:
        present = [name for name in names if name]
        if present:
            self.ctx.registry.enqueue_delete(present)

    def _abandon_in_flight(self) -> None:
        """Queue this window's unfinished Requests for deletion."""

        self._enqueue_delete(
            run.name
            for run in self.shards.values()
            if run.state in ("wait", "unknown") and run.name
        )

    # ------------------------------------------------------------- setup

    def _prepare(self) -> None:
        request = self.request
        domain = getattr(request, "router_domain", None)
        if not domain:
            self._fail("invalid_request", "the sweep request has no router domain")
        if any(candidate.planner for candidate in request.candidates):
            self._fail(
                "invalid_request",
                "a DGDR sweep searches router parameters only (Scenario 1)",
            )
        keys = set(domain)
        startup = self._startup_point(domain)
        try:
            shards = plan_shards(
                domain,
                baseline=request.baseline.router,
                startup=startup,
                max_shards=self.cfg.run.max_shards,
                parallelism=self.cfg.run.parallelism,
            )
        except (KeyError, ValueError) as error:
            self._fail("invalid_domain", str(error))
        replicas = getattr(request, "spec_replicas", None) or request.replicas
        self.details["replicas"] = dict(replicas)
        probe = next((shard for shard in shards if shard.kind == "probe"), None)
        for shard in shards:
            try:
                body = build_request(
                    dgd=self.ctx.dgd,
                    cfg=self.cfg,
                    retune=self.ctx.retune,
                    sla_ttft_ms=request.sla_ttft_ms,
                    sla_itl_ms=request.sla_itl_ms,
                    window_end_ms=self.window_end_ms,
                    shard=shard,
                    spec_replicas=replicas,
                    backend=self.ctx.backend,
                    attempt=1,
                )
            except (KeyError, TypeError, ValueError) as error:
                self._fail("invalid_engine", f"shard {shard.id}: {error}")
            # The baseline must be scored, and the probe's grid twin is what
            # the probe is compared with.
            critical = (
                shard.kind == "probe"
                or shard.contains(request.baseline.router)
                or (probe is not None and shard.contains(probe.points[0]))
            )
            self.shards[shard.id] = _ShardRun(
                shard=shard,
                critical=critical,
                spec=body["spec"],
                digest=body["metadata"]["labels"][LABEL_SPEC],
            )
        uncovered = [
            candidate.router
            for candidate in request.candidates
            if set(candidate.router) != keys
            or not any(
                run.shard.kind != "probe" and run.shard.contains(candidate.router)
                for run in self.shards.values()
            )
        ]
        if uncovered:
            self.details["uncovered_candidates"] = uncovered[:10]

    def _startup_point(self, domain: Mapping[str, Sequence[float]]) -> dict[str, float]:
        """The off-grid candidate other than the baseline, else the baseline.

        The controller's candidates are the grid plus the startup point and
        the baseline, so an off-grid candidate is one of those two.
        """

        baseline = self.request.baseline
        for candidate in self.request.candidates:
            router = candidate.router
            if candidate == baseline or set(router) != set(domain):
                continue
            if not all(float(router[key]) in domain[key] for key in domain):
                return dict(router)
        return dict(baseline.router)

    def _build(self, run: _ShardRun) -> dict:
        body = build_request(
            dgd=self.ctx.dgd,
            cfg=self.cfg,
            retune=self.ctx.retune,
            sla_ttft_ms=self.request.sla_ttft_ms,
            sla_itl_ms=self.request.sla_itl_ms,
            window_end_ms=self.window_end_ms,
            shard=run.shard,
            spec_replicas=self.details["replicas"],
            backend=self.ctx.backend,
            attempt=run.attempt,
        )
        rows = {"num_requests": self.num_requests, "row_counts": self.row_counts}
        body["metadata"]["annotations"] = {
            ROWS_ANNOTATION: json.dumps(rows, sort_keys=True, separators=(",", ":"))
        }
        return body

    # ------------------------------------------------------------- adopt

    def _adopt(self) -> None:
        """Adopt the latest attempt of each shard; queue the rest for deletion.

        Each adopted attempt is observed once here: a Succeeded one needs no
        cut, and a failed one is replaced while there is time (``_run_failed``).
        """

        selector = (
            f"{LABEL_DGD_UID}={self.ctx.dgd.uid},{LABEL_WINDOW}={self.window_end_ms}"
        )
        existing = self._retrying(
            "list window Requests",
            lambda: self.ctx.kube.list_requests(selector, timeout_s=self._timeout()),
        )
        stale: list[Optional[str]] = []
        latest: dict[str, dict] = {}
        for obj in existing:
            metadata = obj.get("metadata") or {}
            labels = metadata.get("labels") or {}
            name = metadata.get("name")
            run = self.shards.get(labels.get(LABEL_SHARD, ""))
            if run is None or labels.get(LABEL_SPEC) != run.digest:
                stale.append(name)
                continue
            attempt = _int(labels.get(LABEL_ATTEMPT), 0)
            if metadata.get("deletionTimestamp"):
                # Being deleted: never adopt it, and never reuse its attempt.
                run.attempt = max(run.attempt, attempt + 1)
                continue
            current = latest.get(run.shard.id)
            if current is None or attempt > _int(
                current["metadata"]["labels"].get(LABEL_ATTEMPT), 0
            ):
                if current is not None:
                    stale.append(current["metadata"]["name"])
                latest[run.shard.id] = obj
            else:
                stale.append(name)
        self._enqueue_delete(stale)
        for shard_id, obj in latest.items():
            run = self.shards[shard_id]
            metadata = obj["metadata"]
            run.name = metadata["name"]
            run.attempt = max(
                run.attempt, _int(metadata["labels"].get(LABEL_ATTEMPT), 1)
            )
            run.adopted = True
            run.state = "wait"
            run.rows = _rows(metadata.get("annotations"))
            logger.info(
                "retune window %s adopts Request %s for shard %s",
                self.request.window_index,
                run.name,
                shard_id,
            )
        if latest:
            self._protect([obj["metadata"]["name"] for obj in latest.values()])
            # Classify the adopted Requests: a window whose shards all
            # succeeded is harvested without cutting it again.
            for run in self.shards.values():
                if run.state == "wait":
                    self._observe(run)

    # ------------------------------------------------------------ create

    def _create_all(self) -> None:
        pending = [run for run in self._ordered() if run.state == "create"]
        if not pending:
            rows = next(
                (run.rows for run in self.shards.values() if run.rows is not None),
                None,
            )
            if rows is not None:
                self.num_requests = _int(rows.get("num_requests"), 0)
                self.row_counts = dict(rows.get("row_counts") or {})
            return
        # Gate every shard before the cut, so a critical shard that cannot
        # finish fails the window without spending the cut budget.
        for run in pending:
            self._too_late(run)
        for run in pending:
            if run.state == "create":
                self._create(run)

    def _ordered(self) -> list[_ShardRun]:
        """Critical shards first, so they get the time that is left."""

        return sorted(self.shards.values(), key=lambda run: not run.critical)

    def _too_late(self, run: _ShardRun) -> bool:
        """Apply the launch gate; True when the shard was given up."""

        need = need_seconds(run.shard, self.cfg, needs_cut=not self.cut_done)
        left = self.deadline - self._now()
        if left >= need:
            return False
        message = (
            f"shard {run.shard.id} needs {need:.0f}s but {max(left, 0.0):.0f}s are "
            "left before the deadline"
        )
        if run.critical:
            self._fail("launch_too_late", message, "timeout")
        run.state = "missing"
        run.reason = "launch_too_late"
        run.message = message
        logger.warning("retune window %s: %s", self.request.window_index, message)
        return True

    def _create(self, run: _ShardRun) -> None:
        self._check_mutation()
        if self._too_late(run):
            return
        if not self.cut_done:
            self._cut()
            if self._too_late(run):
                return
            self._check_mutation()
        body = self._build(run)
        name = body["metadata"]["name"]
        run.name = name
        run.rows = None
        self._protect([name])
        try:
            created = self.ctx.kube.create_request(body, timeout_s=self._timeout())
        except KubeAlreadyExists:
            self._adopt_existing(run)
            return
        except KubeError as error:
            self.k8s_errors += 1
            if _admission_denied(error):
                self._fail("request_rejected", f"Request {name} rejected: {error}")
            if isinstance(error, KubeInvalid):
                self._fail("schema_unsupported", f"Request {name} rejected: {error}")
            if _permanent(error):
                self._fail("k8s_error", f"create Request {name}: {error}")
            # The create may have landed; look it up on the next poll.
            run.created = True
            run.state = "unknown"
            run.message = f"create: {error}"
            logger.warning("retune create of Request %s failed: %s", name, error)
            return
        run.created = True
        dropped = missing_fields(created.get("spec"), body["spec"], "spec")
        labels = (created.get("metadata") or {}).get("labels") or {}
        dropped += [
            f"metadata.labels.{key}"
            for key, value in body["metadata"]["labels"].items()
            if labels.get(key) != value
        ]
        if dropped:
            self._enqueue_delete([name])
            run.state = "missing"
            self._fail(
                "schema_unsupported",
                f"the API server dropped or changed {sorted(dropped)} of Request {name}",
            )
        run.state = "wait"
        run.unaccepted_since = self._now()
        logger.info(
            "retune window %s created Request %s (shard %s, %d points, attempt %d)",
            self.request.window_index,
            name,
            run.shard.id,
            run.shard.size,
            run.attempt,
        )

    def _adopt_existing(self, run: _ShardRun) -> None:
        """The name exists (an earlier create landed): adopt it if it is ours."""

        existing = self._retrying(
            f"get Request {run.name}",
            lambda: self.ctx.kube.get_request(
                run.require_name(), timeout_s=self._timeout()
            ),
        )
        if existing is None:
            # Deleted in between; create it again on the next poll.
            run.state = "create"
            return
        labels = (existing.get("metadata") or {}).get("labels") or {}
        if (
            labels.get(LABEL_SPEC) == run.digest
            and labels.get(LABEL_DGD_UID) == self.ctx.dgd.uid
            and labels.get(LABEL_SHARD) == run.shard.id
        ):
            run.state = "wait"
            run.adopted = True
            run.unaccepted_since = self._now()
            return
        self._shard_failed(
            run, "request_conflict", f"Request {run.name} exists with other labels"
        )

    # ------------------------------------------------------------ window

    def _cut(self) -> None:
        """Cut the window file; an empty window ends the sweep as ``empty``."""

        self._check_interrupt()
        segments_root = (
            self.request.trace_paths[0]
            if self.request.trace_paths
            else os.path.join(
                self.cfg.mount_path, self.cfg.segments_dir, self.ctx.dgd.name
            )
        )
        budget = min(self.cfg.cut_budget_seconds, self.deadline - self._now())
        if budget <= 0:
            self._fail("launch_too_late", "no time left to cut the window", "timeout")
        # WindowCutter takes the deadline as absolute wall-clock seconds.
        cut_deadline_s = self._now() + budget
        cancel = _EitherEvent(self.cancel, self.abort)
        try:
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            result: WindowCutResult = self.ctx.cut_window(
                segments_root,
                int(round(self.request.window_start_ms)),
                self.window_end_ms,
                int(round(self.request.as_of_ms)),
                self.local_path,
                max_rows=self.cfg.max_window_rows,
                deadline_s=cut_deadline_s,
                cancel=cancel,
            )
        except Exception as error:
            # A cut cancelled by shutdown or abort reports which one it was.
            self._check_interrupt()
            reason = getattr(error, "reason", None)
            if reason not in (
                "window_cut_timeout",
                "window_too_large",
                "trace_corrupt",
                "window_cut_failed",
            ):
                reason = "window_cut_failed"
            self._fail(reason, f"window cut: {type(error).__name__}: {error}")
        self.cut_done = True
        self.num_requests = int(result.num_requests)
        self.row_counts = dict(result.row_counts)
        self.details["window"].update(
            cut=True,
            num_requests=result.num_requests,
            reused=result.reused,
            dropped_members=result.dropped_members,
            dropped_bytes=result.dropped_bytes,
            trace_block_size=result.trace_block_size,
        )
        if result.path is None or result.num_requests == 0:
            self._fail(None, "empty window", "empty")

    # -------------------------------------------------------------- wait

    def _poll(self) -> None:
        while True:
            self._check_interrupt()
            active = [
                run
                for run in self._ordered()
                if run.state in ("create", "unknown", "wait")
            ]
            if not active:
                return
            if self._now() >= self.deadline:
                pending = [run.shard.id for run in active if run.run_pending]
                self._fail(
                    "run_pending" if pending else "timeout",
                    "the deadline passed with shards "
                    f"{sorted(run.shard.id for run in active)} unfinished"
                    + (f" (Runs pending: {sorted(pending)})" if pending else ""),
                    "timeout",
                )
            for run in active:
                self._check_interrupt()
                if run.state == "create":
                    self._create(run)
                elif run.state == "unknown":
                    self._resolve_unknown(run)
                else:
                    self._observe(run)
            if any(
                run.state in ("create", "unknown", "wait")
                for run in self.shards.values()
            ):
                left = self.deadline - self._now()
                self.ctx.sleep(max(0.0, min(self.ctx.poll_interval_s, left)))

    def _resolve_unknown(self, run: _ShardRun) -> None:
        """After a create whose outcome is unknown: adopt it or create again."""

        try:
            existing = self.ctx.kube.get_request(
                run.require_name(), timeout_s=self._timeout()
            )
        except KubeError as error:
            self._read_failed(f"get Request {run.name}", error)
            return
        if existing is None:
            run.state = "create"
            return
        self._adopt_existing(run)

    def _shard_failed(self, run: _ShardRun, reason: str, message: str) -> None:
        """A shard ended without scores; its Request (if any) is kept for audit.

        A critical shard fails the window; any other leaves its points unscored.
        """

        run.state = "missing"
        run.reason = reason
        run.message = message
        if run.critical:
            self._fail(reason, f"shard {run.shard.id}: {message}")
        logger.warning(
            "retune window %s shard %s %s: %s; its points are unscored",
            self.request.window_index,
            run.shard.id,
            reason,
            message,
        )

    def _observe(self, run: _ShardRun) -> None:
        """One look at a shard's Request and Run (the §4.5.3 state machine)."""

        kube = self.ctx.kube
        try:
            obj = kube.get_request(run.require_name(), timeout_s=self._timeout())
        except KubeError as error:
            self._read_failed(f"get Request {run.name}", error)
            return
        if obj is None or (obj.get("metadata") or {}).get("deletionTimestamp"):
            self._shard_failed(run, "request_deleted", f"Request {run.name} is gone")
            return
        if run.rows is None:
            run.rows = _rows((obj.get("metadata") or {}).get("annotations"))
        generation = _int((obj.get("metadata") or {}).get("generation"), 1)
        accepted = _condition(obj, "Accepted")
        if accepted is None or _int(accepted.get("observedGeneration"), 0) < generation:
            if run.unaccepted_since is None:
                run.unaccepted_since = self._now()
            elif self._now() - run.unaccepted_since >= ACCEPT_TIMEOUT_S:
                self._fail(
                    "operator_not_reconciling",
                    f"Request {run.name} has no Accepted condition after "
                    f"{ACCEPT_TIMEOUT_S:.0f}s",
                )
            return
        run.unaccepted_since = None
        if accepted.get("status") != "True":
            if accepted.get("reason") == "InvalidSpec":
                self._fail(
                    "invalid_spec",
                    f"Request {run.name}: {accepted.get('message', '')}",
                )
            # RunCreating, RunNameConflict, CancelingSupersededRun: wait.
            return
        try:
            found = self._find_run(obj)
        except KubeError as error:
            self._read_failed(f"get the Run of Request {run.name}", error)
            return
        if found is None:
            return  # created, not yet visible
        run.run_name = found["metadata"]["name"]
        run_uid = found["metadata"].get("uid")
        status = found.get("status") or {}
        completed = _condition(found, "Completed") or {}
        terminal = completed.get("status") == "True"
        succeeded = terminal and completed.get("reason") == "SearchSucceeded"
        outcome = (
            f"Run {run.run_name} {completed.get('reason')}: "
            f"{completed.get('message', '')}"
        )
        if not terminal:
            self._track_pending(run, status)
        if run.checked_run_uid != run_uid:
            effective = _get(status, ("aisimulate", "input"))
            if not isinstance(effective, Mapping):
                if not terminal:
                    return  # the input is recorded together with jobRef
                if not succeeded:
                    self._run_failed(run, outcome)
                    return
                self._input_mismatch(run, ["status.aisimulate.input is missing"])
                return
            if not self._check_input(run, effective):
                return
            run.checked_run_uid = run_uid
        if succeeded:
            run.state = "done"
            run.run_uid = run_uid
        elif terminal:
            self._run_failed(run, outcome)

    def _track_pending(self, run: _ShardRun, status: Mapping[str, Any]) -> None:
        """Flag a live Run that has had no Job for a long time (``run_pending``).

        The operator writes ``status.aisimulate.input`` and ``jobRef`` in one
        status patch once the Job exists; until then the Run has no status
        at all, so this runs before the input is read back.
        """

        if status.get("jobRef") or status.get("startTime"):
            run.pending_since = None
            run.run_pending = False
            return
        now = self._now()
        if run.pending_since is None:
            run.pending_since = now
        elif now - run.pending_since >= max(
            self.cfg.job_startup_seconds, self.ctx.poll_interval_s
        ):
            if not run.run_pending:
                logger.warning(
                    "retune Run %s has no Job after %.0fs (run_pending)",
                    run.run_name,
                    now - run.pending_since,
                )
            run.run_pending = True

    def _run_failed(self, run: _ShardRun, outcome: str) -> None:
        """A Run ended without success.

        An attempt a previous leader created is replaced while there is time
        (design §4.5.3 adopts only a live or Succeeded attempt); otherwise
        the shard fails.
        """

        if run.adopted and not run.created:
            name = run.name
            if self._replace_attempt(run, outcome):
                logger.warning(
                    "retune window %s replaces adopted Request %s of shard %s: %s",
                    self.request.window_index,
                    name,
                    run.shard.id,
                    outcome,
                )
                return
        self._shard_failed(run, "run_failed", outcome)

    def _find_run(self, obj: Mapping[str, Any]) -> Optional[dict]:
        """The Run of the Request's current generation, by the operator's refs."""

        metadata = obj.get("metadata") or {}
        status = obj.get("status") or {}
        generation = _int(metadata.get("generation"), 1)
        names: list[str] = []
        active = (status.get("activeRunRef") or {}).get("name")
        if active:
            names.append(active)
        names += [
            ref.get("name")
            for ref in status.get("recentRunRefs") or ()
            if ref.get("name")
        ]
        names.append(f"{metadata.get('name')}-{generation}")
        uid = metadata.get("uid")
        for name in dict.fromkeys(names):
            found = self.ctx.kube.get_run(name, timeout_s=self._timeout())
            if found is None:
                continue
            run_meta = found.get("metadata") or {}
            annotations = run_meta.get("annotations") or {}
            owners = run_meta.get("ownerReferences") or ()
            if annotations.get(REQUEST_GENERATION_ANNOTATION) == str(
                generation
            ) and any(owner.get("uid") == uid for owner in owners):
                return found
        return None

    def _check_input(self, run: _ShardRun, effective: Mapping[str, Any]) -> bool:
        """Read back the Run's lowered input; False when the shard moved on."""

        dgdr_run = effective.get("dgdr_run") or {}
        image = dgdr_run.get("runtime_image")
        version = dgdr_run.get("runtime_version_override")
        if image or version:
            self.runtime = {"runtime_image": image, "runtime_version": version}
        problems = input_mismatches(run.spec, effective)
        if problems:
            self._input_mismatch(run, problems)
            return False
        if crn_unsupported(version, image, self.ctx.crn_min_version):
            self._fail(
                "crn_unsupported",
                f"Run {run.run_name} runtime {version or image} predates CRN "
                f"trace replay ({self.ctx.crn_min_version})",
            )
        return True

    def _input_mismatch(self, run: _ShardRun, problems: list[str]) -> None:
        """An adopted attempt is replaced while there is time; else it fails."""

        message = f"Run {run.run_name} input differs: {'; '.join(problems[:5])}"
        logger.warning("retune window %s: %s", self.request.window_index, message)
        if run.adopted and self._replace_attempt(run, message):
            return
        self._enqueue_delete([run.name])
        run.state = "missing"
        if run.critical:
            self._fail("input_mismatch", message)
        run.reason = "input_mismatch"
        run.message = message

    def _replace_attempt(self, run: _ShardRun, message: str) -> bool:
        """Queue the attempt for deletion and create attempt+1, if it fits.

        False (nothing changed) when ``D_k - now < need(shard)``.
        """

        need = need_seconds(run.shard, self.cfg, needs_cut=not self.cut_done)
        if self.deadline - self._now() < need:
            return False
        self._enqueue_delete([run.name])
        run.attempt += 1
        run.state = "create"
        run.adopted = False
        run.created = False
        run.name = None
        run.run_name = None
        run.checked_run_uid = None
        run.unaccepted_since = None
        run.pending_since = None
        run.run_pending = False
        run.message = message
        return True

    # ----------------------------------------------------------- harvest

    def _harvest(self) -> SweepResult:
        request = self.request
        domain_keys = set(getattr(request, "router_domain", None) or {})
        results: dict[str, _Scored] = {}
        probe_results: dict[str, _Scored] = {}
        engine_signature: Optional[str] = None
        signature_source: Optional[str] = None
        unmapped: list[str] = []
        for run in self._ordered():
            if run.state != "done":
                continue
            selector = f"{RUN_UID_LABEL}={run.run_uid}"
            candidates = self._retrying(
                f"list Candidates of Run {run.run_name}",
                functools.partial(self._list_candidates, selector),
            )
            run.published = len(candidates)
            for candidate in sorted(
                candidates, key=lambda obj: (obj.get("metadata") or {}).get("name", "")
            ):
                name = (candidate.get("metadata") or {}).get("name", "")
                parameters = _get(candidate, ("spec", "parameters"))
                try:
                    point, signature = _map_parameters(parameters, run.shard)
                except _Unmapped as error:
                    if error.fatal:
                        self._fail("engine_not_pinned", f"Candidate {name}: {error}")
                    unmapped.append(f"{name}: {error}")
                    continue
                if engine_signature is None:
                    engine_signature, signature_source = signature, name
                elif signature != engine_signature:
                    self._fail(
                        "engine_not_pinned",
                        f"Candidates {signature_source} and {name} differ in "
                        "non-router parameters",
                    )
                key = RetuneParams(router=point).key()
                target = probe_results if run.shard.kind == "probe" else results
                if key in target:
                    continue
                metrics = _get(candidate, ("status", "metrics"), {}) or {}
                entry: dict[str, Any] = {
                    "params": RetuneParams(router=point).to_dict(),
                    "shard": run.shard.id,
                    "candidate": name,
                }
                try:
                    score, info = score_summary(
                        metrics, request.window_s, request.objective
                    )
                except ValueError as error:
                    entry["error"] = str(error)
                    target[key] = _Scored(None, entry)
                    continue
                entry.update(info)
                target[key] = _Scored(score, entry)
        if unmapped:
            self.details["unmapped_candidates"] = unmapped[:10]
        self._check_probe(results, probe_results)

        scores: dict[str, Optional[float]] = {}
        metrics_out: dict[str, dict[str, Any]] = {}
        missing = 0
        for candidate in request.candidates:
            key = candidate.key()
            found = results.get(key)
            if found is not None and set(candidate.router) == domain_keys:
                scores[key] = found.score
                metrics_out[key] = found.entry
            else:
                scores[key] = None
                metrics_out[key] = {
                    "params": candidate.to_dict(),
                    "error": self._missing_reason(candidate.router),
                }
            if scores[key] is None:
                missing += 1
        self.details["missing_candidates"] = missing
        logger.info(
            "retune window %s harvested %d of %d candidates from %d Requests",
            request.window_index,
            len(scores) - missing,
            len(scores),
            sum(1 for run in self.shards.values() if run.state == "done"),
        )
        return self._result("ok", scores=scores, metrics=metrics_out)

    def _list_candidates(self, selector: str) -> list[dict]:
        return self.ctx.kube.list_candidates(selector, timeout_s=self._timeout())

    def _missing_reason(self, point: Mapping[str, float]) -> str:
        for run in self.shards.values():
            if run.shard.kind != "probe" and run.shard.contains(point):
                if run.state == "missing":
                    return f"shard {run.shard.id} {run.reason}: {run.message}"
                return f"not published by Run {run.run_name} (infeasible or failed)"
        return "not covered by any shard"

    def _check_probe(
        self, results: Mapping[str, _Scored], probe_results: Mapping[str, _Scored]
    ) -> None:
        probe = self.shards.get("probe")
        if probe is None:
            return
        point = probe.shard.points[0]
        key = RetuneParams(router=point).key()
        grid = results.get(key)
        repeat = probe_results.get(key)
        grid_score = None if grid is None else grid.score
        probe_score = None if repeat is None else repeat.score
        report: dict[str, Any] = {
            "point": point,
            "grid_score": grid_score,
            "probe_score": probe_score,
        }
        self.details["crn_probe"] = report
        if grid_score is None or probe_score is None:
            self._fail(
                "crn_mismatch",
                f"CRN probe point {point} is unscored (grid {grid_score}, "
                f"probe {probe_score})",
            )
        scale = max(abs(grid_score), abs(probe_score))
        error = 0.0 if scale == 0 else abs(grid_score - probe_score) / scale
        report["relative_error"] = error
        if not error <= CRN_REL_TOLERANCE:
            self._fail(
                "crn_mismatch",
                f"CRN probe point {point} scored {probe_score!r}, its grid "
                f"evaluation {grid_score!r} (relative error {error:.3g})",
            )


class _Unmapped(Exception):
    """A Candidate that maps to no point; ``fatal`` when the engine drifted."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


def _map_parameters(parameters: Any, shard: Shard) -> tuple[dict[str, float], str]:
    """A Candidate's live router point and its non-router signature.

    The signature is the canonical JSON of the parameters without the three
    searched knobs; every Candidate of a window must share it. Raises
    ``_Unmapped`` for parameters that name no point of ``shard``, fatally
    for a router other than the pinned ``kv_router`` without a load model.
    """

    if not isinstance(parameters, Mapping):
        raise _Unmapped("spec.parameters is not an object")
    adapters = parameters.get("adapters")
    if not isinstance(adapters, Mapping):
        raise _Unmapped("spec.parameters.adapters is missing")
    router = adapters.get(ROUTER_ADAPTER)
    if not isinstance(router, Mapping):
        raise _Unmapped(f"spec.parameters.adapters[{ROUTER_ADAPTER!r}] is missing")
    load_model = router.get("prefill_load_model")
    load_type = (
        load_model.get("type") if isinstance(load_model, Mapping) else load_model
    )
    if router.get("policy") != "kv_router" or load_type != "none":
        raise _Unmapped(
            f"router policy {router.get('policy')!r} with prefill load model "
            f"{load_type!r}; the Request pinned kv_router and none",
            fatal=True,
        )
    point: dict[str, float] = {}
    for public, live in DGDR_ROUTER_KNOBS.items():
        value = router.get(public)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _Unmapped(f"router knob {public} is {value!r}")
        snapped = _snap(float(value), shard.choices.get(live, ()))
        if snapped is None:
            raise _Unmapped(
                f"router knob {public}={value!r} is not one of the shard's values"
            )
        point[live] = snapped
    rest = dict(parameters)
    rest["adapters"] = {
        **adapters,
        ROUTER_ADAPTER: {
            key: value for key, value in router.items() if key not in DGDR_ROUTER_KNOBS
        },
    }
    signature = json.dumps(rest, sort_keys=True, separators=(",", ":"), default=str)
    return point, signature


def _snap(value: float, values: Sequence[float]) -> Optional[float]:
    """The domain value ``value`` denotes, allowing for float round-trips."""

    for candidate in values:
        if abs(value - candidate) <= _SNAP_TOLERANCE * max(1.0, abs(candidate)):
            return candidate
    return None


def _condition(obj: Mapping[str, Any], kind: str) -> Optional[dict]:
    for condition in _get(obj, ("status", "conditions"), ()) or ():
        if isinstance(condition, Mapping) and condition.get("type") == kind:
            return dict(condition)
    return None


def _permanent(error: KubeError) -> bool:
    """A Kubernetes error a retry cannot fix (bad request, auth, missing API)."""

    return getattr(error, "status", None) in _PERMANENT_STATUSES or isinstance(
        error, (KubeInvalid, KubeNotFound)
    )


def _admission_denied(error: KubeError) -> bool:
    """A permanent rejection whose message comes from an admission webhook."""

    return _permanent(error) and _ADMISSION_DENIED.search(str(error)) is not None


def _int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _rows(annotations: Any) -> Optional[dict]:
    if not isinstance(annotations, Mapping) or ROWS_ANNOTATION not in annotations:
        return None
    try:
        rows = json.loads(annotations[ROWS_ANNOTATION])
    except (TypeError, ValueError):
        return None
    return rows if isinstance(rows, dict) else None
