# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live retune sweep executor: one window through DGDR v1beta2 Requests.

``run_dgdr_sweep`` is the live ``sweep_fn`` of ``RetuneController``. It
runs on the controller's single sweep thread and never raises:

1. Shard the router grid (``plan_shards``): balanced Cartesian sub-grids of
   at most 100 points (a Request's ``maxCandidates`` bound), one
   single-point shard for an off-grid baseline or startup point, and a CRN
   probe that evaluates one fixed grid point a second time.
2. Adopt the Requests a previous leader created for this window, matched by
   ``(dgd-uid, window, shard, spec hash)``; others of the window are queued
   for deletion. The spec names the window file, whose name holds the
   trace selection (``window_claim_path``), so a Request of another window
   start or cutoff with the same end is never adopted. A window whose
   every shard is adopted as Succeeded is harvested without cutting it
   again. An adopted attempt whose Run failed or ran another input is
   replaced by attempt+1 while the shard still fits before the deadline.
3. Cut the window file (a subprocess, ``context.cut_window``), then create
   one Request per remaining shard with Strict field validation and check
   that the API server kept every submitted field.
4. Wait for each Request's Run, check that ``Run.status.aisim.input``
   is what the Request asked for, and read the Run's Candidates.
5. Map each Candidate's router adapter parameters back to live names,
   require identical non-router parameters across the window, score each
   with ``score_summary`` on ``retune.objective``, and check the CRN probe.

The executor never deletes anything itself: Requests to remove go to the
``SweepRegistry`` deletion queue, which the reconciler drains while this
process leads. Shutdown (``cancel``) and loss of leadership (``abort``)
return with ``failure_reason`` ``shutdown``/``lost_leadership`` and leave
every Request in place for the next leader to adopt. Shutdown ends the wait
between polls at once; ``abort`` is checked when it ends. Every mutating
call first checks ``abort`` and ``context.is_leader()``.

A shard that holds the baseline, or the probe, is critical: when it fails
the window fails. Any other failed shard only leaves its points unscored.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import hashlib
import itertools
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, NoReturn, Optional

from dynamo.planner.config.planner_config import (
    DGDR_MAX_SHARD_POINTS,
    DGDR_MAX_WARMUP_BUCKETS,
    DgdrExecutorConfig,
    RetuneConfig,
    balanced_shard_parts,
    dgdr_need_seconds,
    largest_shard_points,
)
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
from dynamo.planner.retune.live.reconciler import trace_pvc_source
from dynamo.planner.retune.live.window_cut import bucket_history
from dynamo.planner.retune.sweep import (
    SweepInterrupted,
    SweepRequest,
    SweepResult,
    SweepStatus,
    WorkerPool,
    evaluated_params,
    evaluated_planner,
    predictor_swap_errors,
    predictor_swap_tasks,
    run_presweeps,
    scales_on_throughput,
    score_summary,
    sweep_throughput_intervals,
    write_observations,
)
from dynamo.planner.simulation.config import inner_planner_config, retune_sweep_base

logger = logging.getLogger(__name__)

ShardKind = Literal["grid", "point", "probe"]

# Seconds a new Request may go without an Accepted condition before the
# window fails as operator_not_reconciling.
ACCEPT_TIMEOUT_S = 60.0
# Relative score difference the CRN probe tolerates.
CRN_REL_TOLERANCE = 1e-9
# Annotation holding the cut window's row counts, so a window adopted as
# Succeeded reports them without cutting the window again.
ROWS_ANNOTATION = LABEL_PREFIX + "rows"
# Operator constants (deploy/operator at #17483).
REQUEST_GENERATION_ANNOTATION = "nvidia.com/dgdr-request-generation"
TRACE_MOUNT_PATH = "/var/run/aisimulate/trace"
ROUTER_ADAPTER = "dynamo.router"
PLANNER_ADAPTER = "dynamo.planner"
# The planner adapter's live retune sweep section
# (``simulation.provider.RETUNE_SWEEP``). A Candidate's planner parameters
# are the section narrowed to its one point.
RETUNE_SWEEP = "retune_sweep"
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
    ``probe`` shard has one value per knob. ``planners`` are the planner
    points (``PlannerConfig`` fields) the shard crosses with them when its
    candidates replay an inner planner (Scenario 2; ``({},)`` when no
    planner field is searched), and ``()`` with fixed replicas.
    """

    id: str
    kind: ShardKind
    choices: dict[str, tuple[float, ...]]
    points: tuple[dict[str, float], ...]
    planners: tuple[dict[str, Any], ...] = ()

    @property
    def size(self) -> int:
        return len(self.points) * max(1, len(self.planners))

    def contains(
        self, point: Mapping[str, float], planner: Optional[Mapping[str, Any]] = None
    ) -> bool:
        if not all(point.get(key) in values for key, values in self.choices.items()):
            return False
        return (
            planner is None
            or not self.planners
            or any(_same_planner(planner, option) for option in self.planners)
        )

    @property
    def searches_planner(self) -> bool:
        """Whether the shard's candidates replay an inner planner."""
        return bool(self.planners)


def _same_planner(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return set(left) == set(right) and all(
        _same_json(left[key], right[key]) for key in left
    )


def plan_shards(
    router_domain: Mapping[str, Sequence[float]],
    *,
    baseline: Mapping[str, float],
    startup: Mapping[str, float],
    max_shards: int,
    parallelism: int,
    probe: bool = True,
    planner_points: Sequence[Mapping[str, Any]] = (),
    baseline_planner: Optional[Mapping[str, Any]] = None,
    startup_planner: Optional[Mapping[str, Any]] = None,
) -> list[Shard]:
    """Split ``router_domain`` (live knob -> values) into Request shards.

    The grid becomes at most ``max_shards`` Cartesian sub-grids of at most
    ``DGDR_MAX_SHARD_POINTS`` points: each knob's sorted values are cut into
    contiguous, near-equal chunks by ``balanced_shard_parts``, the split
    the configuration check (P13) uses. It minimizes the largest shard's
    evaluation rounds at the Run ``parallelism`` (``dgdr_rounds``), then
    the shard count, so a shard that would not shorten the window is not
    added.
    ``baseline`` and ``startup`` (live knob -> value) each get a
    single-point shard when off the grid.
    With ``probe``, a ``probe`` shard repeats one fixed grid point (the
    middle value of each knob, a positive temperature when the grid has
    one) to check common random numbers across Runs. Raises ``ValueError``
    when no split fits or a point does not cover the domain's knobs.

    ``planner_points`` (distinct planner points, in order) add one more
    dimension, cut into contiguous chunks like a router knob, so a shard
    crosses its router sub-grid with a chunk of planner points. The
    baseline and startup points then also carry their planner parts
    (``baseline_planner``, ``startup_planner``), and the probe takes the
    middle planner point.
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

    planners = tuple(dict(point) for point in planner_points)
    sizes = [len(axes[key]) for key in keys]
    if planners:
        sizes.append(len(planners))
    parts = balanced_shard_parts(sizes, max_shards, parallelism=parallelism)
    largest = largest_shard_points(sizes, parts)
    if largest > DGDR_MAX_SHARD_POINTS:
        raise ValueError(
            f"the router grid of {math.prod(sizes)} points cannot be split into "
            f"at most {max_shards} Cartesian shards of at most "
            f"{DGDR_MAX_SHARD_POINTS} points (the smallest largest shard has "
            f"{largest}); raise retune.dgdr.run.max_shards or shrink the domain"
        )

    shards: list[Shard] = []
    chunked: list[list[Any]] = [
        _chunks(axes[key], part) for key, part in zip(keys, parts)
    ]
    if planners:
        chunked.append(_chunks(planners, parts[-1]))
    for index, combo in enumerate(itertools.product(*chunked)):
        choices = dict(zip(keys, combo))
        shards.append(
            Shard(
                id=f"grid-{index}",
                kind="grid",
                choices=choices,
                points=_product(choices),
                planners=tuple(combo[len(keys)]) if planners else (),
            )
        )

    def in_grid(point: Mapping[str, float], planner: Mapping[str, Any]) -> bool:
        return all(float(point[key]) in axes[key] for key in keys) and (
            not planners or any(_same_planner(planner, p) for p in planners)
        )

    def single(
        shard_id: str,
        kind: ShardKind,
        point: Mapping[str, float],
        planner: Mapping[str, Any],
    ) -> Shard:
        values = {key: float(point[key]) for key in keys}
        return Shard(
            id=shard_id,
            kind=kind,
            choices={key: (value,) for key, value in values.items()},
            points=(values,),
            planners=(dict(planner),) if planners else (),
        )

    baseline_point = {key: float(baseline[key]) for key in keys}
    startup_point = {key: float(startup[key]) for key in keys}
    base_planner = dict(baseline_planner or {})
    start_planner = dict(startup_planner or {})
    if not in_grid(baseline_point, base_planner):
        shards.append(single("point-baseline", "point", baseline_point, base_planner))
    if not in_grid(startup_point, start_planner) and (
        startup_point != baseline_point
        or not _same_planner(start_planner, base_planner)
    ):
        shards.append(single("point-startup", "point", startup_point, start_planner))
    if probe:
        middle = planners[len(planners) // 2] if planners else {}
        shards.append(single("probe", "probe", _probe_point(axes), middle))
    return shards


def _chunks(values: Sequence[Any], count: int) -> list[tuple[Any, ...]]:
    """``values`` cut into ``count`` contiguous chunks whose sizes differ by <= 1."""

    base, extra = divmod(len(values), count)
    chunks: list[tuple[Any, ...]] = []
    start = 0
    for index in range(count):
        end = start + base + (1 if index < extra else 0)
        chunks.append(tuple(values[start:end]))
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


# ----------------------------------------------------------------- Request


def spec_hash(spec: Mapping[str, Any]) -> str:
    """First 16 hex digits of the SHA-256 of the canonical JSON of ``spec``."""

    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def window_selection(window_start_ms: int, window_end_ms: int, as_of_ms: int) -> str:
    """First 12 hex digits of the SHA-256 of ``<start>|<end>|<as_of>`` (epoch ms)."""

    key = f"{int(window_start_ms)}|{int(window_end_ms)}|{int(as_of_ms)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def window_claim_path(
    cfg: DgdrExecutorConfig,
    dgd: DgdRef,
    *,
    window_start_ms: int,
    window_end_ms: int,
    as_of_ms: int,
) -> str:
    """Claim-relative path of a window file:
    ``<windows_dir>/<dgd>/<end_ms>-<window_selection>.jsonl.gz``.

    The Request spec names this path, so the selection (``[start, end)``
    as of ``as_of``) is part of the spec hash and the Request name: a
    window cut or a Request of another selection with the same end is
    never reused. The reconciler reads the window end from the leading
    digits.
    """

    selection = window_selection(window_start_ms, window_end_ms, as_of_ms)
    return (
        f"{cfg.windows_dir.strip('/')}/{dgd.name}/"
        f"{int(window_end_ms)}-{selection}.jsonl.gz"
    )


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
    window_start_ms: int,
    window_end_ms: int,
    as_of_ms: int,
    shard: Shard,
    spec_replicas: Mapping[str, int],
    backend: str,
    attempt: int,
    planner_base: Optional[Mapping[str, Any]] = None,
    planner_warmup: Optional[Mapping[str, Any]] = None,
) -> dict:
    """The DGDR v1beta2 Request that replays one shard of a window.

    The trace is the window file of ``[window_start_ms, window_end_ms)`` as
    of ``as_of_ms`` (``window_claim_path``). The engine is ``cfg.engine``
    with every leaf a scalar and each used role's
    ``parallelism.preset[0].replicas`` set from ``spec_replicas``
    (unused roles are deleted with ``null``); the GPU budget is the sum of
    replicas times GPUs per worker (tensor x pipeline x attention_data).
    Router knobs use their AISimulate public names: ``{choices: [...]}``
    for a grid shard, a scalar for a point or probe shard. ``cfg.engine``
    is the one the configuration check (P5) accepted. Raises
    ``ValueError`` for ``spec_replicas`` or a shard the Request cannot
    express.

    A shard with planner points (Scenario 2) adds the planner adapter's
    ``retune_sweep`` section: ``planner_base`` (the live config's
    ``retune_sweep_base``), the shard's points, and ``planner_warmup``
    (history rows by throughput interval). ``spec_replicas`` are then the
    ready counts at the window start, where each candidate's inner planner
    starts scaling.
    """

    engine = _pinned_engine(cfg.engine, spec_replicas, backend)
    size = shard.size
    if not 1 <= size <= DGDR_MAX_SHARD_POINTS:
        raise ValueError(
            f"shard {shard.id} has {size} points; "
            f"1..{DGDR_MAX_SHARD_POINTS} fit a Request"
        )
    spec: dict[str, Any] = {
        "model": {"name": cfg.model},
        "backends": [backend],
        "hardware": {
            "gpu": {"sku": cfg.sku, "budget": _gpu_budget(engine, spec_replicas)}
        },
        # The trace format is left to its CRD default, Dynamo.
        "workload": {
            "trace": {
                "source": {
                    "persistentVolumeClaim": {
                        "claimName": cfg.claim_name,
                        "subPath": window_claim_path(
                            cfg,
                            dgd,
                            window_start_ms=window_start_ms,
                            window_end_ms=window_end_ms,
                            as_of_ms=as_of_ms,
                        ),
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
        "aisim": {
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
                **(
                    {
                        "planner": {
                            RETUNE_SWEEP: {
                                "base": copy.deepcopy(dict(planner_base or {})),
                                "points": [dict(point) for point in shard.planners],
                                "warmup": copy.deepcopy(dict(planner_warmup or {})),
                            }
                        }
                    }
                    if shard.searches_planner
                    else {}
                ),
                # The planner reads only each Candidate's parameters and
                # score, never its DGD. The direct generator keeps retune
                # Runs independent of the AIC generator, whose rendering can
                # reject a Candidate (ai-dynamo/dynamo#17581: no explicit
                # block size) and so fail the Run and the window.
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
    """``engine`` with the deployed backend and the DGD spec replicas.

    The configuration check (P5) has already pinned ``engine``: its backend,
    if written, is ``backend``, and each worker role of the planner mode has
    one parallelism preset entry of positive integers. ``spec_replicas``
    comes from the live DGD, so its roles and counts are checked here.
    """

    roles = set(spec_replicas)
    if roles not in ({"aggregated"}, {"prefill", "decode"}):
        raise ValueError(
            "spec_replicas must name 'aggregated', or 'prefill' and 'decode'; "
            f"got {sorted(roles)}"
        )
    pinned = copy.deepcopy(dict(engine))
    pinned["backend"] = backend
    workers = pinned["workers"]
    for role in _ROLES:
        if role not in roles:
            workers[role] = None
            continue
        replicas = int(spec_replicas[role])
        if replicas < 1:
            raise ValueError(
                f"{role} has {replicas} spec replicas; at least 1 is needed"
            )
        workers[role]["parallelism"]["preset"][0]["replicas"] = replicas
    return pinned


def _gpu_budget(engine: Mapping[str, Any], spec_replicas: Mapping[str, int]) -> int:
    budget = 0
    for role, replicas in spec_replicas.items():
        entry = engine["workers"][role]["parallelism"]["preset"][0]
        per_worker = entry["tensor"] * entry["pipeline"] * entry["attention_data"]
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


def _get(obj: Any, path: Sequence[str], default: Any = None) -> Any:
    for key in path:
        if not isinstance(obj, Mapping) or key not in obj:
            return default
        obj = obj[key]
    return obj


def _same_json(left: Any, right: Any) -> bool:
    """JSON equality of leaves: numbers by value (2 == 2.0), booleans never
    equal numbers."""

    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return type(left) is type(right) and left == right


def missing_fields(stored: Any, submitted: Any, path: str = "") -> list[str]:
    """Submitted leaves the stored object lacks or changed (pruned or rejected).

    ``stored`` may hold more (admission defaults); a submitted ``null`` only
    requires the stored value to be absent or ``null``.
    """

    prefix = (path,) if path else ()
    return [
        ".".join(where) or "."
        for where, _, _ in _differences(stored, submitted, prefix)
    ]


def _differences(
    stored: Any, submitted: Any, path: tuple[str, ...]
) -> list[tuple[tuple[str, ...], Any, Any]]:
    """``(path, submitted, stored)`` of every submitted value ``stored`` lacks.

    A mapping may hold more keys than submitted, at any depth, including
    the mappings of a list of the submitted length. A submitted ``null``
    requires the stored value to be absent or ``null``; ``_ABSENT`` stands
    for an absent stored value.
    """

    if submitted is None:
        if stored is None or stored is _ABSENT:
            return []
        return [(path, None, stored)]
    if isinstance(submitted, Mapping):
        if not isinstance(stored, Mapping):
            return [(path, submitted, stored)]
        found: list[tuple[tuple[str, ...], Any, Any]] = []
        for key, value in submitted.items():
            found += _differences(stored.get(key, _ABSENT), value, (*path, str(key)))
        return found
    if (
        isinstance(submitted, list)
        and isinstance(stored, list)
        and len(stored) == len(submitted)
    ):
        found = []
        for index, (left, right) in enumerate(zip(stored, submitted)):
            found += _differences(left, right, (*path, str(index)))
        return found
    return [] if _same_json(stored, submitted) else [(path, submitted, stored)]


def _overlay(target: Any, patch: Any) -> Any:
    """``patch`` merged over ``target`` like a merge patch, but a ``null`` in
    ``patch`` stays (as "must be unset") instead of deleting the key."""

    if not isinstance(patch, Mapping):
        return copy.deepcopy(patch)
    result = dict(target) if isinstance(target, Mapping) else {}
    for key, value in patch.items():
        result[key] = _overlay(result.get(key), value)
    return result


def input_mismatches(
    spec: Mapping[str, Any], effective: Mapping[str, Any]
) -> list[str]:
    """How ``Run.status.aisim.input`` differs from what ``spec`` asks for.

    The expected input is what the operator lowers ``spec``'s typed fields
    to (model, backends, GPU SKU and budget, objective, SLA, the trace file
    on its read-only mount, ``maxCandidates``) with ``inputOverrides``
    merged over them. ``inputOverrides`` wins: nothing restores the typed
    trace source after it. The check is the create path's subset check
    (``missing_fields``), so the input may carry more, such as a new
    operator default, but must hold every expected value; an expected
    ``null`` (a worker role the overrides delete, an SLA or a replay
    speedup the Request leaves unset) must be absent or ``null``.
    """

    _, sub_path = trace_pvc_source(spec)
    mounted = TRACE_MOUNT_PATH + (
        "/trace.jsonl.gz" if (sub_path or "").endswith(".gz") else "/trace.jsonl"
    )
    backends = list(spec.get("backends") or [])
    sla = _get(spec, ("objective", "sla"), {}) or {}
    typed = {
        "engine": {
            "model": _get(spec, ("model", "name")),
            "backend": {"choices": backends} if backends else None,
            "hardware": _get(spec, ("hardware", "gpu", "sku")),
        },
        "optimization": {
            "target": _METRIC_TARGETS.get(_get(spec, ("objective", "metric"), "")),
            "constraints": {
                "max_candidate_gpus": _get(spec, ("hardware", "gpu", "budget"))
            },
        },
        "evaluation": {
            "sla": {
                "ttft_ms": sla.get("ttftMs"),
                "itl_ms": sla.get("itlMs"),
                "e2e_ms": sla.get("e2eMs"),
            }
        },
        "traffic": {
            "source": {"type": "trace", "format": "dynamo", "paths": [mounted]},
            "load": {
                "type": "trace_timestamps",
                "speedup": _get(
                    spec, ("workload", "trace", "replay", "arrivalSpeedupRatio")
                ),
            },
        },
        "dgdr_run": {"max_candidates": _get(spec, ("recommendation", "maxCandidates"))},
    }
    overrides = _get(spec, ("aisim", "inputOverrides"), {}) or {}
    # Normalize like the API server (JSON numbers, no tuples).
    expected = json.loads(json.dumps(_overlay(typed, overrides)))
    actual = json.loads(json.dumps(effective))
    return [
        f"{'.'.join(path)}: expected "
        f"{'unset' if want is None else repr(want)}, got "
        f"{'unset' if got is _ABSENT else repr(got)}"
        for path, want, got in _differences(actual, expected, ())
    ]


# ---------------------------------------------------------------- executor


@dataclass(frozen=True)
class DgdrContext:
    """Everything ``run_dgdr_sweep`` needs besides the ``SweepRequest``.

    ``backend`` is the deployed backend (``vllm``, ``sglang``, ``trtllm``).
    ``is_leader`` is checked before every mutating call. ``now`` is the
    wall clock (``deadline_s`` is epoch seconds). Between polls the sweep
    waits on ``cancel``, so shutdown ends the wait at once; ``sleep``, when
    set, replaces that wait (a test's simulated clock).
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
    sleep: Optional[Callable[[float], None]] = None
    poll_interval_s: float = 5.0
    # Scenario 2: the history reader and the pre-sweep's worker pool
    # (window_cut.bucket_history and sweep.WorkerPool; tests replace them).
    bucket_history: Callable[..., Any] = field(default=bucket_history)
    worker_pool: Callable[..., Any] = field(default=WorkerPool)


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


class _Stop:
    """Shutdown (``cancel``) or loss of leadership (``abort``) as the one
    ``is_set()`` flag the window cutter polls."""

    def __init__(
        self, cancel: threading.Event, abort: Optional[threading.Event]
    ) -> None:
        self._events = tuple(event for event in (cancel, abort) if event is not None)

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)


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
    # Why the shard waits: the Request's Accepted reason and message while
    # it has no Run (RunNameConflict, RunCreating, CancelingSupersededRun),
    # then the open Run's Completed reason and message (JobRunning,
    # ContainerWaiting, TraceSourceMissing); None once the Run ends.
    run_status: Optional[str] = None
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

    def status(self) -> Optional[str]:
        """Why the open Request or Run waits, else why the shard ended or
        its last attempt was replaced."""

        return self.run_status or self.message

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
            "message": self.status(),
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
        self.stop = _Stop(cancel, abort)
        self.ctx = context
        self.cfg = context.cfg
        self.started = started
        self.deadline = (
            float(request.deadline_s)
            if request.deadline_s is not None
            else started + request.timeout_s
        )
        # Rounded once: the window file's path, the Request specs and the
        # cut must agree on the selection.
        self.window_start_ms = int(round(request.window_start_ms))
        self.window_end_ms = int(round(request.window_end_ms))
        self.as_of_ms = int(round(request.as_of_ms))
        # Registry references this sweep took; each is released at the end.
        self._protections = 0
        self.claim_path = window_claim_path(
            self.cfg,
            context.dgd,
            window_start_ms=self.window_start_ms,
            window_end_ms=self.window_end_ms,
            as_of_ms=self.as_of_ms,
        )
        self.local_path = os.path.join(self.cfg.mount_path, self.claim_path)
        self.shards: dict[str, _ShardRun] = {}
        self.cut_done = False
        self.num_requests = 0
        self.row_counts: dict[str, int] = {}
        self.details: dict[str, Any] = {
            "window": {"path": self.claim_path, "cut": False},
            "deadline_s": self.deadline,
            "generation": request.generation,
            "replicas_verified": request.replicas_verified,
        }
        self.k8s_errors = 0
        self.runtime: dict[str, Any] = {}
        # Scenario 2 (set by _prepare_planner): what each candidate replays
        # as, the planner points and warmup the Requests carry, and the
        # pre-sweep and swap inputs the decision reads.
        self.evaluated: dict[str, RetuneParams] = {}
        self.planner_points: list[dict[str, Any]] = []
        self.planner_base: Optional[dict[str, Any]] = None
        self.planner_warmup: Optional[dict[str, Any]] = None
        self.presweep: Optional[dict[str, Any]] = None
        self.interval_observations: dict[int, tuple[Any, ...]] = {}
        self.warmup_errors: dict[int, str] = {}
        self.base_interval: Optional[int] = None

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
        base_interval = self.base_interval
        return SweepResult(
            status=status,
            num_requests=self.num_requests,
            row_counts=dict(self.row_counts),
            scores=scores or {},
            metrics=metrics or {},
            predictor_presweep=self.presweep,
            warmup_observations=(
                None
                if base_interval is None
                else self.interval_observations.get(base_interval)
            ),
            predictor_warmup_error=(
                None if base_interval is None else self.warmup_errors.get(base_interval)
            ),
            wall_s=max(0.0, self._now() - self.started),
            error=error,
            failure_reason=failure_reason,
            details=details,
            interval_observations=dict(self.interval_observations),
            predictor_warmup_errors=dict(self.warmup_errors),
        )

    # ----------------------------------------------------------- helpers

    def _now(self) -> float:
        return self.ctx.now()

    def _timeout(self) -> float:
        return max(_MIN_K8S_TIMEOUT_S, min(_K8S_TIMEOUT_S, self.deadline - self._now()))

    def _pause(self, seconds: float) -> None:
        """Wait between polls. Shutdown ends the wait at once; loss of
        leadership is seen when it ends, and every mutating call checks
        ``is_leader()`` first anyway."""

        if self.ctx.sleep is not None:
            self.ctx.sleep(seconds)
        else:
            self.cancel.wait(seconds)
        self._check_interrupt()

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
                self._pause(self.ctx.poll_interval_s)

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
        domain = request.router_domain
        if not domain:
            self._fail("invalid_request", "the sweep request has no router domain")
        if request.base_planner is None and any(
            candidate.planner for candidate in request.candidates
        ):
            self._fail(
                "invalid_request",
                "planner candidates need the inner planner's base configuration",
            )
        keys = set(domain)
        if request.base_planner is not None:
            self._prepare_planner()
        startup = self._startup_candidate(domain)
        baseline = self._evaluated(request.baseline)
        planner_points = self.planner_points
        try:
            shards = plan_shards(
                domain,
                baseline=baseline.router,
                startup=startup.router,
                max_shards=self.cfg.run.max_shards,
                parallelism=self.cfg.run.parallelism,
                planner_points=planner_points,
                baseline_planner=baseline.planner,
                startup_planner=startup.planner,
            )
        except (KeyError, ValueError) as error:
            self._fail("invalid_domain", str(error))
        # A scaling planner starts each candidate from the ready counts at
        # the window start (SweepRequest.replicas); a fixed one from the spec.
        replicas = (
            request.replicas
            if request.base_planner is not None
            else request.spec_replicas or request.replicas
        )
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
                    window_start_ms=self.window_start_ms,
                    window_end_ms=self.window_end_ms,
                    as_of_ms=self.as_of_ms,
                    shard=shard,
                    spec_replicas=replicas,
                    backend=self.ctx.backend,
                    attempt=1,
                    planner_base=self.planner_base,
                    planner_warmup=self.planner_warmup,
                )
            except (KeyError, TypeError, ValueError) as error:
                self._fail("invalid_engine", f"shard {shard.id}: {error}")
            # The baseline must be scored, and the probe's grid twin is what
            # the probe is compared with.
            critical = (
                shard.kind == "probe"
                or shard.contains(baseline.router, baseline.planner)
                or (
                    probe is not None
                    and shard.contains(
                        probe.points[0], probe.planners[0] if probe.planners else None
                    )
                )
            )
            self.shards[shard.id] = _ShardRun(
                shard=shard,
                critical=critical,
                spec=body["spec"],
                digest=body["metadata"]["labels"][LABEL_SPEC],
            )
        uncovered = [
            candidate.to_dict()
            for candidate in request.candidates
            if set(candidate.router) != keys
            or not self._covered(self._evaluated(candidate))
        ]
        if uncovered:
            self.details["uncovered_candidates"] = uncovered[:10]

    def _covered(self, point: RetuneParams) -> bool:
        return any(
            run.shard.kind != "probe"
            and run.shard.contains(point.router, point.planner)
            for run in self.shards.values()
        )

    def _evaluated(self, candidate: RetuneParams) -> RetuneParams:
        """The point a candidate replays as (``sweep.evaluated_params``)."""

        return self.evaluated.get(candidate.key(), candidate)

    def _startup_candidate(self, domain: Mapping[str, Sequence[float]]) -> RetuneParams:
        """The off-grid candidate other than the baseline, else the baseline.

        The controller's candidates are the grid plus the startup point and
        the baseline, so an off-grid candidate is one of those two. A
        candidate is off the grid when its router point is, or when its
        evaluated planner part is not on the planner axis.
        """

        baseline = self._evaluated(self.request.baseline)
        axis = self.planner_points
        for candidate in self.request.candidates:
            router = candidate.router
            if candidate == self.request.baseline or set(router) != set(domain):
                continue
            evaluated = self._evaluated(candidate)
            if not all(float(router[key]) in domain[key] for key in domain) or (
                axis and not any(_same_planner(evaluated.planner, p) for p in axis)
            ):
                return evaluated
        return baseline

    def _prepare_planner(self) -> None:
        """Bucket the history, run the predictor pre-sweep and fix the
        planner points each Request carries (Scenario 2).

        Within ``cut_budget_seconds`` of the deadline, before any Request is
        planned, because the points and the warmup are part of every
        Request's spec: the history before the window (and the window
        itself, for the pre-sweep) is read from the trace segments at each
        throughput interval a candidate scales at; the pre-sweep picks a
        predictor per interval; each candidate is mapped to the point it
        replays as (``evaluated_params``); and every predictor a decision
        could swap in is prepared on the same observations, as replay
        retune does, so the decision can check it.
        """

        request = self.request
        budget = min(self.cfg.cut_budget_seconds, self.deadline - self._now())
        if budget <= 0:
            self._fail("launch_too_late", "no time left to read the history", "timeout")
        deadline_s = self._now() + budget
        intervals = sweep_throughput_intervals(request)
        observations: dict[int, tuple[Any, ...]] = {}
        warmups: dict[int, tuple[Any, ...]] = {}
        if intervals:
            history = self._bucket_history(intervals, deadline_s)
            observations = dict(history.observations)
            warmups = dict(history.warmups)
            self.details["history"] = {
                "start_ms": int(round(request.history_start_ms)),
                "requests": history.history_requests,
                "dropped_members": history.dropped_members,
                "intervals": list(intervals),
            }
        presweep = None
        warmup_errors: dict[int, str] = {}
        # The configuration the Runs replay (only its scaling fields): the
        # live dump may hold settings, such as non-advisory batch
        # scheduling, that need what a dump leaves out.
        inner_base = inner_planner_config(retune_sweep_base(request.base_planner or {}))
        with tempfile.TemporaryDirectory(prefix="retune-dgdr-planner-") as work:
            work_dir = Path(work)
            paths = {
                interval: write_observations(
                    work_dir / f"observations-{interval}.json", rows
                )
                for interval, rows in observations.items()
            }
            pool = self.ctx.worker_pool(
                self.ctx.retune.max_parallel,
                time.monotonic() + (deadline_s - self._now()),
                self.stop,
                work,
            )
            try:
                if request.predictor_candidates and intervals:
                    presweep = run_presweeps(
                        request, pool, observations, paths, intervals
                    )
                for candidate in request.candidates:
                    self.evaluated[candidate.key()] = evaluated_params(
                        candidate, request.baseline, presweep, request.base_planner
                    )
                tasks = predictor_swap_tasks(
                    request,
                    list(self.evaluated.values()),
                    inner_base,
                    observations,
                    paths,
                    work_dir,
                )
                if tasks:
                    warmup_errors = predictor_swap_errors(pool.run(tasks))
            except SweepInterrupted as interrupted:
                self._check_interrupt()
                self._fail(
                    "window_cut_timeout",
                    f"history and predictor pre-sweep: {interrupted}",
                    "timeout",
                )
            finally:
                pool.close()
        # The planner axis: the grid's planner points as evaluated (what
        # P13 counts). The baseline, evaluated as is, and the startup point
        # are off the grid when their planner part is not among them, and
        # plan_shards gives each a single-point shard.
        if request.grid_planners is None:
            sources = [point.planner for point in self.evaluated.values()]
        else:
            sources = [
                evaluated_planner(
                    planner, request.baseline, presweep, request.base_planner
                )
                for planner in request.grid_planners
            ]
        planners: dict[str, dict[str, Any]] = {}
        for planner in sources:
            planners.setdefault(RetuneParams(planner=planner).key(), planner)
        self.planner_points = [planners[key] for key in sorted(planners)]
        self.planner_base = retune_sweep_base(request.base_planner or {})
        self.planner_warmup = {
            str(interval): [list(dataclasses.astuple(row)) for row in rows]
            for interval, rows in warmups.items()
        }
        oversized = {
            interval: len(rows)
            for interval, rows in warmups.items()
            if len(rows) > DGDR_MAX_WARMUP_BUCKETS
        }
        if oversized:
            self._fail(
                "invalid_request",
                f"the history warmup has {oversized} buckets by interval, more "
                f"than {DGDR_MAX_WARMUP_BUCKETS} per interval a Request carries",
            )
        self.presweep = presweep
        self.interval_observations = observations
        self.warmup_errors = warmup_errors
        running = {**(request.base_planner or {}), **request.baseline.planner}
        self.base_interval = scales_on_throughput({}, running)
        self.details["planner"] = {
            "points": len(self.planner_points),
            "presweep": presweep,
            "warmup_errors": {str(k): v for k, v in warmup_errors.items()},
        }

    def _bucket_history(self, intervals: Sequence[int], deadline_s: float) -> Any:
        request = self.request
        try:
            return self.ctx.bucket_history(
                self._segments_root(),
                int(round(request.history_start_ms)),
                self.window_start_ms,
                self.window_end_ms,
                self.as_of_ms,
                intervals,
                deadline_s=deadline_s,
                cancel=self.stop,
            )
        except Exception as error:
            self._check_interrupt()
            reason = getattr(error, "reason", None)
            if reason not in ("window_cut_timeout", "trace_corrupt"):
                reason = "window_cut_failed"
            self._fail(reason, f"history read: {type(error).__name__}: {error}")

    def _segments_root(self) -> str:
        if self.request.trace_paths:
            return self.request.trace_paths[0]
        return os.path.join(
            self.cfg.mount_path, self.cfg.segments_dir, self.ctx.dgd.name
        )

    def _build(self, run: _ShardRun) -> dict:
        body = build_request(
            dgd=self.ctx.dgd,
            cfg=self.cfg,
            retune=self.ctx.retune,
            sla_ttft_ms=self.request.sla_ttft_ms,
            sla_itl_ms=self.request.sla_itl_ms,
            window_start_ms=self.window_start_ms,
            window_end_ms=self.window_end_ms,
            as_of_ms=self.as_of_ms,
            shard=run.shard,
            spec_replicas=self.details["replicas"],
            backend=self.ctx.backend,
            attempt=run.attempt,
            planner_base=self.planner_base,
            planner_warmup=self.planner_warmup,
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

        need = dgdr_need_seconds(run.shard.size, self.cfg, needs_cut=not self.cut_done)
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
        segments_root = self._segments_root()
        budget = min(self.cfg.cut_budget_seconds, self.deadline - self._now())
        if budget <= 0:
            self._fail("launch_too_late", "no time left to cut the window", "timeout")
        # WindowCutter takes the deadline as absolute wall-clock seconds.
        cut_deadline_s = self._now() + budget
        try:
            os.makedirs(os.path.dirname(self.local_path), exist_ok=True)
            result: WindowCutResult = self.ctx.cut_window(
                segments_root,
                self.window_start_ms,
                self.window_end_ms,
                self.as_of_ms,
                self.local_path,
                max_rows=self.cfg.max_window_rows,
                deadline_s=cut_deadline_s,
                cancel=self.stop,
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
                last = [
                    f"{run.shard.id}: {run.status()}"
                    for run in sorted(active, key=lambda run: run.shard.id)
                    if run.status()
                ]
                self._fail(
                    "run_pending" if pending else "timeout",
                    "the deadline passed with shards "
                    f"{sorted(run.shard.id for run in active)} unfinished"
                    + (f" (Runs pending: {sorted(pending)})" if pending else "")
                    + (f"; {'; '.join(last)}" if last else ""),
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
                self._pause(max(0.0, min(self.ctx.poll_interval_s, left)))

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
        """One look at a shard's Request and Run.

        The Request must be Accepted for its current generation (within
        ``ACCEPT_TIMEOUT_S``); its Run's input is read back once per Run
        UID; a Succeeded Run makes the shard ``done``, a failed one goes to
        ``_run_failed``, and a live one is waited on.
        """

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
            # RunCreating, RunNameConflict, CancelingSupersededRun: wait, and
            # keep why, so a deadline that passes here reports it.
            run.run_status = (
                f"Request {run.name} {accepted.get('reason')}: "
                f"{accepted.get('message', '')}"
            )
            return
        try:
            found = self._find_run(obj)
        except KubeError as error:
            self._read_failed(f"get the Run of Request {run.name}", error)
            return
        if found is None:
            run.run_status = None
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
        run.run_status = outcome if not terminal and completed.get("reason") else None
        if not terminal:
            self._track_pending(run, status)
        if run.checked_run_uid != run_uid:
            effective = _get(status, ("aisim", "input"))
            if not isinstance(effective, Mapping):
                if not terminal:
                    return  # the input is recorded together with jobRef
                if not succeeded:
                    self._run_failed(run, outcome)
                    return
                self._input_mismatch(run, ["status.aisim.input is missing"])
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

        The operator writes ``status.aisim.input`` and ``jobRef`` in one
        status patch once the Job exists; until then the Run has no status,
        or only a ``Completed=False`` condition that says why it waits (such
        as ``TraceSourceMissing``), so this runs before the input is read
        back.
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
                    "retune Run %s has no Job after %.0fs (run_pending): %s",
                    run.run_name,
                    now - run.pending_since,
                    run.run_status or "no status yet",
                )
            run.run_pending = True

    def _run_failed(self, run: _ShardRun, outcome: str) -> None:
        """A Run ended without success.

        An attempt a previous leader created is replaced while there is time,
        since only a live or Succeeded attempt is worth adopting; otherwise
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
            # For audit: the runtime that replayed the window.
            self.runtime = {"runtime_image": image, "runtime_version": version}
        problems = input_mismatches(run.spec, effective)
        if problems:
            self._input_mismatch(run, problems)
            return False
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

        need = dgdr_need_seconds(run.shard.size, self.cfg, needs_cut=not self.cut_done)
        if self.deadline - self._now() < need:
            return False
        self._enqueue_delete([run.name])
        run.attempt += 1
        run.state = "create"
        run.adopted = False
        run.created = False
        run.name = None
        run.run_name = None
        run.run_status = None
        run.checked_run_uid = None
        run.unaccepted_since = None
        run.pending_since = None
        run.run_pending = False
        run.message = message
        return True

    # ----------------------------------------------------------- harvest

    def _harvest(self) -> SweepResult:
        request = self.request
        domain_keys = set(request.router_domain or {})
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
                    point, planner, signature = _map_parameters(parameters, run.shard)
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
                params = RetuneParams(router=point, planner=planner)
                key = params.key()
                target = probe_results if run.shard.kind == "probe" else results
                if key in target:
                    continue
                metrics = _get(candidate, ("status", "metrics"), {}) or {}
                entry: dict[str, Any] = {
                    "params": params.to_dict(),
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
            evaluated = self._evaluated(candidate)
            found = results.get(evaluated.key())
            if found is not None and set(candidate.router) == domain_keys:
                scores[key] = found.score
                metrics_out[key] = found.entry
            else:
                scores[key] = None
                metrics_out[key] = {
                    "params": evaluated.to_dict(),
                    "error": self._missing_reason(evaluated),
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

    def _missing_reason(self, point: RetuneParams) -> str:
        for run in self.shards.values():
            if run.shard.kind != "probe" and run.shard.contains(
                point.router, point.planner
            ):
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
        planner = probe.shard.planners[0] if probe.shard.planners else {}
        key = RetuneParams(router=point, planner=planner).key()
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


def _map_parameters(
    parameters: Any, shard: Shard
) -> tuple[dict[str, float], dict[str, Any], str]:
    """A Candidate's live router point, planner point and other signature.

    The signature is the canonical JSON of the parameters without the three
    searched router knobs and the planner point; every Candidate of a
    window must share it. The planner point is ``{}`` for a shard without
    planner points. Raises ``_Unmapped`` for parameters that name no point
    of ``shard``, fatally for a router other than the pinned ``kv_router``
    without a load model.
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
    planner: dict[str, Any] = {}
    rest_adapters = {
        **adapters,
        ROUTER_ADAPTER: {
            key: value for key, value in router.items() if key not in DGDR_ROUTER_KNOBS
        },
    }
    if shard.searches_planner:
        section = adapters.get(PLANNER_ADAPTER)
        if not isinstance(section, Mapping):
            raise _Unmapped(f"spec.parameters.adapters[{PLANNER_ADAPTER!r}] is missing")
        sweep = section.get(RETUNE_SWEEP)
        points = sweep.get("points") if isinstance(sweep, Mapping) else None
        chosen = points[0] if isinstance(points, list) and len(points) == 1 else None
        if not isinstance(chosen, Mapping):
            raise _Unmapped(
                f"spec.parameters.adapters[{PLANNER_ADAPTER!r}].{RETUNE_SWEEP} "
                "does not name one planner point"
            )
        match = next(
            (option for option in shard.planners if _same_planner(chosen, option)),
            None,
        )
        if match is None:
            raise _Unmapped(f"planner point {dict(chosen)} is not one of the shard's")
        planner = dict(match)
        # The base and warmup are the Request's; only the point varies.
        rest_adapters[PLANNER_ADAPTER] = {
            key: value for key, value in section.items() if key != RETUNE_SWEEP
        }
    rest = dict(parameters)
    rest["adapters"] = rest_adapters
    signature = json.dumps(rest, sort_keys=True, separators=(",", ":"), default=str)
    return point, planner, signature


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
