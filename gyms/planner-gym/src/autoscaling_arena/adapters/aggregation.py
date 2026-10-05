# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fleet-observation helpers for Arena rival autoscalers.

Lives in the Autoscaling Arena (not in ``dynamo.planner.core``): these
helpers are consumed only by the Arena's rival adapters, so they belong
here rather than in the Dynamo planner core, which never uses them. Only
dynamo *types* are referenced (under ``TYPE_CHECKING``).

Rival autoscaler adapters (KEDA queue-depth, Ray Serve, llm-d, reactive
thresholds) scale on *fleet-level* scalar signals — total requests waiting,
mean KV-cache utilization. The simulation (and the planner's observation
contract) expose those per (worker, data-parallel rank), inside
:class:`~dynamo.planner.core.types.FpmObservations`. These helpers collapse the
per-worker FPM view into the scalar signals rival autoscalers consume, so every
adapter derives them the same way rather than each re-implementing the reduction
(and disagreeing on it).

Absolute scaling targets use the lifecycle-aware ``expected_num_*`` worker
counts (active + starting), while utilization and queue signals continue to
come from ready workers' FPMs.

The ``pool`` argument selects which engine pool to read:

* ``"prefill"`` / ``"decode"`` — a single disaggregated pool.
* ``"all"`` — both pools combined (the natural choice for aggregated topology,
  where only the decode pool is populated anyway).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Literal, Optional

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics
    from dynamo.planner.core.types import (
        FpmObservations,
        WorkerCapabilities,
        WorkerCounts,
    )

Pool = Literal["prefill", "decode", "all"]
Role = Literal["prefill", "decode"]


def current_replica_target(
    worker_counts: Optional["WorkerCounts"],
    *,
    role: Role,
    minimum: int,
) -> int:
    """Return the absolute-target baseline for a replay scaling decision.

    Dynamo applies absolute targets against the non-draining fleet
    (active + starting). ``expected_num_*`` carries that value, including any
    scale-up still in flight. Fall back to ready workers only when the expected
    count is unavailable, then enforce the policy's replica floor.
    """

    if worker_counts is None:
        return minimum
    if role == "prefill":
        expected = worker_counts.expected_num_prefill
        ready = worker_counts.ready_num_prefill
    else:
        expected = worker_counts.expected_num_decode
        ready = worker_counts.ready_num_decode
    if expected is not None:
        return max(expected, minimum)
    if ready is not None:
        return max(ready, minimum)
    return minimum


def _iter_pool(
    fpm: Optional["FpmObservations"], pool: Pool
) -> Iterable[tuple[Role, str, "ForwardPassMetrics"]]:
    """Yield each rank's pool, worker ID, and FPM for the requested pool(s)."""
    if fpm is None:
        return
    pools = []
    if pool in ("prefill", "all"):
        pools.append(("prefill", fpm.prefill))
    if pool in ("decode", "all"):
        pools.append(("decode", fpm.decode))
    for role, by_worker in pools:
        if by_worker:
            for (worker_id, _rank), metric in by_worker.items():
                yield role, worker_id, metric


def aggregate_queue_depth(
    fpm: Optional["FpmObservations"], pool: Pool = "all"
) -> Optional[int]:
    """Total *waiting* (queued, not-yet-scheduled) requests across ``pool``.

    Sums each worker's queued request counts. For the prefill pool this is the
    backlog of admitted-but-unstarted prefills; for the decode pool it is
    preempted (evicted-to-waiting) decode requests. This is the simulator
    analogue of vLLM's ``num_requests_waiting``, the signal a KEDA-style
    queue-depth trigger scales on. Returns ``None`` when no rank in the
    selected pool reported; an observed idle heartbeat instead returns zero.
    """
    total = 0
    reported = False
    for _, _, metric in _iter_pool(fpm, pool):
        reported = True
        queued = metric.queued_requests
        total += queued.num_prefill_requests + queued.num_decode_requests
    return total if reported else None


def aggregate_queued_prefill_tokens(
    fpm: Optional["FpmObservations"], pool: Pool = "all"
) -> int:
    """Total queued prefill *tokens* across ``pool``.

    Token-weighted backlog (a single 100k-token prompt is far more work than
    100 one-token prompts). Useful for compute-bound prefill triggers where
    request count under-states the load.
    """
    total = 0
    for _, _, metric in _iter_pool(fpm, pool):
        total += metric.queued_requests.sum_prefill_tokens
    return total


def _pool_capacity(
    capabilities: Optional["WorkerCapabilities"], pool: Role
) -> Optional[int]:
    """Per-worker ``max_kv_tokens`` for one engine pool, if known."""
    if capabilities is None:
        return None
    eng = capabilities.prefill if pool == "prefill" else capabilities.decode
    if eng is None or not eng.max_kv_tokens:
        return None
    return eng.max_kv_tokens


def aggregate_kv_util(
    fpm: Optional["FpmObservations"],
    capabilities: Optional["WorkerCapabilities"],
    pool: Pool = "decode",
) -> Optional[float]:
    """Mean KV-cache utilization (fraction in ``[0, 1]``) across ``pool``.

    Sum each worker's rank-level KV token usage, divide by that pool's
    per-worker ``max_kv_tokens`` (which already includes every DP rank), then
    average over the workers that reported. Decode KV residency
    (``sum_decode_kv_tokens``) is the memory-pressure signal; prefill workers are scored on their in-flight
    prefill KV (``sum_prefill_kv_tokens``). This is the simulator analogue of
    vLLM's ``gpu_cache_usage_perc``.

    Returns ``None`` when capacity is unknown or no worker reported, so callers
    can distinguish "no datapoint" from a genuine ``0.0``.
    """
    used_by_worker: dict[tuple[Role, str], int] = {}
    for role, worker_id, metric in _iter_pool(fpm, pool):
        scheduled = metric.scheduled_requests
        # Worker IDs belong to a pool: do not combine prefill and decode rows
        # merely because their IDs match when both pools are requested.
        key = (role, worker_id)
        used_by_worker[key] = (
            used_by_worker.get(key, 0)
            + scheduled.sum_decode_kv_tokens
            + scheduled.sum_prefill_kv_tokens
        )

    if not used_by_worker:
        return None
    fractions: list[float] = []
    for (role, _worker_id), in_use in used_by_worker.items():
        capacity = _pool_capacity(capabilities, role)
        if capacity is None:
            return None
        fractions.append(in_use / capacity)
    return sum(fractions) / len(fractions)


__all__ = [
    "Pool",
    "Role",
    "aggregate_queue_depth",
    "aggregate_queued_prefill_tokens",
    "aggregate_kv_util",
    "current_replica_target",
]
