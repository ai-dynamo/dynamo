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
contract) expose those only *per worker*, inside
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
) -> Iterable["ForwardPassMetrics"]:
    """Yield the per-worker FPMs for the requested pool(s)."""
    if fpm is None:
        return
    pools = []
    if pool in ("prefill", "all"):
        pools.append(fpm.prefill)
    if pool in ("decode", "all"):
        pools.append(fpm.decode)
    for by_worker in pools:
        if by_worker:
            yield from by_worker.values()


def aggregate_queue_depth(fpm: Optional["FpmObservations"], pool: Pool = "all") -> int:
    """Total *waiting* (queued, not-yet-scheduled) requests across ``pool``.

    Sums each worker's queued request counts. For the prefill pool this is the
    backlog of admitted-but-unstarted prefills; for the decode pool it is
    preempted (evicted-to-waiting) decode requests. This is the simulator
    analogue of vLLM's ``num_requests_waiting``, the signal a KEDA-style
    queue-depth trigger scales on.
    """
    total = 0
    for m in _iter_pool(fpm, pool):
        q = m.queued_requests
        total += q.num_prefill_requests + q.num_decode_requests
    return total


def aggregate_queued_prefill_tokens(
    fpm: Optional["FpmObservations"], pool: Pool = "all"
) -> int:
    """Total queued prefill *tokens* across ``pool``.

    Token-weighted backlog (a single 100k-token prompt is far more work than
    100 one-token prompts). Useful for compute-bound prefill triggers where
    request count under-states the load.
    """
    total = 0
    for m in _iter_pool(fpm, pool):
        total += m.queued_requests.sum_prefill_tokens
    return total


def _pool_capacity(
    capabilities: Optional["WorkerCapabilities"], pool: Pool
) -> Optional[int]:
    """Per-worker ``max_kv_tokens`` for the pool, if known.

    For ``"all"`` the decode pool's capacity is used (aggregated topology has a
    single engine whose capabilities live under ``decode``).
    """
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

    Per worker: ``kv_tokens_in_use / max_kv_tokens``, then averaged over the
    workers that reported. Decode KV residency (``sum_decode_kv_tokens``) is the
    memory-pressure signal; prefill workers are scored on their in-flight
    prefill KV (``sum_prefill_kv_tokens``). This is the simulator analogue of
    vLLM's ``gpu_cache_usage_perc``.

    Returns ``None`` when capacity is unknown or no worker reported, so callers
    can distinguish "no datapoint" from a genuine ``0.0``.
    """
    capacity = _pool_capacity(capabilities, pool)
    if capacity is None:
        return None

    fractions: list[float] = []
    for m in _iter_pool(fpm, pool):
        sched = m.scheduled_requests
        # Decode KV dominates residency; include prefill KV so a prefill-only
        # pool isn't reported as perpetually empty.
        in_use = sched.sum_decode_kv_tokens + sched.sum_prefill_kv_tokens
        fractions.append(in_use / capacity)

    if not fractions:
        return None
    return sum(fractions) / len(fractions)


__all__ = [
    "Pool",
    "Role",
    "aggregate_queue_depth",
    "aggregate_queued_prefill_tokens",
    "aggregate_kv_util",
    "current_replica_target",
]
