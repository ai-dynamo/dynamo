# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, cast, overload

from typing_extensions import Unpack

from dynamo._core import MockEngineArgs
from dynamo._core import _OfflineReplaySession as _NativeOfflineReplaySession
from dynamo._core import _ReplayPoolSpec as _NativeReplayPoolSpec
from dynamo._core import _ReplayWorkerSpec as _NativeReplayWorkerSpec
from dynamo._core import (
    run_mocker_synthetic_trace_replay as _run_mocker_synthetic_trace_replay,
)
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay
from dynamo.replay.report import PlannerReplayDetails, ReplayReport


InteractiveRouter = Literal["external", "round_robin", "kv_router"]
PoolRouter = Literal["round_robin"]
TerminalStatus = Literal["completed", "rejected", "canceled", "failed"]
WorkerLifecycleStatus = Literal[
    "active", "static_inactive", "starting", "draining", "removed"
]


class ReplayStepStatus(TypedDict):
    status: Literal["advanced", "quiescent", "drained"]
    now_ms: float


class ReplayWorkerTargetData(TypedDict):
    pool_id: str
    worker_id: int
    dp_rank: int


class ReplayRoutingConstraintsData(TypedDict):
    required_taints: list[str]
    preferred_taints: dict[str, float]


class ReplayPlacementCandidate(TypedDict):
    target: ReplayWorkerTargetData
    active: bool
    draining: bool
    eligible: bool
    constraint_reason: str | None
    in_flight_requests: int
    queued_requests: int | None
    running_requests: int | None
    queued_tokens: int | None
    running_tokens: int | None
    max_num_seqs: int | None
    preemption_count: int | None
    kv_prefix_overlap_tokens: int | None
    kv_capacity_blocks: int | None
    kv_occupied_blocks: int | None
    kv_free_blocks: int | None
    tags: list[str]
    taints: list[str]
    capabilities: list[str]


class ReplayEventData(TypedDict):
    logical_request_id: str
    attempt_id: str
    group_id: str
    internal_uuid: str
    session_id: str
    authored_turn_index: int
    timestamp_ms: float
    pool_id: str | None
    worker_id: int | None
    dp_rank: int | None
    terminal_status: TerminalStatus | None
    input_length: int
    # Redacted at placement time so ordinary routing policies cannot inspect
    # future work. Terminal/final lifecycle records may populate it.
    requested_output_length: int | None
    emitted_output_count: int
    reused_input_tokens: int | None
    ttft_ms: float | None
    e2e_latency_ms: float | None
    priority: int
    strict_priority: int
    policy_class: str | None
    routing_constraints: ReplayRoutingConstraintsData
    eligible_pool_ids: list[str]
    candidates: list[ReplayPlacementCandidate]


class ReplayEvent(TypedDict):
    event_type: Literal[
        "placement_needed",
        "routed",
        "queued",
        "admitted",
        "first_token",
        "terminal",
    ]
    event: ReplayEventData


class ReplayPendingPlacement(TypedDict):
    logical_request_id: str
    attempt_id: str
    group_id: str
    internal_uuid: str
    session_id: str
    authored_turn_index: int
    ready_at_ms: float
    input_length: int
    priority: int
    strict_priority: int
    policy_class: str | None
    routing_constraints: ReplayRoutingConstraintsData
    eligible_pool_ids: list[str]
    candidates: list[ReplayPlacementCandidate]


class ReplayWorkerSnapshot(TypedDict):
    pool_id: str
    worker_id: int
    dp_rank: int
    lifecycle_status: WorkerLifecycleStatus
    provisioned: bool
    active: bool
    draining: bool
    in_flight_requests: int
    queued_requests: int | None
    running_requests: int | None
    queued_tokens: int | None
    running_tokens: int | None
    max_num_seqs: int | None
    preemption_count: int | None
    kv_capacity_blocks: int | None
    kv_occupied_blocks: int | None
    kv_free_blocks: int | None
    tags: list[str]
    taints: list[str]
    capabilities: list[str]


class ReplaySnapshot(TypedDict):
    now_ms: float
    admission_open: bool
    pending_request_count: int
    pending_placement_count: int
    workers: list[ReplayWorkerSnapshot]


@dataclass(frozen=True, slots=True)
class WorkerTarget:
    """Stable logical worker and attention-DP rank selected by a controller."""

    worker_id: int
    dp_rank: int = 0
    pool_id: str = "default"

    def to_native(self) -> dict[str, Any]:
        return {
            "pool_id": self.pool_id,
            "worker_id": self.worker_id,
            "dp_rank": self.dp_rank,
        }


@dataclass(frozen=True, slots=True)
class ReplayRoutingConstraints:
    """Request routing constraints for static replay.

    ``required_taints`` is a hard eligibility filter for external placement,
    pool round-robin, native round-robin, and native KV routing.
    ``preferred_taints`` is advisory scoring metadata: native KV routing and an
    external controller may use it, while round-robin does not turn it into an
    eligibility requirement. Taint names must be non-empty and already trimmed;
    required taints must be unique; preferred weights must be finite (negative
    weights are supported). Unsupported constraint keys fail schema parsing.
    """

    required_taints: Sequence[str] = ()
    preferred_taints: Mapping[str, float] = field(default_factory=dict)

    def to_native(self) -> dict[str, Any]:
        required = list(self.required_taints)
        if any(not name or name.strip() != name for name in required):
            raise ValueError("required taint names must be non-empty and trimmed")
        if len(set(required)) != len(required):
            raise ValueError("required taint names must be unique")
        preferred = dict(self.preferred_taints)
        if any(not name or name.strip() != name for name in preferred):
            raise ValueError("preferred taint names must be non-empty and trimmed")
        if any(not math.isfinite(weight) for weight in preferred.values()):
            raise ValueError("preferred taint weights must be finite")
        return {
            "required_taints": required,
            "preferred_taints": preferred,
        }


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    """One stable worker in a static replay pool.

    ``active=False`` and ``draining=False`` means provisioned static-inactive
    capacity: it is billed for the full session but never starts or becomes
    eligible for placement.

    ``taints`` participate in the routing contract. ``tags`` and
    ``capabilities`` are descriptive policy observations only; this milestone
    has no request-side tag/capability constraint fields.
    """

    worker_id: int
    max_num_seqs: int | None = None
    tags: Sequence[str] = ()
    taints: Sequence[str] = ()
    capabilities: Sequence[str] = ()
    active: bool = True
    draining: bool = False

    def to_native(self) -> Any:
        return _NativeReplayWorkerSpec(
            worker_id=self.worker_id,
            max_num_seqs=self.max_num_seqs,
            tags=list(self.tags),
            taints=list(self.taints),
            capabilities=list(self.capabilities),
            active=self.active,
            draining=self.draining,
        )


_WORKER_SPEC_FIELDS = frozenset(
    {
        "worker_id",
        "max_num_seqs",
        "tags",
        "taints",
        "capabilities",
        "active",
        "draining",
    }
)


def _worker_spec_native(spec: WorkerSpec | Mapping[str, Any]) -> Any:
    if isinstance(spec, WorkerSpec):
        return spec.to_native()
    payload = dict(spec)
    unknown_fields = payload.keys() - _WORKER_SPEC_FIELDS
    if unknown_fields:
        raise ValueError(
            "unknown replay worker topology fields: "
            + ", ".join(sorted(unknown_fields))
        )
    return _NativeReplayWorkerSpec(**payload)


@dataclass(frozen=True, slots=True)
class PoolSpec:
    """A static aggregated-vLLM pool with its own engine configuration."""

    pool_id: str
    engine_args: MockEngineArgs
    workers: Sequence[WorkerSpec | Mapping[str, Any]]
    router: PoolRouter = "round_robin"

    def to_native(self) -> Any:
        return _NativeReplayPoolSpec(
            pool_id=self.pool_id,
            engine_args=self.engine_args,
            workers=[_worker_spec_native(worker) for worker in self.workers],
            router=self.router,
        )


_POOL_SPEC_FIELDS = frozenset({"pool_id", "engine_args", "workers", "router"})


def _pool_spec_native(spec: PoolSpec | Mapping[str, Any]) -> Any:
    if isinstance(spec, PoolSpec):
        return spec.to_native()
    payload = dict(spec)
    unknown_fields = payload.keys() - _POOL_SPEC_FIELDS
    if unknown_fields:
        raise ValueError(
            "unknown replay pool topology fields: "
            + ", ".join(sorted(unknown_fields))
        )
    workers = payload.get("workers")
    if workers is None:
        raise ValueError("replay pool topology requires workers")
    payload["workers"] = [_worker_spec_native(worker) for worker in workers]
    return _NativeReplayPoolSpec(**payload)


@dataclass(frozen=True, slots=True)
class ReplayRequestSpec:
    """Compact authored request admitted to an interactive replay session."""

    logical_request_id: str
    attempt_id: str
    group_id: str
    session_id: str
    authored_turn_index: int
    input_length: int
    hash_ids: Sequence[int]
    trace_block_size: int
    output_length: int
    ready_time_ms: float = 0.0
    internal_uuid: str | None = None
    output_token_ids: Sequence[int] | None = None
    priority: int = 0
    strict_priority: int = 0
    policy_class: str | None = None
    routing_constraints: ReplayRoutingConstraints = field(
        default_factory=ReplayRoutingConstraints
    )
    target: WorkerTarget | None = None

    def to_native(self) -> dict[str, Any]:
        return {
            "logical_request_id": self.logical_request_id,
            "attempt_id": self.attempt_id,
            "group_id": self.group_id,
            "internal_uuid": self.internal_uuid,
            "session_id": self.session_id,
            "authored_turn_index": self.authored_turn_index,
            "ready_time_ms": self.ready_time_ms,
            "input_length": self.input_length,
            "hash_ids": list(self.hash_ids),
            "trace_block_size": self.trace_block_size,
            "output_length": self.output_length,
            "output_token_ids": (
                None if self.output_token_ids is None else list(self.output_token_ids)
            ),
            "priority": self.priority,
            "strict_priority": self.strict_priority,
            "policy_class": self.policy_class,
            "routing_constraints": self.routing_constraints.to_native(),
            "target": None if self.target is None else self.target.to_native(),
        }


@dataclass(frozen=True, slots=True)
class ReplayAgenticRequest:
    """One request row and its causal DAG edge metadata.

    ``prefix_reset=True`` is unsupported by the static replay kernel and fails
    closed instead of being treated as reporting-only metadata.
    """

    request: ReplayRequestSpec | Mapping[str, Any]
    wait_for: Sequence[str] = ()
    dependency_delay_ms: float = 0.0
    prefix_reset: bool = False

    def to_native(self) -> dict[str, Any]:
        if self.prefix_reset:
            raise ValueError("Dynamo Replay does not support prefix_reset=true")
        request = (
            self.request.to_native()
            if isinstance(self.request, ReplayRequestSpec)
            else dict(self.request)
        )
        return {
            "request": request,
            "wait_for": list(self.wait_for),
            "dependency_delay_ms": self.dependency_delay_ms,
            "prefix_reset": self.prefix_reset,
        }


@dataclass(frozen=True, slots=True)
class ReplayAgenticWorkflow:
    """An appendable independent request DAG using one bundle block namespace."""

    trace_block_size: int
    requests: Sequence[ReplayAgenticRequest | Mapping[str, Any]]

    def to_native(self) -> dict[str, Any]:
        return {
            "trace_block_size": self.trace_block_size,
            "requests": [
                request.to_native()
                if isinstance(request, ReplayAgenticRequest)
                else dict(request)
                for request in self.requests
            ],
        }


class _CommonReplayOptions(TypedDict, total=False):
    extra_engine_args: Any
    prefill_engine_args: Any
    decode_engine_args: Any
    router_config: Any
    aic_perf_config: Any
    num_workers: int
    num_prefill_workers: int
    num_decode_workers: int
    replay_concurrency: int | None
    router_mode: Literal["round_robin", "kv_router"]
    arrival_speedup_ratio: float
    model_name: str | None
    sla_ttft_ms: float | None
    sla_itl_ms: float | None
    sla_e2e_ms: float | None
    planner_config: Any
    benchmark_granularity: int
    capture_per_request: bool
    capture_planner_details: bool


class _TraceReplayOptions(_CommonReplayOptions, total=False):
    trace_block_size: int | None
    trace_format: str
    trace_shared_prefix_ratio: float
    trace_num_prefix_groups: int
    report_jsonl_path: str | os.PathLike[str] | None
    max_sim_time_ms: float | None


class _SyntheticReplayOptions(_CommonReplayOptions, total=False):
    request_rate: float | None
    arrival_interval_ms: float | None
    arrival_seed: int
    turns_per_session: int
    shared_prefix_ratio: float
    num_prefix_groups: int
    inter_turn_delay_ms: float


def _normalize_trace_files(trace_files):
    if isinstance(trace_files, (str, os.PathLike)):
        return [trace_files]
    return list(trace_files)


def _planner_config_arg(planner_config):
    """Normalize a planner config to the JSON-string form ``_prepare_planner_replay``
    expects: a dict is json-encoded; a str (path or inline JSON) passes through."""
    if isinstance(planner_config, dict):
        return json.dumps(planner_config)
    return planner_config


def _materialize_offline_report(
    native,
    *,
    planner: PlannerReplayDetails | None,
) -> ReplayReport:
    return ReplayReport(
        summary=native.summary,
        per_request=native.per_request,
        coverage=native.coverage,
        planner=planner,
    )


def _request_payload(
    request: ReplayRequestSpec | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(request, ReplayRequestSpec):
        return request.to_native()
    return dict(request)


def _workflow_payload(
    workflow: ReplayAgenticWorkflow | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(workflow, ReplayAgenticWorkflow):
        return workflow.to_native()
    return dict(workflow)


def _target_payload(target: WorkerTarget | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(target, WorkerTarget):
        return target.to_native()
    return dict(target)


class OfflineReplaySession:
    """Drive one long-lived Dynamo offline replay on an explicit virtual clock.

    The session is synchronous and polling based. Placement decisions are made
    after draining a ``placement_needed`` event (or reading
    :meth:`pending_placements`) and supplied explicitly through :meth:`assign`.
    Exactly one external placement boundary is exposed at a time so the next
    same-time observation includes the prior assignment's scheduler effects;
    the replay loop never invokes a Python callback.
    """

    def __init__(
        self,
        engine_args: MockEngineArgs | None = None,
        trace_block_size: int | None = None,
        num_workers: int = 1,
        router: InteractiveRouter = "external",
        *,
        pools: Sequence[PoolSpec | Mapping[str, Any]] | None = None,
        session_affinity: bool = False,
    ) -> None:
        if trace_block_size is None:
            raise ValueError("interactive replay requires trace_block_size")
        if pools is not None:
            if engine_args is not None:
                raise ValueError(
                    "interactive replay accepts either engine_args or pools, not both"
                )
            if num_workers != 1:
                raise ValueError(
                    "num_workers is configured by WorkerSpec when pools are supplied"
                )
            self._native = _NativeOfflineReplaySession.from_pools(
                pools=[_pool_spec_native(pool) for pool in pools],
                trace_block_size=trace_block_size,
                router=router,
                session_affinity=session_affinity,
            )
            return
        if engine_args is None:
            raise ValueError("interactive replay requires engine_args or pools")
        self._native = _NativeOfflineReplaySession(
            engine_args=engine_args,
            trace_block_size=trace_block_size,
            num_workers=num_workers,
            router=router,
            session_affinity=session_affinity,
        )

    def submit(self, request: ReplayRequestSpec | Mapping[str, Any]) -> None:
        """Submit one independent request without predicting its completion."""
        self._native.submit(_request_payload(request))

    def append_agentic_workflow(
        self,
        workflow: ReplayAgenticWorkflow | Mapping[str, Any],
        release_at_ms: float,
    ) -> None:
        """Append an independent causal workflow at a controller-selected time."""
        self._native.append_agentic_workflow(
            _workflow_payload(workflow),
            release_at_ms,
        )

    def now_ms(self) -> float:
        return self._native.now_ms()

    def next_event_time_ms(self) -> float | None:
        return self._native.next_event_time_ms()

    def advance_next(self) -> ReplayStepStatus:
        return cast(ReplayStepStatus, self._native.advance_next())

    def advance_to(self, target_ms: float) -> ReplayStepStatus:
        return cast(ReplayStepStatus, self._native.advance_to(target_ms))

    def settle_current_time(self) -> ReplayStepStatus:
        return cast(ReplayStepStatus, self._native.settle_current_time())

    def drain_events(self) -> list[ReplayEvent]:
        return cast(list[ReplayEvent], self._native.drain_events())

    def pending_placements(self) -> list[ReplayPendingPlacement]:
        return cast(
            list[ReplayPendingPlacement],
            self._native.pending_placements(),
        )

    def assign(
        self,
        logical_request_id: str,
        target: WorkerTarget | Mapping[str, Any],
    ) -> None:
        self._native.assign(logical_request_id, _target_payload(target))

    def assign_pool(self, logical_request_id: str, pool_id: str) -> None:
        """Assign pending work to a pool's deterministic internal router."""
        self._native.assign_pool(logical_request_id, pool_id)

    def snapshot(self) -> ReplaySnapshot:
        return cast(ReplaySnapshot, self._native.snapshot())

    def close_admission(self) -> None:
        self._native.close_admission()

    def close(self) -> None:
        """Alias used by service adapters when no more work will be appended."""
        self.close_admission()

    def is_quiescent(self) -> bool:
        return self._native.is_quiescent()

    def is_drained(self) -> bool:
        return self._native.is_drained()

    def finalize(self) -> ReplayReport:
        """Consume a closed, drained replay and return its retained report."""
        return _materialize_offline_report(
            self._native.finalize(),
            planner=None,
        )


@overload
def run_trace_replay(
    trace_files,
    *,
    replay_mode: Literal["offline"] = "offline",
    **kwargs: Unpack[_TraceReplayOptions],
) -> ReplayReport:
    ...


@overload
def run_trace_replay(
    trace_files,
    *,
    replay_mode: Literal["online"],
    **kwargs: Unpack[_TraceReplayOptions],
) -> dict[str, Any]:
    ...


@overload
def run_trace_replay(
    trace_files,
    *,
    replay_mode: str,
    **kwargs: Unpack[_TraceReplayOptions],
) -> ReplayReport | dict[str, Any]:
    ...


def run_trace_replay(
    trace_files,
    *,
    extra_engine_args=None,
    prefill_engine_args=None,
    decode_engine_args=None,
    router_config=None,
    aic_perf_config=None,
    num_workers=1,
    num_prefill_workers=1,
    num_decode_workers=1,
    replay_concurrency=None,
    replay_mode="offline",
    router_mode="round_robin",
    arrival_speedup_ratio=1.0,
    trace_block_size=None,
    trace_format="mooncake",
    trace_shared_prefix_ratio=0.0,
    trace_num_prefix_groups=0,
    report_jsonl_path=None,
    max_sim_time_ms=None,
    model_name=None,
    sla_ttft_ms=None,
    sla_itl_ms=None,
    sla_e2e_ms=None,
    planner_config=None,
    benchmark_granularity=8,
    capture_per_request=False,
    capture_planner_details=True,
) -> ReplayReport | dict[str, Any]:
    """Run trace replay.

    ``wall_time_ms`` and derived throughput measure Rust runtime construction
    and execution. Planner creation and bootstrap happen before that boundary.
    """
    trace_files = _normalize_trace_files(trace_files)
    replay_kwargs = {
        "extra_engine_args": extra_engine_args,
        "prefill_engine_args": prefill_engine_args,
        "decode_engine_args": decode_engine_args,
        "router_config": router_config,
        "aic_perf_config": aic_perf_config,
        "num_workers": num_workers,
        "num_prefill_workers": num_prefill_workers,
        "num_decode_workers": num_decode_workers,
        "replay_concurrency": replay_concurrency,
        "replay_mode": replay_mode,
        "router_mode": router_mode,
        "arrival_speedup_ratio": arrival_speedup_ratio,
        "trace_block_size": trace_block_size,
        "trace_format": trace_format,
        "trace_shared_prefix_ratio": trace_shared_prefix_ratio,
        "trace_num_prefix_groups": trace_num_prefix_groups,
        "report_jsonl_path": report_jsonl_path,
        "max_sim_time_ms": max_sim_time_ms,
        "model_name": model_name,
        "sla_ttft_ms": sla_ttft_ms,
        "sla_itl_ms": sla_itl_ms,
        "sla_e2e_ms": sla_e2e_ms,
        "capture_per_request": capture_per_request,
        "capture_planner_details": capture_planner_details,
    }
    if capture_per_request and replay_mode == "online":
        raise ValueError(
            "capture_per_request only supports replay_mode='offline'; "
            "use report_jsonl_path for online request records"
        )
    if planner_config is not None:
        # Planner replay is offline-only; reject controls the
        # planner path ignores so callers fail fast instead of silently getting an
        # offline planner run (matches the CLI's guardrails).
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        if trace_format not in ("mooncake", "dynamo"):
            raise ValueError(
                "planner_config replay only supports trace_format='mooncake' or 'dynamo'"
            )
        if max_sim_time_ms is not None:
            raise ValueError("max_sim_time_ms is not supported with planner_config")
        if trace_format != "dynamo" and len(trace_files) != 1:
            raise ValueError(
                f"planner_config replay with trace_format={trace_format!r} "
                "requires exactly one trace file"
            )
        if trace_format == "dynamo" and not trace_files:
            raise ValueError(
                "planner_config replay with trace_format='dynamo' "
                "requires at least one trace file"
            )
        from dynamo.replay.main import _planner_replay_adapter

        adapter_scope = _planner_replay_adapter(
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            planner_config_arg=_planner_config_arg(planner_config),
            benchmark_granularity=benchmark_granularity,
            capture_details=capture_planner_details,
        )
        with adapter_scope as adapter:
            native = _run_mocker_trace_replay(
                trace_files,
                **replay_kwargs,
                scaling_policy=adapter,
            )
            return _materialize_offline_report(
                native,
                planner=adapter.finalize(native.lifecycle_operations),
            )
    result = _run_mocker_trace_replay(
        trace_files,
        **replay_kwargs,
        scaling_policy=None,
    )
    if replay_mode == "online":
        return result
    return _materialize_offline_report(
        result,
        planner=None,
    )


@overload
def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    replay_mode: Literal["offline"] = "offline",
    **kwargs: Unpack[_SyntheticReplayOptions],
) -> ReplayReport:
    ...


@overload
def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    replay_mode: Literal["online"],
    **kwargs: Unpack[_SyntheticReplayOptions],
) -> dict[str, Any]:
    ...


@overload
def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    replay_mode: str,
    **kwargs: Unpack[_SyntheticReplayOptions],
) -> ReplayReport | dict[str, Any]:
    ...


def run_synthetic_trace_replay(
    input_tokens,
    output_tokens,
    request_count,
    *,
    extra_engine_args=None,
    prefill_engine_args=None,
    decode_engine_args=None,
    router_config=None,
    aic_perf_config=None,
    num_workers=1,
    num_prefill_workers=1,
    num_decode_workers=1,
    replay_concurrency=None,
    replay_mode="offline",
    router_mode="round_robin",
    arrival_speedup_ratio=1.0,
    request_rate=None,
    arrival_interval_ms=None,
    arrival_seed=42,
    turns_per_session=1,
    shared_prefix_ratio=0.0,
    num_prefix_groups=0,
    inter_turn_delay_ms=0.0,
    model_name=None,
    sla_ttft_ms=None,
    sla_itl_ms=None,
    sla_e2e_ms=None,
    planner_config=None,
    benchmark_granularity=8,
    capture_per_request=False,
    capture_planner_details=True,
) -> ReplayReport | dict[str, Any]:
    """Run synthetic replay with the same timing boundary as trace replay."""
    replay_kwargs = {
        "extra_engine_args": extra_engine_args,
        "prefill_engine_args": prefill_engine_args,
        "decode_engine_args": decode_engine_args,
        "router_config": router_config,
        "aic_perf_config": aic_perf_config,
        "num_workers": num_workers,
        "num_prefill_workers": num_prefill_workers,
        "num_decode_workers": num_decode_workers,
        "replay_concurrency": replay_concurrency,
        "replay_mode": replay_mode,
        "router_mode": router_mode,
        "arrival_speedup_ratio": arrival_speedup_ratio,
        "request_rate": request_rate,
        "arrival_interval_ms": arrival_interval_ms,
        "arrival_seed": arrival_seed,
        "turns_per_session": turns_per_session,
        "shared_prefix_ratio": shared_prefix_ratio,
        "num_prefix_groups": num_prefix_groups,
        "inter_turn_delay_ms": inter_turn_delay_ms,
        "model_name": model_name,
        "sla_ttft_ms": sla_ttft_ms,
        "sla_itl_ms": sla_itl_ms,
        "sla_e2e_ms": sla_e2e_ms,
        "capture_per_request": capture_per_request,
        "capture_planner_details": capture_planner_details,
    }
    if capture_per_request and replay_mode == "online":
        raise ValueError("capture_per_request only supports replay_mode='offline'")
    if planner_config is not None:
        if replay_mode != "offline":
            raise ValueError(
                "planner_config replay only supports replay_mode='offline'"
            )
        from dynamo.replay.main import _planner_replay_adapter

        adapter_scope = _planner_replay_adapter(
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            planner_config_arg=_planner_config_arg(planner_config),
            benchmark_granularity=benchmark_granularity,
            capture_details=capture_planner_details,
        )
        with adapter_scope as adapter:
            native = _run_mocker_synthetic_trace_replay(
                input_tokens,
                output_tokens,
                request_count,
                **replay_kwargs,
                scaling_policy=adapter,
            )
            return _materialize_offline_report(
                native,
                planner=adapter.finalize(native.lifecycle_operations),
            )
    result = _run_mocker_synthetic_trace_replay(
        input_tokens,
        output_tokens,
        request_count,
        **replay_kwargs,
        scaling_policy=None,
    )
    if replay_mode == "online":
        return result
    return _materialize_offline_report(
        result,
        planner=None,
    )
