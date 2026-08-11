# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast, overload

from typing_extensions import Unpack

from dynamo._core import MockEngineArgs
from dynamo._core import _OfflineReplaySession as _NativeOfflineReplaySession
from dynamo._core import (
    run_mocker_synthetic_trace_replay as _run_mocker_synthetic_trace_replay,
)
from dynamo._core import run_mocker_trace_replay as _run_mocker_trace_replay
from dynamo.replay.report import PlannerReplayDetails, ReplayReport


InteractiveRouter = Literal["external", "round_robin", "kv_router"]
TerminalStatus = Literal["completed", "rejected", "canceled", "failed"]


class ReplayStepStatus(TypedDict):
    status: Literal["advanced", "quiescent", "drained"]
    now_ms: float


class ReplayEventData(TypedDict):
    logical_request_id: str
    internal_uuid: str
    session_id: str
    authored_turn_index: int
    timestamp_ms: float
    worker_id: int | None
    dp_rank: int | None
    terminal_status: TerminalStatus | None
    input_length: int
    requested_output_length: int
    emitted_output_count: int
    reused_input_tokens: int | None
    ttft_ms: float | None
    e2e_latency_ms: float | None


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
    internal_uuid: str
    session_id: str
    authored_turn_index: int
    ready_at_ms: float


class ReplayWorkerSnapshot(TypedDict):
    worker_id: int
    dp_rank: int
    active: bool
    draining: bool
    in_flight_requests: int
    queued_requests: int | None
    queued_tokens: int | None
    running_tokens: int | None


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

    def to_native(self) -> dict[str, int]:
        return {"worker_id": self.worker_id, "dp_rank": self.dp_rank}


@dataclass(frozen=True, slots=True)
class ReplayRequestSpec:
    """Compact authored request admitted to an interactive replay session."""

    logical_request_id: str
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
    target: WorkerTarget | None = None

    def to_native(self) -> dict[str, Any]:
        return {
            "logical_request_id": self.logical_request_id,
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
            "target": None if self.target is None else self.target.to_native(),
        }


@dataclass(frozen=True, slots=True)
class ReplayAgenticRequest:
    """One request row and its causal DAG edge metadata."""

    request: ReplayRequestSpec | Mapping[str, Any]
    wait_for: Sequence[str] = ()
    dependency_delay_ms: float = 0.0
    prefix_reset: bool = False

    def to_native(self) -> dict[str, Any]:
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


def _target_payload(target: WorkerTarget | Mapping[str, int]) -> dict[str, int]:
    if isinstance(target, WorkerTarget):
        return target.to_native()
    return dict(target)


class OfflineReplaySession:
    """Drive one long-lived Dynamo offline replay on an explicit virtual clock.

    The session is synchronous and polling based. Placement decisions are made
    after draining ``placement_needed`` events and supplied explicitly through
    :meth:`assign`; the replay loop never invokes a Python callback.
    """

    def __init__(
        self,
        engine_args: MockEngineArgs,
        trace_block_size: int,
        num_workers: int = 1,
        router: InteractiveRouter = "external",
    ) -> None:
        self._native = _NativeOfflineReplaySession(
            engine_args=engine_args,
            trace_block_size=trace_block_size,
            num_workers=num_workers,
            router=router,
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
        target: WorkerTarget | Mapping[str, int],
    ) -> None:
        self._native.assign(logical_request_id, _target_payload(target))

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
