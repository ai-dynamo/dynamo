# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multi-deployment offline Replay control plane.

The Rust Replay world owns virtual-time data-plane progress.  This module owns
the Python planner state for that world: one local planner session per
deployment, one shared Global Planner orchestrator, and an in-memory capacity
backend that stages approved actions for Rust to apply after the callback.
"""

from __future__ import annotations

import asyncio
import json
import math
import operator
from dataclasses import dataclass, field
from os import PathLike
from typing import Any, Mapping, Optional

from dynamo.global_planner.capacity_manager import PoolSpec
from dynamo.global_planner.orchestrator import Orchestrator
from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.connectors.protocol import ScaleStatus
from dynamo.planner.offline.replay_adapter import (
    ReplayPlannerAdapter,
    ReplayPlannerReport,
    ScalingProposal,
)
from dynamo.planner.plugins.clock import VirtualClock
from dynamo.replay.global_planner import (
    ReplayCapacityManager,
    ReplayParticipantSpec,
    ReplayScaleAction,
)


def _require_integer(value: object, *, name: str, minimum: int) -> int:
    """Validate an integer-valued public configuration field.

    ``bool`` is intentionally rejected even though it subclasses ``int``:
    accepting ``True`` as one worker is almost always a configuration typo.
    """

    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        normalized = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if normalized < minimum:
        qualifier = "positive" if minimum == 1 else f"at least {minimum}"
        raise ValueError(f"{name} must be {qualifier}")
    return normalized


@dataclass(frozen=True)
class ReplayGlobalPlannerConfig:
    """Global Planner policy for one Replay world."""

    max_total_gpus: int = -1
    min_total_gpus: int = -1
    intent_cache_ttl_seconds: float = 360.0

    def __post_init__(self) -> None:
        max_total_gpus = _require_integer(
            self.max_total_gpus, name="max_total_gpus", minimum=-1
        )
        min_total_gpus = _require_integer(
            self.min_total_gpus, name="min_total_gpus", minimum=-1
        )
        if (
            max_total_gpus >= 0
            and min_total_gpus >= 0
            and min_total_gpus > max_total_gpus
        ):
            raise ValueError("min_total_gpus cannot exceed max_total_gpus")
        if (
            not math.isfinite(self.intent_cache_ttl_seconds)
            or self.intent_cache_ttl_seconds < 0
        ):
            raise ValueError("intent_cache_ttl_seconds must be finite and non-negative")


@dataclass(frozen=True)
class ReplayTraceWorkload:
    """One deployment-local Mooncake trace workload."""

    trace_file: str | PathLike[str]
    arrival_speedup_ratio: float = 1.0
    trace_block_size: int = 512

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.arrival_speedup_ratio)
            or self.arrival_speedup_ratio <= 0
        ):
            raise ValueError("arrival_speedup_ratio must be finite and positive")
        _require_integer(self.trace_block_size, name="trace_block_size", minimum=1)


@dataclass(frozen=True)
class ReplaySyntheticWorkload:
    """One deployment-local deterministic synthetic workload distribution."""

    input_tokens: int
    output_tokens: int
    request_count: int
    arrival_interval_ms: float = 1.0
    turns_per_session: int = 1
    shared_prefix_ratio: float = 0.0
    num_prefix_groups: int = 0
    inter_turn_delay_ms: float = 0.0
    arrival_speedup_ratio: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("input_tokens", self.input_tokens),
            ("output_tokens", self.output_tokens),
            ("request_count", self.request_count),
            ("turns_per_session", self.turns_per_session),
        ):
            _require_integer(value, name=name, minimum=1)
        _require_integer(self.num_prefix_groups, name="num_prefix_groups", minimum=0)
        if not math.isfinite(self.arrival_interval_ms) or self.arrival_interval_ms < 0:
            raise ValueError("arrival_interval_ms must be finite and non-negative")
        if (
            not math.isfinite(self.arrival_speedup_ratio)
            or self.arrival_speedup_ratio <= 0
        ):
            raise ValueError("arrival_speedup_ratio must be finite and positive")


@dataclass(frozen=True)
class ReplayDeploymentConfig:
    """Configuration for one independent serving deployment in a Replay world.

    ``planner_config`` accepts the same inline JSON dictionary, JSON string, or
    file path as the existing single-deployment Replay API.  Engine/router
    objects are the existing Python binding types.
    """

    deployment_id: str
    planner_config: dict[str, Any] | str
    workload: ReplayTraceWorkload | ReplaySyntheticWorkload
    model_name: Optional[str] = None
    extra_engine_args: Any = None
    prefill_engine_args: Any = None
    decode_engine_args: Any = None
    router_config: Any = None
    num_workers: int = 1
    num_prefill_workers: int = 1
    num_decode_workers: int = 1
    replay_concurrency: Optional[int] = None
    router_mode: str = "round_robin"
    benchmark_granularity: int = 8
    sla_ttft_ms: Optional[float] = None
    sla_itl_ms: Optional[float] = None
    sla_e2e_ms: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.deployment_id, str) or not self.deployment_id.strip():
            raise ValueError("deployment_id must be non-empty")
        for name, value in (
            ("num_workers", self.num_workers),
            ("num_prefill_workers", self.num_prefill_workers),
            ("num_decode_workers", self.num_decode_workers),
        ):
            _require_integer(value, name=name, minimum=0)
        if self.replay_concurrency is not None:
            _require_integer(
                self.replay_concurrency, name="replay_concurrency", minimum=1
            )
        _require_integer(
            self.benchmark_granularity,
            name="benchmark_granularity",
            minimum=1,
        )


def _validate_deployment_engine_args(
    deployment: ReplayDeploymentConfig,
    planner_mode: str,
) -> None:
    """Reject engine arguments that would otherwise be silently ignored."""

    if planner_mode == "agg":
        if (
            deployment.prefill_engine_args is not None
            or deployment.decode_engine_args is not None
        ):
            raise ValueError(
                f"aggregated deployment {deployment.deployment_id!r} accepts only "
                "extra_engine_args"
            )
        return
    if planner_mode == "disagg":
        if deployment.extra_engine_args is not None:
            raise ValueError(
                f"disaggregated deployment {deployment.deployment_id!r} does not "
                "accept extra_engine_args"
            )
        if (
            deployment.prefill_engine_args is None
            or deployment.decode_engine_args is None
        ):
            raise ValueError(
                f"disaggregated deployment {deployment.deployment_id!r} requires "
                "prefill_engine_args and decode_engine_args"
            )
        return
    raise ValueError(f"unsupported planner mode {planner_mode!r}")


@dataclass(frozen=True)
class GlobalPlannerReplayEvent:
    """One local proposal's Global Planner outcome."""

    at_s: float
    participant_id: str
    status: str
    target_prefill: Optional[int]
    target_decode: Optional[int]
    message: str = ""


@dataclass
class ReplayWorldReport:
    """Planner-enriched reports and allocation decisions for a Replay world."""

    deployments: dict[str, ReplayPlannerReport]
    global_planner_events: list[GlobalPlannerReplayEvent] = field(default_factory=list)
    duration_ms: float = 0.0
    wall_time_ms: float = 0.0


class ReplayWorldPlannerController:
    """Python callback used by the Rust multi-deployment Replay world.

    The callback is intentionally timestamp-batched.  Every due local planner
    first produces a proposal from its frozen metrics; only then are proposals
    serialized through the shared Global Planner.  Approved capacity writes are
    staged in :class:`ReplayCapacityManager` and returned as owned values, so
    Python never re-enters a mutably borrowed Rust world.
    """

    def __init__(
        self,
        sessions: Mapping[str, ReplayPlannerAdapter],
        participants: list[ReplayParticipantSpec],
        *,
        config: ReplayGlobalPlannerConfig = ReplayGlobalPlannerConfig(),
        event_loop: asyncio.AbstractEventLoop,
        clock: VirtualClock,
    ) -> None:
        if not sessions:
            raise ValueError("a Replay world requires at least one deployment")
        if event_loop.is_running():
            raise ValueError("Replay world event loop must not already be running")

        self._sessions = dict(sessions)
        if any(not participant_id for participant_id in self._sessions):
            raise ValueError("deployment IDs must be non-empty")

        participant_ids = {spec.participant_id for spec in participants}
        session_ids = set(self._sessions)
        if participant_ids != session_ids:
            missing_sessions = sorted(participant_ids - session_ids)
            missing_participants = sorted(session_ids - participant_ids)
            raise ValueError(
                "planner sessions and capacity participants must match exactly "
                f"(missing_sessions={missing_sessions}, "
                f"missing_participants={missing_participants})"
            )

        self._loop: Optional[asyncio.AbstractEventLoop] = event_loop
        self._clock = clock
        self._capacity_manager = ReplayCapacityManager(participants)
        self._orchestrator = Orchestrator(
            self._capacity_manager,
            managed_deployments=sorted(session_ids),
            max_total_gpus=config.max_total_gpus,
            min_total_gpus=config.min_total_gpus,
            intent_cache_ttl_seconds=config.intent_cache_ttl_seconds,
            now=self._clock.now,
            use_lock=False,
        )
        for participant_id in sorted(session_ids):
            self._orchestrator.register(
                participant_id,
                caller_name=participant_id,
                namespace="",
                deployment_name=participant_id,
            )
        self._orchestrator.startup()

        self._started: set[str] = set()
        self._global_planner_events: list[GlobalPlannerReplayEvent] = []
        self._closed = False

    def _run_sync(self, coro):
        loop = self._loop
        if loop is None:
            raise RuntimeError("Replay world planner controller is closed")
        return loop.run_until_complete(coro)

    def initial_tick_ms(self, deployment_id: str) -> float:
        """Return one deployment's first local-planner deadline."""
        if deployment_id not in self._sessions:
            raise KeyError(f"unknown Replay deployment {deployment_id!r}")
        if deployment_id in self._started:
            raise RuntimeError(
                f"initial planner tick requested twice for {deployment_id!r}"
            )
        first_tick_ms = self._run_sync(
            self._sessions[deployment_id].initial_tick_ms_async()
        )
        self._started.add(deployment_id)
        return first_tick_ms

    def on_ticks(self, ticks: list[dict[str, Any]]) -> dict[str, Any]:
        """Process one complete world planner barrier."""
        return self._run_sync(self._on_ticks_async(ticks))

    async def _on_ticks_async(self, ticks: list[dict[str, Any]]) -> dict[str, Any]:
        if not ticks:
            raise ValueError("world planner callback requires at least one due tick")

        by_id: dict[str, dict[str, Any]] = {}
        now_ms: Optional[float] = None
        for tick in ticks:
            participant_id = str(tick["deployment_id"])
            if participant_id not in self._sessions:
                raise KeyError(f"unknown Replay deployment {participant_id!r}")
            if participant_id in by_id:
                raise ValueError(
                    f"deployment {participant_id!r} fired twice at one timestamp"
                )
            tick_now_ms = float(tick["now_ms"])
            if not math.isfinite(tick_now_ms):
                raise ValueError(f"planner timestamp must be finite, got {tick_now_ms}")
            if now_ms is None:
                now_ms = tick_now_ms
            elif tick_now_ms != now_ms:
                raise ValueError(
                    "one world callback must contain exactly one timestamp "
                    f"(saw {now_ms} and {tick_now_ms})"
                )
            by_id[participant_id] = tick

        assert now_ms is not None
        now_s = now_ms / 1000.0
        delta_s = now_s - self._clock.monotonic()
        if delta_s < 0:
            raise ValueError(
                f"Replay world clock cannot move backwards "
                f"({self._clock.monotonic()} -> {now_s})"
            )
        if delta_s > 0:
            self._clock.advance(delta_s)

        # Freeze the local control-plane phase: every due session proposes before
        # any Global Planner action changes another session's expected state.
        proposals: dict[str, Optional[ScalingProposal]] = {}
        next_ticks: dict[str, Optional[float]] = {}
        for participant_id in sorted(by_id):
            result = await self._sessions[participant_id].propose_tick_async(
                by_id[participant_id]
            )
            proposals[participant_id] = result.proposal
            next_ticks[participant_id] = result.next_tick_ms

        self._capacity_manager.begin_batch()
        actions: tuple[ReplayScaleAction, ...]
        try:
            for participant_id in sorted(proposals):
                proposal = proposals[participant_id]
                if proposal is None:
                    continue
                if not self._orchestrator.is_authorized(participant_id):
                    raise PermissionError(
                        f"Replay participant {participant_id!r} is not authorized"
                    )

                outcome = await self._orchestrator.submit(
                    participant_id,
                    self._targets_for(proposal),
                    blocking=False,
                    deployment_name=participant_id,
                    caller_name=participant_id,
                )
                self._global_planner_events.append(
                    GlobalPlannerReplayEvent(
                        at_s=now_s,
                        participant_id=participant_id,
                        status=outcome.status.value,
                        target_prefill=proposal.target_prefill,
                        target_decode=proposal.target_decode,
                        message=outcome.message,
                    )
                )
                if outcome.status == ScaleStatus.REJECTED:
                    self._sessions[participant_id].observe_rejection(
                        proposal, message=outcome.message
                    )
                elif outcome.status == ScaleStatus.SUCCESS:
                    self._sessions[participant_id].observe_mediation(
                        proposal, status="approved", message=outcome.message
                    )
                else:
                    self._sessions[participant_id].observe_mediation(
                        proposal, status="error", message=outcome.message
                    )
                    raise RuntimeError(
                        f"Global Planner returned {outcome.status.value} for "
                        f"{participant_id!r}: {outcome.message}"
                    )

            actions = self._capacity_manager.finish_batch()
        except BaseException:
            if self._capacity_manager.batch_active:
                self._capacity_manager.abort_batch()
            raise

        # Outcome notifications are delayed until every same-time local planner
        # has run.  An action may target a cached partner that did not tick.
        action_dicts: list[dict[str, Any]] = []
        for action in actions:
            target_prefill: Optional[int] = None
            target_decode: Optional[int] = None
            for target in action.targets:
                if target.sub_type == SubComponentType.PREFILL.value:
                    target_prefill = target.desired_replicas
                elif target.sub_type == SubComponentType.DECODE.value:
                    target_decode = target.desired_replicas
                else:  # Defensive: ReplayCapacityManager validates known roles.
                    raise ValueError(
                        f"unsupported Replay pool role {target.sub_type!r}"
                    )

            self._sessions[action.participant_id].observe_committed_scale(
                at_s=now_s,
                target_prefill=target_prefill,
                target_decode=target_decode,
                reason="global_planner",
            )
            action_dicts.append(
                {
                    "deployment_id": action.participant_id,
                    "target_prefill": target_prefill,
                    "target_decode": target_decode,
                }
            )

        return {"next_ticks": next_ticks, "actions": action_dicts}

    @staticmethod
    def _targets_for(proposal: ScalingProposal) -> list[TargetReplica]:
        targets: list[TargetReplica] = []
        if proposal.target_prefill is not None:
            targets.append(
                TargetReplica(
                    sub_component_type=SubComponentType.PREFILL,
                    desired_replicas=proposal.target_prefill,
                )
            )
        if proposal.target_decode is not None:
            targets.append(
                TargetReplica(
                    sub_component_type=SubComponentType.DECODE,
                    desired_replicas=proposal.target_decode,
                )
            )
        if not targets:
            raise ValueError("a planner proposal must target at least one component")
        return targets

    def finalize(self, raw_report: Mapping[str, Any]) -> ReplayWorldReport:
        """Build per-deployment planner reports and close all planner resources."""
        try:
            raw_deployments = raw_report.get("deployments", [])
            if isinstance(raw_deployments, Mapping):
                report_items = raw_deployments.items()
            else:
                report_items = raw_deployments
            trace_reports = {
                str(participant_id): report for participant_id, report in report_items
            }
            if set(trace_reports) != set(self._sessions):
                raise ValueError(
                    "Rust world report deployment IDs do not match planner sessions"
                )
            result = ReplayWorldReport(
                deployments={
                    participant_id: self._sessions[participant_id].build_report(
                        trace_reports[participant_id]
                    )
                    for participant_id in sorted(self._sessions)
                },
                global_planner_events=list(self._global_planner_events),
                duration_ms=float(raw_report.get("duration_ms", 0.0)),
                wall_time_ms=float(raw_report.get("wall_time_ms", 0.0)),
            )
        except BaseException as exc:
            try:
                self.close()
            except BaseException as cleanup_exc:
                exc.add_note(f"Replay world cleanup also failed: {cleanup_exc}")
            raise
        else:
            self.close()
            return result

    def close(self) -> None:
        """Shut down every local planner and the shared event loop."""
        if self._closed:
            return
        self._closed = True
        loop = self._loop
        self._loop = None
        if loop is None:
            return

        async def shutdown_sessions() -> list[BaseException]:
            errors: list[BaseException] = []
            for participant_id in sorted(self._sessions):
                try:
                    await self._sessions[participant_id].shutdown_async()
                except BaseException as exc:  # Continue closing sibling engines.
                    errors.append(exc)
            return errors

        errors: list[BaseException] = []
        try:
            errors.extend(loop.run_until_complete(shutdown_sessions()))
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        finally:
            loop.close()
        if errors:
            raise RuntimeError(
                f"{len(errors)} Replay planner session(s) failed to shut down"
            ) from errors[0]


def _planner_config_arg(config: dict[str, Any] | str) -> str:
    return json.dumps(config) if isinstance(config, dict) else config


def run_replay_world(
    deployments: list[ReplayDeploymentConfig],
    *,
    global_planner: ReplayGlobalPlannerConfig = ReplayGlobalPlannerConfig(),
) -> ReplayWorldReport:
    """Run a multi-deployment offline simulation with shared GP arbitration.

    Every deployment owns its trace or synthetic distribution.  Deployments may
    share ``model_name``; request assignment remains explicit and no Global
    Router is simulated. This synchronous API owns a private event loop and must
    not run on a thread that already has an active asyncio loop; async callers
    can offload it with :func:`asyncio.to_thread`.
    """

    if not deployments:
        raise ValueError("run_replay_world requires at least one deployment")
    by_id = {deployment.deployment_id: deployment for deployment in deployments}
    if len(by_id) != len(deployments):
        raise ValueError("Replay world deployment IDs must be unique")
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise RuntimeError(
            "run_replay_world is synchronous and cannot run on a thread with an "
            "active asyncio event loop"
        )

    # Imports are intentionally local: the legacy Replay API remains importable
    # in static environments where the Rust extension is not installed.
    from dynamo.mocker import MockEngineArgs, ReplayWorldBridge
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.core.types import WorkerCapabilities
    from dynamo.replay.main import (
        SyntheticWorkload,
        _build_planner_replay,
        _engine_caps,
    )

    loop = asyncio.new_event_loop()
    sessions: dict[str, ReplayPlannerAdapter] = {}
    controller: Optional[ReplayWorldPlannerController] = None

    try:
        clock = VirtualClock()
        participants: list[ReplayParticipantSpec] = []
        world_bridge = ReplayWorldBridge()
        for participant_id in sorted(by_id):
            deployment = by_id[participant_id]
            planner_config_arg = _planner_config_arg(deployment.planner_config)
            parsed_planner_config = PlannerConfig.from_config_arg(planner_config_arg)
            _validate_deployment_engine_args(deployment, parsed_planner_config.mode)

            extra_engine_args = deployment.extra_engine_args
            if parsed_planner_config.mode == "agg" and extra_engine_args is None:
                extra_engine_args = MockEngineArgs()

            workload = deployment.workload
            if isinstance(workload, ReplayTraceWorkload):
                trace_file: Optional[str] = str(workload.trace_file)
                synthetic = None
                arrival_speedup_ratio = workload.arrival_speedup_ratio
                trace_block_size = workload.trace_block_size
            else:
                trace_file = None
                synthetic = SyntheticWorkload(
                    input_tokens=workload.input_tokens,
                    output_tokens=workload.output_tokens,
                    request_count=workload.request_count,
                    arrival_interval_ms=workload.arrival_interval_ms,
                    turns_per_session=workload.turns_per_session,
                    shared_prefix_ratio=workload.shared_prefix_ratio,
                    num_prefix_groups=workload.num_prefix_groups,
                    inter_turn_delay_ms=workload.inter_turn_delay_ms,
                )
                arrival_speedup_ratio = workload.arrival_speedup_ratio
                trace_block_size = 512

            bridge, adapter = _build_planner_replay(
                trace_file=trace_file,
                extra_engine_args=extra_engine_args,
                prefill_engine_args=deployment.prefill_engine_args,
                decode_engine_args=deployment.decode_engine_args,
                router_config=deployment.router_config,
                num_workers=deployment.num_workers,
                num_prefill_workers=deployment.num_prefill_workers,
                num_decode_workers=deployment.num_decode_workers,
                router_mode=deployment.router_mode,
                arrival_speedup_ratio=arrival_speedup_ratio,
                trace_block_size=trace_block_size,
                planner_config_arg=planner_config_arg,
                model_name=deployment.model_name,
                benchmark_granularity=deployment.benchmark_granularity,
                sla_ttft_ms=deployment.sla_ttft_ms,
                sla_itl_ms=deployment.sla_itl_ms,
                sla_e2e_ms=deployment.sla_e2e_ms,
                replay_concurrency=deployment.replay_concurrency,
                synthetic=synthetic,
                event_loop=loop,
                clock=clock,
                report_namespace=participant_id,
            )
            sessions[participant_id] = adapter
            world_bridge.add_deployment(participant_id, bridge)

            if parsed_planner_config.mode == "agg":
                assert extra_engine_args is not None
                capabilities = WorkerCapabilities(
                    decode=_engine_caps(extra_engine_args)
                )
                gpu_per_replica = capabilities.decode.num_gpu
                if gpu_per_replica is None or gpu_per_replica <= 0:
                    raise ValueError(
                        f"deployment {participant_id!r} has invalid GPU width "
                        f"{gpu_per_replica!r}"
                    )
                pools = (
                    PoolSpec(
                        sub_type=SubComponentType.DECODE.value,
                        current_replicas=deployment.num_workers,
                        gpu_per_replica=gpu_per_replica,
                    ),
                )
            elif parsed_planner_config.mode == "disagg":
                assert deployment.prefill_engine_args is not None
                assert deployment.decode_engine_args is not None
                prefill_gpu = _engine_caps(deployment.prefill_engine_args).num_gpu
                decode_gpu = _engine_caps(deployment.decode_engine_args).num_gpu
                if (
                    prefill_gpu is None
                    or prefill_gpu <= 0
                    or decode_gpu is None
                    or decode_gpu <= 0
                ):
                    raise ValueError(
                        f"deployment {participant_id!r} has invalid GPU widths "
                        f"(prefill={prefill_gpu!r}, decode={decode_gpu!r})"
                    )
                pools = (
                    PoolSpec(
                        sub_type=SubComponentType.PREFILL.value,
                        current_replicas=deployment.num_prefill_workers,
                        gpu_per_replica=prefill_gpu,
                    ),
                    PoolSpec(
                        sub_type=SubComponentType.DECODE.value,
                        current_replicas=deployment.num_decode_workers,
                        gpu_per_replica=decode_gpu,
                    ),
                )
            else:  # Validated before constructing the deployment adapter.
                raise AssertionError(
                    f"unexpected planner mode {parsed_planner_config.mode!r}"
                )
            participants.append(
                ReplayParticipantSpec(participant_id=participant_id, pools=pools)
            )

        controller = ReplayWorldPlannerController(
            sessions,
            participants,
            config=global_planner,
            event_loop=loop,
            clock=clock,
        )
        raw_report = world_bridge.run(controller)
        return controller.finalize(raw_report)
    except BaseException as exc:
        try:
            if controller is not None:
                controller.close()
            else:

                async def shutdown_built_sessions() -> list[BaseException]:
                    errors: list[BaseException] = []
                    for participant_id in sorted(sessions):
                        try:
                            await sessions[participant_id].shutdown_async()
                        except BaseException as shutdown_exc:
                            errors.append(shutdown_exc)
                    return errors

                try:
                    shutdown_errors = loop.run_until_complete(shutdown_built_sessions())
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.run_until_complete(loop.shutdown_default_executor())
                finally:
                    loop.close()
                if shutdown_errors:
                    raise RuntimeError(
                        f"{len(shutdown_errors)} Replay planner session(s) "
                        "failed to shut down"
                    ) from shutdown_errors[0]
        except BaseException as cleanup_exc:
            exc.add_note(f"Replay world cleanup also failed: {cleanup_exc}")
        raise


__all__ = [
    "GlobalPlannerReplayEvent",
    "ReplayDeploymentConfig",
    "ReplayGlobalPlannerConfig",
    "ReplaySyntheticWorkload",
    "ReplayTraceWorkload",
    "ReplayWorldPlannerController",
    "ReplayWorldReport",
    "run_replay_world",
]
