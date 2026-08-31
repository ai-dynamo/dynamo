# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Builtin observation plugin backed by ``PlannerEnvironment``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from dynamo.planner.core.types import ScheduledTick, TickInput, WorkerCounts
from dynamo.planner.environment.interface import PlannerEnvironment


@dataclass
class ObserveStageRequest:
    scheduled_tick: ScheduledTick
    now_s: float


@dataclass
class ObserveStageResponse:
    tick_input: TickInput


class EnvironmentObserver(Protocol):
    """In-process contract for collecting one tick's environment observations."""

    async def Observe(self, request: ObserveStageRequest) -> ObserveStageResponse:
        """Collect the observations requested by ``request.scheduled_tick``."""
        ...


class EnvironmentObservePlugin:
    """Collect native planner observations through ``PlannerEnvironment``.

    This is intentionally in-process only for now. It gives the pipeline an
    OBSERVE-shaped boundary without exposing deployment-state observation as a
    public plugin wire contract yet.
    """

    plugin_id = "builtin_environment_observe"

    def __init__(
        self,
        environment: PlannerEnvironment,
        *,
        require_prefill: bool,
        require_decode: bool,
    ) -> None:
        self.environment = environment
        self.require_prefill = require_prefill
        self.require_decode = require_decode

    async def Observe(self, request: ObserveStageRequest) -> ObserveStageResponse:
        tick = request.scheduled_tick
        traffic = None
        worker_counts = None
        fpm_observations = None

        if tick.need_traffic_metrics:
            if tick.use_full_traffic_metrics:
                traffic = await self.environment.collect_traffic()
            else:
                traffic = await self.environment.collect_kv_hit_rate_observation(
                    tick.traffic_metrics_duration_s
                )
        if tick.need_worker_states:
            worker_counts = self._collect_worker_counts()
        if tick.need_worker_fpm:
            fpm_observations = self.environment.collect_fpm()

        return ObserveStageResponse(
            tick_input=TickInput(
                now_s=request.now_s,
                traffic=traffic,
                worker_counts=worker_counts,
                fpm_observations=fpm_observations,
            )
        )

    def _collect_worker_counts(self) -> WorkerCounts:
        state = self.environment.deployment_state()
        return WorkerCounts(
            ready_num_prefill=(
                state.prefill.replicas.active if self.require_prefill else None
            ),
            ready_num_decode=(
                state.decode.replicas.active if self.require_decode else None
            ),
            expected_num_prefill=(
                state.prefill.replicas.expected if self.require_prefill else None
            ),
            expected_num_decode=(
                state.decode.replicas.expected if self.require_decode else None
            ),
            prefill_scaling_in_progress=(
                self.require_prefill and state.prefill.replicas.scaling
            ),
            decode_scaling_in_progress=(
                self.require_decode and state.decode.replicas.scaling
            ),
        )


__all__ = [
    "EnvironmentObserver",
    "EnvironmentObservePlugin",
    "ObserveStageRequest",
    "ObserveStageResponse",
]
