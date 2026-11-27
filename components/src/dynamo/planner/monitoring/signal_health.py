# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging

from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.types import (
    ScheduledTick,
    TickDiagnostics,
    TickInput,
    WorkerCapabilities,
)
from dynamo.planner.monitoring.planner_metrics import (
    SIGNAL_DEGRADATION_OBSERVATIONS,
    PluginFrameworkMetrics,
)

logger = logging.getLogger(__name__)


class PlannerSignalHealth:
    def __init__(self, config: PlannerConfig, metrics: PluginFrameworkMetrics | None):
        self._config = config
        self._metrics = metrics
        self._streaks: dict[tuple[str, str], int] = {}

    def _record(self, component: str, signal: str, unavailable: bool) -> None:
        key = component, signal
        previous = self._streaks.get(key, 0)
        current = previous + 1 if unavailable else 0
        self._streaks[key] = current
        if self._metrics is not None:
            self._metrics.signal_unavailable.labels(
                component=component, signal=signal
            ).set(int(unavailable))
        if current == SIGNAL_DEGRADATION_OBSERVATIONS:
            logger.warning(
                "Planner degraded signal: component=%s signal=%s unavailable for %s consecutive observations",
                component,
                signal,
                SIGNAL_DEGRADATION_OBSERVATIONS,
            )
            if self._metrics is not None:
                self._metrics.degraded_signals_total.labels(
                    component=component, signal=signal
                ).inc()
        elif previous >= SIGNAL_DEGRADATION_OBSERVATIONS and current == 0:
            logger.info(
                "Planner signal recovered: component=%s signal=%s", component, signal
            )

    def observe(
        self,
        scheduled: ScheduledTick,
        inputs: TickInput,
        capabilities: WorkerCapabilities,
        diagnostics: TickDiagnostics,
    ) -> None:
        if scheduled.need_traffic_metrics and scheduled.use_full_traffic_metrics:
            self._record("fleet", "traffic", inputs.traffic is None)
        if not scheduled.need_worker_fpm:
            return

        roles = (
            (
                "prefill",
                self._config.mode in ("prefill", "disagg"),
                capabilities.prefill,
            ),
            (
                "decode",
                self._config.mode in ("decode", "disagg", "agg"),
                capabilities.decode,
            ),
        )
        for component, enabled, caps in roles:
            if not enabled:
                continue
            count = None
            if inputs.worker_counts is not None:
                count = (
                    inputs.worker_counts.ready_num_prefill
                    if component == "prefill"
                    else inputs.worker_counts.ready_num_decode
                )
            expected = count != 0
            stats = None
            reason = None
            if inputs.fpm_observations is not None:
                stats = (
                    inputs.fpm_observations.prefill
                    if component == "prefill"
                    else inputs.fpm_observations.decode
                )
                reason = inputs.fpm_observations.unavailable_reasons.get(component)
            self._record(component, "fpm", expected and not stats)
            for rejected in ("stale_fpm", "worker_join_failure", "invalid_fpm"):
                self._record(component, rejected, expected and reason == rejected)
            for path, ran in (
                ("load", scheduled.run_load_scaling),
                ("throughput", scheduled.run_throughput_scaling),
            ):
                available = getattr(diagnostics, f"{path}_metadata_available").get(
                    component
                )
                if not expected or (ran and available is not None):
                    self._record(
                        component, f"{path}_inputs", expected and available is False
                    )
            if component == "decode" and scheduled.need_traffic_metrics:
                speculative = self._config.speculative_nextn or (
                    caps.speculative_nextn if caps is not None else 0
                )
                self._record(
                    component,
                    "accept_length",
                    bool(
                        expected
                        and speculative
                        and (
                            inputs.traffic is None
                            or inputs.traffic.accept_length is None
                        )
                    ),
                )
