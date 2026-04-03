# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import BasePlanner
from dynamo.planner.core.budget import _apply_global_gpu_budget, _initialize_gpu_counts
from dynamo.planner.core.decode import DecodePlanner
from dynamo.planner.core.prefill import PrefillPlanner
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.monitoring.planner_metrics import PlannerPrometheusMetrics
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class DisaggPlanner:
    def __init__(self, runtime: DistributedRuntime, config: PlannerConfig) -> None:
        self.config = config
        self.shared_state = PlannerSharedState()
        prometheus_metrics = PlannerPrometheusMetrics()

        self.enable_throughput = config.enable_throughput_scaling
        self.enable_load = config.enable_load_scaling

        self.prefill_planner = PrefillPlanner(
            runtime,
            config,
            shared_state=self.shared_state,
            prometheus_metrics=prometheus_metrics,
            start_prometheus_server=True,
        )
        self.decode_planner = DecodePlanner(
            runtime,
            config,
            shared_state=self.shared_state,
            prometheus_metrics=prometheus_metrics,
            prometheus_traffic_client=getattr(
                self.prefill_planner, "prometheus_traffic_client", None
            ),
            connector=getattr(self.prefill_planner, "connector", None),
            start_prometheus_server=False,
        )

    async def _async_init(self):
        # DisaggPlanner overrides _async_init to handle both prefill+decode
        # and share WorkerInfo between the two sub-planners.
        defaults = WORKER_COMPONENT_NAMES.get(self.config.backend)

        if not self.config.no_operation:
            # Connector init (prefill/decode share the same connector)
            connector = getattr(self.prefill_planner, "connector", None)
            if connector and hasattr(connector, "_async_init"):
                await connector._async_init()

            logger.info("Validating deployment...")
            await self.prefill_planner.connector.validate_deployment(
                prefill_component_name=(
                    defaults.prefill_worker_k8s_name if defaults else None
                ),
                decode_component_name=(
                    defaults.decode_worker_k8s_name if defaults else None
                ),
                require_prefill=True,
                require_decode=True,
            )
            logger.info("Successfully validated the deployment")

            _initialize_gpu_counts(
                self.config,
                self.prefill_planner.connector,
                require_prefill=True,
                require_decode=True,
            )

            await self.prefill_planner.connector.wait_for_deployment_ready(
                include_planner=False
            )

        await self.prefill_planner._init_worker_info(
            require_prefill=True, require_decode=True
        )
        # Share WorkerInfo and model name with decode planner
        self.decode_planner.prefill_worker_info = (
            self.prefill_planner.prefill_worker_info
        )
        self.decode_planner.decode_worker_info = self.prefill_planner.decode_worker_info
        self.decode_planner.model_name = self.prefill_planner.model_name

        # Start FPM tracking for both planners. DisaggPlanner bypasses each
        # sub-planner's _async_init(), so we init subscribers explicitly here.
        if self.enable_load:
            if self.prefill_planner.runtime is not None:
                await self.prefill_planner._init_fpm_subscriber()
            if self.decode_planner.runtime is not None:
                await self.decode_planner._init_fpm_subscriber()

    async def run(self):
        """Main scaling loop. Call _async_init() before this."""
        self.shared_state.last_adjustment_time = time.time()
        self.shared_state.last_load_adjustment_time = time.time()

        # FPM tracking (started in _async_init) replaces the former
        # DirectRouterMetricsClient.run_sampling_loop().
        loops = []
        if self.enable_throughput:
            loops.append(self._throughput_loop())
        if self.enable_load:
            loops.append(self._load_loop())

        await asyncio.gather(*loops)

    async def _throughput_loop(self) -> None:
        """Throughput-based scaling loop for disagg mode."""
        while True:
            current_time = time.time()

            if (
                current_time - self.shared_state.last_adjustment_time
                >= self.config.throughput_adjustment_interval
            ):
                self.shared_state.last_adjustment_time = time.time()
                logger.info("New throughput adjustment interval started!")

                await self.prefill_planner.observe_traffic_stats(
                    require_prefill=True, require_decode=True
                )
                self.decode_planner.update_predictors_from_metrics(
                    self.shared_state.last_metrics
                )
                next_num_p = self.prefill_planner.plan_adjustment()
                next_num_d = self.decode_planner.plan_adjustment()
                if next_num_p is None or next_num_d is None:
                    await asyncio.sleep(self.config.throughput_adjustment_interval / 10)
                    continue

                if self.enable_load:
                    # When load-based is also enabled: just set lower bounds
                    self.shared_state.throughput_lower_bound_p = next_num_p
                    self.shared_state.throughput_lower_bound_d = next_num_d
                    logger.info(
                        f"Throughput lower bounds set: prefill={next_num_p}, decode={next_num_d}"
                    )
                else:
                    # Throughput-only: apply scaling directly
                    next_num_p, next_num_d = _apply_global_gpu_budget(
                        next_num_p, next_num_d, self.config
                    )
                    self.prefill_planner.update_predicted_replicas_metric(next_num_p)
                    self.decode_planner.update_predicted_replicas_metric(next_num_d)

                    if not self.config.no_operation:
                        target_replicas = [
                            TargetReplica(
                                sub_component_type=SubComponentType.PREFILL,
                                component_name=self.prefill_planner.prefill_worker_info.k8s_name,
                                desired_replicas=next_num_p,
                            ),
                            TargetReplica(
                                sub_component_type=SubComponentType.DECODE,
                                component_name=self.prefill_planner.decode_worker_info.k8s_name,
                                desired_replicas=next_num_d,
                            ),
                        ]
                        await self.prefill_planner.connector.set_component_replicas(
                            target_replicas, blocking=False
                        )

            await asyncio.sleep(self.config.throughput_adjustment_interval / 10)

    async def _load_loop(self) -> None:
        """FPM-driven load-based scaling loop for disagg mode."""
        while True:
            await asyncio.sleep(self.config.load_adjustment_interval)
            logger.info("New load-based adjustment interval started!")

            num_p, num_d, _ = await self.prefill_planner.get_workers_info(
                require_prefill=True, require_decode=True
            )
            self.shared_state.num_p_workers = num_p
            self.shared_state.num_d_workers = num_d

            # Observe FPM stats and feed into regression models
            p_stats = self.prefill_planner.observe_fpm_load_stats()
            d_stats = self.decode_planner.observe_fpm_load_stats()

            if not p_stats and not d_stats:
                logger.warning("No FPM data for either prefill or decode, skipping")
                continue

            if p_stats and not BasePlanner._reconcile_fpm_worker_count(
                p_stats, num_p, "prefill"
            ):
                continue
            if d_stats and not BasePlanner._reconcile_fpm_worker_count(
                d_stats, num_d, "decode"
            ):
                continue

            p_desired = self.prefill_planner.load_plan_adjustment()
            d_desired = self.decode_planner.load_plan_adjustment()

            final_p = (
                p_desired if p_desired is not None else self.shared_state.num_p_workers
            )
            final_d = (
                d_desired if d_desired is not None else self.shared_state.num_d_workers
            )

            if (
                final_p == self.shared_state.num_p_workers
                and final_d == self.shared_state.num_d_workers
            ):
                logger.info("Load-based scaling: no scaling needed")
                continue

            # Enforce lower bounds from throughput-based
            if self.enable_throughput:
                final_p = max(final_p, self.shared_state.throughput_lower_bound_p)
                final_d = max(final_d, self.shared_state.throughput_lower_bound_d)

            # Enforce minimum endpoints
            final_p = max(final_p, self.config.min_endpoint)
            final_d = max(final_d, self.config.min_endpoint)

            # Apply GPU budget
            final_p, final_d = _apply_global_gpu_budget(final_p, final_d, self.config)

            logger.info(
                f"Load-based disagg scaling: prefill {self.shared_state.num_p_workers}->{final_p}, "
                f"decode {self.shared_state.num_d_workers}->{final_d}"
            )

            self.prefill_planner.update_predicted_replicas_metric(final_p)
            self.decode_planner.update_predicted_replicas_metric(final_d)

            if not self.config.no_operation:
                target_replicas = [
                    TargetReplica(
                        sub_component_type=SubComponentType.PREFILL,
                        component_name=self.prefill_planner.prefill_worker_info.k8s_name,
                        desired_replicas=final_p,
                    ),
                    TargetReplica(
                        sub_component_type=SubComponentType.DECODE,
                        component_name=self.prefill_planner.decode_worker_info.k8s_name,
                        desired_replicas=final_d,
                    ),
                ]
                await self.prefill_planner.connector.set_component_replicas(
                    target_replicas, blocking=True
                )
