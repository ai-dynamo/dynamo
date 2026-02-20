# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import Optional

from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.utils.decode_planner import DecodePlanner
from dynamo.planner.utils.planner_config import PlannerConfig
from dynamo.planner.utils.planner_core import (
    PlannerPrometheusMetrics,
    PlannerSharedState,
    _apply_global_gpu_budget,
    _initialize_gpu_counts,
)
from dynamo.planner.utils.prefill_planner import PrefillPlanner
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class DisaggPlanner:
    def __init__(
        self, runtime: Optional[DistributedRuntime], config: PlannerConfig
    ) -> None:
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
            prometheus_engine_client=getattr(
                self.prefill_planner, "prometheus_engine_client", None
            ),
            connector=getattr(self.prefill_planner, "connector", None),
            start_prometheus_server=False,
        )

    async def _async_init(self):
        # Prefill/Decode share the same connector instance in disagg mode.
        await self.prefill_planner._async_init()

    async def run(self):
        if not self.config.no_operation:
            logger.info("Validating deployment...")
            await self.prefill_planner.connector.validate_deployment(
                prefill_component_name=self.prefill_planner.prefill_component_name,
                decode_component_name=self.prefill_planner.decode_component_name,
                require_prefill=True,
                require_decode=True,
            )
            logger.info("Successfully validated the deployment")

            # Initialize GPU counts
            _initialize_gpu_counts(
                self.config,
                self.prefill_planner.connector,
                require_prefill=True,
                require_decode=True,
            )

            await self.prefill_planner.connector.wait_for_deployment_ready()

        # Model name discovery runs in all modes (needed for metrics collection)
        if not self.config.no_operation:
            model_name = await self.prefill_planner._get_model_name(
                require_prefill=True, require_decode=True
            )
            logger.info(f"Detected model name from deployment: {model_name}")
            model_name = model_name.lower()
        else:
            model_name = getattr(self.config, "model_name", None)
            if not model_name:
                raise ValueError(
                    "Model name is required in no-operation mode. "
                    "Please set model_name in the config."
                )
            model_name = model_name.lower()
        self.prefill_planner.model_name = model_name
        self.decode_planner.model_name = model_name

        self.shared_state.last_adjustment_time = time.time()
        self.shared_state.last_load_adjustment_time = time.time()

        # Build list of concurrent loops based on enabled scaling modes
        loops = []
        if self.enable_throughput:
            loops.append(self._throughput_loop())
        if self.enable_load:
            loops.append(self._load_loop())
            loops.append(
                self.prefill_planner.prometheus_engine_client.run_sampling_loop(
                    self.config.load_metric_samples,
                    self.config.load_adjustment_interval,
                )
            )

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
                                component_name=self.prefill_planner.prefill_component_name,
                                desired_replicas=next_num_p,
                            ),
                            TargetReplica(
                                sub_component_type=SubComponentType.DECODE,
                                component_name=self.prefill_planner.decode_component_name,
                                desired_replicas=next_num_d,
                            ),
                        ]
                        await self.prefill_planner.connector.set_component_replicas(
                            target_replicas, blocking=False
                        )

            await asyncio.sleep(self.config.throughput_adjustment_interval / 10)

    async def _load_loop(self) -> None:
        """Load-based scaling loop for disagg mode at shorter interval."""
        while True:
            await asyncio.sleep(self.config.load_adjustment_interval)
            logger.info("New load-based adjustment interval started!")

            # Query DGD for fresh worker counts
            num_p, num_d, _ = await self.prefill_planner.get_workers_info(
                require_prefill=True, require_decode=True
            )
            self.shared_state.num_p_workers = num_p
            self.shared_state.num_d_workers = num_d

            # Observe per-worker metrics from router
            await self.prefill_planner.observe_engine_load_stats()
            await self.decode_planner.observe_engine_load_stats()

            # Reconcile DGD worker counts with router Prometheus counts
            p_prom_count = len(self.prefill_planner.cached_load_metrics.recent)
            d_prom_count = len(self.decode_planner.cached_load_metrics.recent)
            if p_prom_count != num_p or d_prom_count != num_d:
                logger.warning(
                    f"Worker count mismatch: DGD reports P={num_p}, D={num_d}; "
                    f"router metrics reports P={p_prom_count}, D={d_prom_count}. "
                    "Skipping load-based scaling adjustment."
                )
                continue

            # Scale prefill and decode independently
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
                        component_name=self.prefill_planner.prefill_component_name,
                        desired_replicas=final_p,
                    ),
                    TargetReplica(
                        sub_component_type=SubComponentType.DECODE,
                        component_name=self.prefill_planner.decode_component_name,
                        desired_replicas=final_d,
                    ),
                ]
                await self.prefill_planner.connector.set_component_replicas(
                    target_replicas, blocking=True
                )
