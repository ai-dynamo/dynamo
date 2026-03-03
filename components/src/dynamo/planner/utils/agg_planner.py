# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Optional

from dynamo.planner import SubComponentType, TargetReplica
from dynamo.planner.utils.load_based_regression import LoadBasedRegressionModel
from dynamo.planner.utils.planner_config import PlannerConfig
from dynamo.planner.utils.planner_core import (
    BasePlanner,
    PlannerPrometheusMetrics,
    PlannerSharedState,
    _apply_component_gpu_budget,
    _initialize_gpu_counts,
)
from dynamo.planner.utils.prometheus import CachedLoadMetrics
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class AggPlanner:
    """Aggregated planner: load-based scaling only, single engine type.

    In aggregated mode, engines handle both prefill and decode (chunked prefill).
    Engine metrics are labeled "decode" by the router.

    Scaling logic:
    - TTFT and ITL regression models are both maintained.
    - Regression uses per-worker time-averaged metrics (not latest snapshot)
      because chunked prefill adds noise to instantaneous TTFT/ITL.
    - Scale up if either prefill or decode target is exceeded.
    - Scale down if both prefill and decode are below their boundaries.
    """

    # Engine metrics from agg workers are labeled "decode" by the router
    ENGINE_WORKER_TYPE = "decode"

    def __init__(
        self, runtime: Optional[DistributedRuntime], config: PlannerConfig
    ) -> None:
        self.config = config
        self.shared_state = PlannerSharedState()

        if config.enable_throughput_scaling:
            raise ValueError(
                "Aggregated planner only supports load-based scaling. "
                "Set enable_throughput_scaling to false in the config."
            )
        if not config.enable_load_scaling:
            raise ValueError(
                "Aggregated planner requires enable_load_scaling to be true."
            )

        prometheus_metrics = PlannerPrometheusMetrics()

        # Use a single BasePlanner instance for infra (connector, prometheus, etc.)
        # We use DECODE component_type because engine metrics are labeled "decode"
        self.planner = BasePlanner(
            runtime,
            config,
            shared_state=self.shared_state,
            prometheus_metrics=prometheus_metrics,
            start_prometheus_server=True,
        )
        # Override: agg planner uses component_type DECODE for metrics fetching
        self.planner.component_type = SubComponentType.DECODE

        # Create both regression models (agg needs both TTFT and ITL)
        self.ttft_regression = LoadBasedRegressionModel(
            window_size=config.load_learning_window,
            min_observations=config.load_min_observations,
        )
        self.itl_regression = LoadBasedRegressionModel(
            window_size=config.load_learning_window,
            min_observations=config.load_min_observations,
        )

        self.cached_load_metrics = CachedLoadMetrics()

    async def _async_init(self):
        await self.planner._async_init()

    async def run(self):
        if not self.config.no_operation:
            logger.info("Validating deployment...")
            # Agg mode: only decode component exists (engines serve both P and D)
            await self.planner.connector.validate_deployment(
                prefill_component_name=None,
                decode_component_name=self.planner.decode_component_name,
                require_prefill=False,
                require_decode=True,
            )
            logger.info("Successfully validated the deployment")

            _initialize_gpu_counts(
                self.config,
                self.planner.connector,
                require_prefill=False,
                require_decode=True,
            )

            await self.planner.connector.wait_for_deployment_ready()

        # Model name discovery runs in all modes (needed for metrics collection)
        if not self.config.no_operation:
            model_name = await self.planner._get_model_name(
                require_prefill=False, require_decode=True
            )
            logger.info(f"Detected model name from deployment: {model_name}")
            self.planner.model_name = model_name.lower()
        else:
            model_name = getattr(self.config, "model_name", None)
            if not model_name:
                raise ValueError(
                    "Model name is required in no-operation mode. "
                    "Please set model_name in the config."
                )
            self.planner.model_name = model_name.lower()

        loops = [
            self._load_loop(),
            self.planner.prometheus_engine_client.run_sampling_loop(
                self.config.load_metric_samples,
                self.config.load_adjustment_interval,
            ),
        ]
        await asyncio.gather(*loops)

    async def _observe_engine_load_stats(self) -> None:
        """Fetch metrics and update regression models using per-worker time-averaged data."""
        result = self.planner.prometheus_engine_client.get_recent_and_averaged_metrics(
            self.ENGINE_WORKER_TYPE
        )
        if result is None:
            logger.warning(
                f"No per-worker metrics available yet for {self.ENGINE_WORKER_TYPE} (buffer empty)"
            )
            return

        recent, per_worker_averaged, cluster_averaged = result
        self.cached_load_metrics = CachedLoadMetrics(
            recent=recent,
            per_worker_averaged=per_worker_averaged,
            cluster_averaged=cluster_averaged,
        )

        # Agg uses per-worker time-averaged metrics for regression
        # because chunked prefill adds noise to instantaneous TTFT/ITL
        for wid, m in per_worker_averaged.items():
            # TTFT regression: (active_prefill_tokens + ISL) -> TTFT
            active_prefill = m.get("active_prefill_tokens", 0.0)
            last_isl = m.get("last_isl", 0.0)
            last_ttft = m.get("last_ttft", 0.0)
            if last_ttft > 0 and last_isl > 0:
                x = active_prefill + last_isl
                y = last_ttft * 1000  # seconds -> ms
                logger.info(
                    f"Agg Worker {wid} prefill observation: TTFT {y:.2f}ms @ tokens {x:.2f}"
                )
                self.ttft_regression.add_observation(x, y)

            # ITL regression: active_decode_blocks -> ITL
            active_decode = m.get("active_decode_blocks", 0.0)
            last_itl = m.get("last_itl", 0.0)
            if last_itl > 0 and active_decode > 0:
                x = active_decode
                y = last_itl * 1000  # seconds -> ms
                logger.info(
                    f"Agg Worker {wid} decode observation: ITL {y:.2f}ms @ blocks {x:.2f}"
                )
                self.itl_regression.add_observation(x, y)

    def _prefill_scaling_decision(self, num_workers: int) -> Optional[str]:
        """Returns "up", "down", or None for prefill dimension."""
        if not self.cached_load_metrics.recent:
            return None
        if not self.ttft_regression.has_sufficient_data():
            logger.info(
                f"TTFT regression: insufficient data ({self.ttft_regression.num_observations}"
                f"/{self.ttft_regression.min_observations}), skipping"
            )
            return None

        x_sla = self.ttft_regression.predict_x_from_sla(self.config.ttft)
        if x_sla is None:
            return None

        recent = self.cached_load_metrics.recent
        cluster_averaged = self.cached_load_metrics.cluster_averaged
        avg_isl = cluster_averaged.get("last_isl", 0.0)
        target = x_sla - avg_isl

        if target <= 0:
            logger.warning(
                f"Agg TTFT SLA unachievable at current ISL: x_sla={x_sla:.1f}, "
                f"avg_isl={avg_isl:.1f}, skipping prefill scaling decision"
            )
            return None

        logger.info(
            f"Agg prefill: x_sla={x_sla:.1f}, avg_isl={avg_isl:.1f}, "
            f"target_active_tokens={target:.1f}, workers={num_workers}"
        )

        # Scale up: ALL workers above target
        if all(m.get("active_prefill_tokens", 0.0) > target for m in recent.values()):
            return "up"

        # Scale down: ALL workers below boundary
        if num_workers > 1:
            sensitivity = self.config.load_scaling_down_sensitivity / 100.0
            boundary = target * (num_workers - 1) / num_workers * sensitivity
            if all(
                m.get("active_prefill_tokens", 0.0) < boundary for m in recent.values()
            ):
                return "down"

        return None

    def _decode_scaling_decision(self, num_workers: int) -> Optional[str]:
        """Returns "up", "down", or None for decode dimension."""
        if not self.cached_load_metrics.recent:
            return None
        if not self.itl_regression.has_sufficient_data():
            logger.info(
                f"ITL regression: insufficient data ({self.itl_regression.num_observations}"
                f"/{self.itl_regression.min_observations}), skipping"
            )
            return None

        x_sla = self.itl_regression.predict_x_from_sla(self.config.itl)
        if x_sla is None:
            return None

        if x_sla <= 0:
            logger.warning(
                f"Agg ITL SLA unachievable: x_sla={x_sla:.1f}, "
                "skipping decode scaling decision"
            )
            return None

        recent = self.cached_load_metrics.recent

        logger.info(f"Agg decode: x_sla={x_sla:.1f}, workers={num_workers}")

        # Scale up: ALL workers above target
        if all(m.get("active_decode_blocks", 0.0) > x_sla for m in recent.values()):
            return "up"

        # Scale down: ALL workers below boundary
        # TODO: should we strictly enforce all workers below boundary?
        # how about user-configurable percentage?
        if num_workers > 1:
            sensitivity = self.config.load_scaling_down_sensitivity / 100.0
            boundary = x_sla * (num_workers - 1) / num_workers * sensitivity
            if all(
                m.get("active_decode_blocks", 0.0) < boundary for m in recent.values()
            ):
                return "down"

        return None

    async def _load_loop(self) -> None:
        """Load-based scaling loop for aggregated mode."""
        while True:
            await asyncio.sleep(self.config.load_adjustment_interval)
            logger.info("New agg load-based adjustment interval started!")

            # Query DGD for fresh worker counts
            _, num_d, _ = await self.planner.get_workers_info(
                require_prefill=False, require_decode=True
            )
            self.shared_state.num_d_workers = num_d
            num_workers = num_d

            # Observe per-worker metrics
            await self._observe_engine_load_stats()

            # Reconcile worker counts
            prom_count = len(self.cached_load_metrics.recent)
            if prom_count != num_workers:
                logger.warning(
                    f"Worker count mismatch: DGD reports {num_workers}, "
                    f"router metrics reports {prom_count}. Skipping."
                )
                continue

            if not self.cached_load_metrics.recent:
                continue

            # Make scaling decisions separately for prefill and decode
            p_decision = self._prefill_scaling_decision(num_workers)
            d_decision = self._decode_scaling_decision(num_workers)

            logger.info(
                f"Agg scaling decisions: prefill={p_decision}, decode={d_decision}"
            )

            # Scale up if EITHER needs scale up
            # Scale down if BOTH need scale down
            if p_decision == "up" or d_decision == "up":
                desired = num_workers + 1
            elif p_decision == "down" and d_decision == "down":
                desired = num_workers - 1
            else:
                logger.info("Agg scaling: no scaling needed")
                continue

            desired = max(desired, self.config.min_endpoint)
            assert self.config.decode_engine_num_gpu is not None
            desired = _apply_component_gpu_budget(
                desired, self.config.decode_engine_num_gpu, self.config
            )

            logger.info(f"Agg load-based scaling: {num_workers} -> {desired}")

            if (
                self.planner.prometheus_port != 0
                and self.planner.prometheus_metrics is not None
            ):
                self.planner.prometheus_metrics.predicted_num_d.set(desired)

            if not self.config.no_operation:
                target_replicas = [
                    TargetReplica(
                        sub_component_type=SubComponentType.DECODE,
                        component_name=self.planner.decode_component_name,
                        desired_replicas=desired,
                    )
                ]
                await self.planner.connector.set_component_replicas(
                    target_replicas, blocking=True
                )
