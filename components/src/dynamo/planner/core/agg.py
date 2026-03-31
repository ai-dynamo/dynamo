# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics

from dynamo.planner.core.base import BasePlanner
from dynamo.planner.core.budget import (
    _apply_component_gpu_budget,
    _initialize_gpu_counts,
)
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.monitoring.planner_metrics import PlannerPrometheusMetrics
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class AggPlanner:
    """Aggregated planner: FPM-driven load-based scaling, single engine type.

    In aggregated mode, engines handle both prefill and decode (chunked prefill).
    A single AggRegressionModel maps (sum_prefill_tokens, sum_decode_kv_tokens)
    to wall_time using 2D linear regression.

    Scaling logic:
    - Estimate next TTFT per engine by simulating prefill chunking with
      piggybacked decode (steady-state decode load).
    - Estimate next ITL per engine by predicting decode iteration time with
      average piggybacked prefill load.
    - Scale up if (ALL TTFT > SLA) OR (ALL ITL > SLA).
    - Scale down if (ALL TTFT < SLA * sensitivity) AND (ALL ITL < SLA * sensitivity).
    """

    def __init__(self, runtime: DistributedRuntime, config: PlannerConfig) -> None:
        self.config = config
        self.runtime = runtime
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

        self.planner = BasePlanner(
            runtime,
            config,
            shared_state=self.shared_state,
            prometheus_metrics=prometheus_metrics,
            start_prometheus_server=True,
            component_type=SubComponentType.DECODE,
        )

        from dynamo.planner.core.load.fpm_regression import AggRegressionModel

        self.regression = AggRegressionModel(
            window_size=config.load_learning_window,
            min_observations=config.load_min_observations,
        )

    async def _async_init(self):
        defaults = WORKER_COMPONENT_NAMES.get(self.config.backend)

        if not self.config.no_operation:
            connector = getattr(self.planner, "connector", None)
            if connector and hasattr(connector, "_async_init"):
                await connector._async_init()

            logger.info("Validating deployment...")
            await self.planner.connector.validate_deployment(
                prefill_component_name=None,
                decode_component_name=(
                    defaults.decode_worker_k8s_name if defaults else None
                ),
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

            await self.planner.connector.wait_for_deployment_ready(
                include_planner=False
            )

        await self.planner._init_worker_info(require_prefill=False, require_decode=True)

        # Delegate FPM tracking to the inner BasePlanner (component_type=DECODE).
        if self.runtime is not None:
            await self.planner._init_fpm_subscriber()

    async def run(self):
        """Main scaling loop. Call _async_init() before this."""
        await asyncio.gather(self._load_loop())

    async def _load_loop(self) -> None:
        """FPM-driven load-based scaling loop for aggregated mode."""
        pending_desired: Optional[int] = None
        while True:
            await asyncio.sleep(self.config.load_adjustment_interval)
            logger.info("New agg load-based adjustment interval started!")

            _, num_d, _ = await self.planner.get_workers_info(
                require_prefill=False, require_decode=True
            )
            self.shared_state.num_d_workers = num_d
            num_workers = num_d

            # Always observe FPM stats and update regression, even during scaling.
            fpm_stats = self.planner._get_fpm_stats()
            if not fpm_stats:
                logger.warning("No FPM data available for agg engines")
                continue

            for (wid, dp), fpm in fpm_stats.items():
                BasePlanner._log_fpm(wid, dp, fpm, "agg")
                self.regression.add_observation(fpm)

            # If a previous scaling action is still in progress, skip decisions.
            if pending_desired is not None:
                if num_workers == pending_desired:
                    logger.info(
                        f"Scaling to {pending_desired} complete, resuming decisions"
                    )
                    pending_desired = None
                else:
                    logger.info(
                        f"Scaling in progress ({num_workers} -> {pending_desired}), "
                        "observing only"
                    )
                    continue

            if not BasePlanner._reconcile_fpm_worker_count(
                fpm_stats, num_workers, "agg"
            ):
                continue

            if not self.regression.has_sufficient_data():
                logger.info(
                    f"Agg regression: insufficient data "
                    f"({self.regression.num_observations}/{self.regression.min_observations})"
                )
                continue

            max_num_batched_tokens = getattr(
                self.planner.decode_worker_info, "max_num_batched_tokens", None
            )
            if not max_num_batched_tokens or max_num_batched_tokens <= 0:
                logger.warning(
                    "max_num_batched_tokens not available from WorkerInfo, "
                    "skipping agg scaling"
                )
                continue

            p_desired = self._prefill_scaling_decision(
                fpm_stats, num_workers, max_num_batched_tokens
            )
            d_desired = self._decode_scaling_decision(fpm_stats, num_workers)

            logger.info(
                f"Agg scaling decisions: prefill={p_desired}, decode={d_desired} "
                f"(current={num_workers})"
            )

            # Scale up if EITHER dimension wants more workers.
            # Scale down only if BOTH dimensions agree on fewer.
            if p_desired is not None and p_desired > num_workers:
                desired = p_desired
            elif d_desired is not None and d_desired > num_workers:
                desired = d_desired
            elif (
                p_desired is not None
                and p_desired < num_workers
                and d_desired is not None
                and d_desired < num_workers
            ):
                desired = max(p_desired, d_desired)
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
                pending_desired = desired
                target_replicas = [
                    TargetReplica(
                        sub_component_type=SubComponentType.DECODE,
                        component_name=self.planner.decode_worker_info.k8s_name,
                        desired_replicas=desired,
                    )
                ]
                await self.planner.connector.set_component_replicas(
                    target_replicas, blocking=False
                )

    def _prefill_scaling_decision(
        self,
        fpm_stats: "dict[tuple[str, int], ForwardPassMetrics]",
        num_workers: int,
        max_num_batched_tokens: int,
    ) -> Optional[int]:
        """Returns desired replica count for the prefill (TTFT) dimension, or None."""
        estimated_ttfts: list[float] = []
        for (wid, dp), fpm in fpm_stats.items():
            est = self.regression.estimate_next_ttft(
                queued_prefill_tokens=fpm.queued_requests.sum_prefill_tokens,
                max_num_batched_tokens=max_num_batched_tokens,
                current_decode_kv=fpm.scheduled_requests.sum_decode_kv_tokens,
            )
            if est is not None:
                estimated_ttfts.append(est * 1000)

        return self.planner._load_based_scaling_decision_from_estimates(
            estimated_ttfts, self.config.ttft, num_workers, "agg TTFT"
        )

    def _decode_scaling_decision(
        self,
        fpm_stats: "dict[tuple[str, int], ForwardPassMetrics]",
        num_workers: int,
    ) -> Optional[int]:
        """Returns desired replica count for the decode (ITL) dimension, or None."""
        estimated_itls: list[float] = []
        for (wid, dp), fpm in fpm_stats.items():
            est = self.regression.estimate_next_itl(
                scheduled_decode_kv=fpm.scheduled_requests.sum_decode_kv_tokens,
                queued_decode_kv=fpm.queued_requests.sum_decode_kv_tokens,
            )
            if est is not None:
                estimated_itls.append(est * 1000)

        return self.planner._load_based_scaling_decision_from_estimates(
            estimated_itls, self.config.itl, num_workers, "agg ITL"
        )
