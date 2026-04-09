# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import math
import time
from typing import TYPE_CHECKING, Optional

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.core.base import BasePlanner
from dynamo.planner.core.budget import (
    _apply_component_gpu_budget,
    _initialize_gpu_counts,
)
from dynamo.planner.core.perf_model import AggRegressionModel
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.monitoring.perf_metrics import fetch_pre_deployment_metrics
from dynamo.planner.monitoring.planner_metrics import PlannerPrometheusMetrics
from dynamo.runtime import DistributedRuntime

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class AggPlanner:
    """Aggregated planner: FPM-driven scaling for single engine type.

    In aggregated mode, engines handle both prefill and decode (chunked prefill).
    A single AggRegressionModel maps (sum_prefill_tokens, sum_decode_kv_tokens)
    to wall_time using 2D linear regression.

    Supports load-only, throughput-only, or both scaling modes.

    Scaling logic (load-based):
    - Estimate next TTFT per engine by simulating prefill chunking with
      piggybacked decode (steady-state decode load).
    - Estimate next ITL per engine by predicting decode iteration time with
      average piggybacked prefill load.
    - Scale up if (ALL TTFT > SLA) OR (ALL ITL > SLA).
    - Scale down if (ALL TTFT < SLA * sensitivity) AND (ALL ITL < SLA * sensitivity).

    Scaling logic (throughput-based):
    - Use compute_agg_replicas() to find minimum replicas where both SLAs
      are met under predicted traffic load.
    """

    def __init__(self, runtime: DistributedRuntime, config: PlannerConfig) -> None:
        self.config = config
        self.runtime = runtime
        self.shared_state = PlannerSharedState()

        self.enable_throughput = config.enable_throughput_scaling
        self.enable_load = config.enable_load_scaling

        if not self.enable_throughput and not self.enable_load:
            raise ValueError(
                "Aggregated planner requires at least one scaling mode enabled."
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

        self.regression = AggRegressionModel(
            max_num_fpm_samples=config.max_num_fpm_samples,
            min_observations=config.load_min_observations,
            bucket_count=config.fpm_sample_bucket_size,
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

        if self.runtime is not None:
            await self.planner._init_fpm_subscriber()

        await self._bootstrap_regression()

    async def _bootstrap_regression(self) -> None:
        """Bootstrap agg regression from pre-deployment benchmark data."""
        worker_info = self.planner.decode_worker_info
        try:
            fpms = await fetch_pre_deployment_metrics(
                runtime=self.runtime,
                namespace=self.config.namespace,
                worker_info=worker_info,
                profile_results_dir=self.config.profile_results_dir,
                component_type=SubComponentType.DECODE,
            )
            self.regression.load_benchmark_fpms(fpms)
            logger.info(
                f"Bootstrapped agg regression with {len(fpms)} pre-deployment FPMs"
            )
        except Exception as e:
            if self.enable_throughput:
                raise
            logger.warning(
                f"No pre-deployment data for agg regression: {e}. "
                "Load-based scaling will learn from live FPM only."
            )

    async def run(self):
        """Main scaling loop. Call _async_init() before this."""
        self.shared_state.last_adjustment_time = time.time()

        loops = []
        if self.enable_throughput:
            loops.append(self._throughput_loop())
        loops.append(self._load_and_fpm_update_loop())

        await asyncio.gather(*loops)

    async def _throughput_loop(self) -> None:
        """Throughput-based scaling loop for agg mode."""
        while True:
            current_time = time.time()

            if (
                current_time - self.shared_state.last_adjustment_time
                >= self.config.throughput_adjustment_interval
            ):
                self.shared_state.last_adjustment_time = time.time()
                logger.info("New agg throughput adjustment interval started!")

                await self.planner.observe_traffic_stats(
                    require_prefill=False, require_decode=True
                )
                metrics = self.shared_state.last_metrics
                if not metrics.is_valid():
                    logger.info("Metrics invalid, skipping agg throughput adjustment")
                    await asyncio.sleep(self.config.throughput_adjustment_interval / 10)
                    continue

                next_num_req = self.planner.num_req_predictor.predict_next()
                next_isl = self.planner.isl_predictor.predict_next()
                next_osl = self.planner.osl_predictor.predict_next()

                max_num_batched_tokens = getattr(
                    self.planner.decode_worker_info, "max_num_batched_tokens", None
                )
                if not max_num_batched_tokens or max_num_batched_tokens <= 0:
                    logger.warning(
                        "max_num_batched_tokens not available, skipping agg throughput"
                    )
                    await asyncio.sleep(self.config.throughput_adjustment_interval / 10)
                    continue

                (
                    engine_rps,
                    actual_ttft,
                    actual_itl,
                ) = self.regression.find_best_engine_agg_rps(
                    isl=next_isl,
                    osl=next_osl,
                    max_num_batched_tokens=max_num_batched_tokens,
                    ttft_sla=self.config.ttft,
                    itl_sla=self.config.itl,
                )
                if engine_rps <= 0:
                    logger.warning(
                        "Agg perf model not ready, skipping throughput scaling"
                    )
                    await asyncio.sleep(self.config.throughput_adjustment_interval / 10)
                    continue

                if actual_ttft > self.config.ttft or actual_itl > self.config.itl:
                    logger.warning(
                        f"Agg SLA not fully met: TTFT={actual_ttft:.1f}ms "
                        f"(target {self.config.ttft:.1f}ms), "
                        f"ITL={actual_itl:.1f}ms (target {self.config.itl:.1f}ms), "
                        "scaling with best achievable rate"
                    )

                demand_rps = next_num_req / self.config.throughput_adjustment_interval
                desired = math.ceil(demand_rps / engine_rps)
                desired = max(desired, self.config.min_endpoint)
                logger.info(
                    f"Agg: {demand_rps:.2f}(demand rps) / "
                    f"{engine_rps:.2f}(engine rps) = {desired}(replicas), "
                    f"est_ttft={actual_ttft:.1f}ms, est_itl={actual_itl:.1f}ms"
                )

                if self.enable_load:
                    self.shared_state.throughput_lower_bound_d = desired
                    logger.info(f"Agg throughput lower bound set to {desired}")
                else:
                    assert self.config.decode_engine_num_gpu is not None
                    desired = _apply_component_gpu_budget(
                        desired, self.config.decode_engine_num_gpu, self.config
                    )
                    if (
                        self.planner.prometheus_port != 0
                        and self.planner.prometheus_metrics is not None
                    ):
                        self.planner.prometheus_metrics.predicted_num_d.set(desired)

                    if not self.config.no_operation:
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

            await asyncio.sleep(self.config.throughput_adjustment_interval / 10)

    async def _load_and_fpm_update_loop(self) -> None:
        """FPM observation and (optionally) load-based scaling for agg mode.

        Always updates regression with live FPM. When load-based scaling
        is enabled, makes scaling decisions immediately after.
        """
        pending_desired: Optional[int] = None
        while True:
            await asyncio.sleep(self.config.load_adjustment_interval)
            logger.info("New agg load/FPM update interval started!")

            _, num_d, _ = await self.planner.get_workers_info(
                require_prefill=False, require_decode=True
            )
            self.shared_state.num_d_workers = num_d
            num_workers = num_d

            fpm_stats = self.planner._get_fpm_stats()
            if not fpm_stats:
                continue

            for (wid, dp), fpm in fpm_stats.items():
                BasePlanner._log_fpm(wid, dp, fpm, "agg")
                self.regression.add_observation(fpm)

            if not self.enable_load:
                continue

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
            if self.enable_throughput:
                desired = max(desired, self.shared_state.throughput_lower_bound_d)
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
