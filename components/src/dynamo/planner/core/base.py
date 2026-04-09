# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Optional, Union

from prometheus_client import start_http_server

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.connectors.virtual import VirtualConnector
from dynamo.planner.core.budget import (
    _apply_component_gpu_budget,
    _initialize_gpu_counts,
)
from dynamo.planner.core.load.predictors import LOAD_PREDICTORS
from dynamo.planner.core.perf_model import DecodeRegressionModel, PrefillRegressionModel
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.monitoring.perf_metrics import fetch_pre_deployment_metrics
from dynamo.planner.monitoring.planner_metrics import PlannerPrometheusMetrics
from dynamo.planner.monitoring.traffic_metrics import Metrics, PrometheusAPIClient
from dynamo.planner.monitoring.worker_info import WorkerInfo, resolve_worker_info
from dynamo.planner.offline.trace_data import extract_metrics_from_mooncake

if TYPE_CHECKING:
    from dynamo.common.forward_pass_metrics import ForwardPassMetrics
    from dynamo.llm import FpmEventSubscriber


from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

ConnectorType = Union[GlobalPlannerConnector, KubernetesConnector, VirtualConnector]

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class BasePlanner:
    component_type: SubComponentType

    def __init__(
        self,
        runtime: Optional[DistributedRuntime],
        config: PlannerConfig,
        shared_state: Optional[PlannerSharedState] = None,
        prometheus_metrics: Optional[PlannerPrometheusMetrics] = None,
        prometheus_traffic_client: Optional[PrometheusAPIClient] = None,
        connector: Optional[ConnectorType] = None,
        start_prometheus_server: bool = True,
        component_type: Optional[SubComponentType] = None,
    ):
        if component_type is not None:
            self.component_type = component_type

        self.config = config
        self.shared_state = shared_state or PlannerSharedState()

        self.runtime = runtime
        self.namespace = config.namespace
        self.model_name: Optional[str] = None
        self.connector: ConnectorType

        if connector is not None:
            self.connector = connector
        elif not config.no_operation:
            if config.environment == "global-planner":
                assert config.global_planner_namespace is not None
                assert runtime is not None
                self.connector = GlobalPlannerConnector(
                    runtime,
                    self.namespace,
                    config.global_planner_namespace,
                    "GlobalPlanner",
                    config.model_name,
                )
            elif config.environment == "kubernetes":
                self.connector = KubernetesConnector(self.namespace, config.model_name)
            elif config.environment == "virtual":
                assert runtime is not None
                self.connector = VirtualConnector(
                    runtime,
                    self.namespace,
                    config.model_name,
                )
            else:
                raise ValueError(f"Invalid environment: {config.environment}")

        self.prometheus_traffic_client = (
            prometheus_traffic_client
            or PrometheusAPIClient(
                config.metric_pulling_prometheus_endpoint,
                config.namespace,
                metrics_source=config.throughput_metrics_source,
            )
        )
        if config.throughput_metrics_source == "router":
            self.prometheus_traffic_client.warn_if_router_not_scraped()

        predictor_cls = LOAD_PREDICTORS[config.load_predictor]
        self.num_req_predictor = predictor_cls(config)
        self.isl_predictor = predictor_cls(config)
        self.osl_predictor = predictor_cls(config)

        # Optional warmup: preload predictors with historical observations from a
        # mooncake-style JSONL trace (request_count/avg_isl/avg_osl per interval).
        if config.load_predictor_warmup_trace is not None:
            warmup_trace = config.load_predictor_warmup_trace
            try:
                metrics = extract_metrics_from_mooncake(
                    warmup_trace, config.throughput_adjustment_interval
                )
                for m in metrics:
                    self.num_req_predictor.add_data_point(float(m["request_count"]))
                    self.isl_predictor.add_data_point(float(m["avg_isl"]))
                    self.osl_predictor.add_data_point(float(m["avg_osl"]))
                logger.info(
                    f"Warmed load predictors with {len(metrics)} intervals from {warmup_trace}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to warm load predictors from {warmup_trace}: {e}"
                )
            finally:
                # Even with warmup data, ignore the initial post-deploy idle
                # period (leading zeros) when live metrics start coming in.
                for p in (
                    self.num_req_predictor,
                    self.isl_predictor,
                    self.osl_predictor,
                ):
                    if hasattr(p, "reset_idle_skip"):
                        p.reset_idle_skip()

        self.enable_load = config.enable_load_scaling
        self.enable_throughput = config.enable_throughput_scaling

        self.prefill_worker_info = WorkerInfo()
        self.decode_worker_info = WorkerInfo()

        self.prefill_client = None
        self.workers_client = None

        self.prometheus_port = config.metric_reporting_prometheus_port
        self.prometheus_metrics: PlannerPrometheusMetrics | None = None

        if prometheus_metrics is None:
            self.prometheus_metrics = PlannerPrometheusMetrics()
        else:
            self.prometheus_metrics = prometheus_metrics

        if start_prometheus_server and self.prometheus_port != 0:
            try:
                start_http_server(self.prometheus_port)
                logger.info(
                    f"Started Prometheus metrics server on port {self.prometheus_port}"
                )
            except Exception as e:
                logger.error(f"Failed to start Prometheus metrics server: {e}")

        self.fpm_subscriber: "Optional[FpmEventSubscriber]" = None

        if self.component_type == SubComponentType.PREFILL:
            self.ttft_regression = PrefillRegressionModel(
                max_num_fpm_samples=self.config.max_num_fpm_samples,
                min_observations=self.config.load_min_observations,
                bucket_count=self.config.fpm_sample_bucket_size,
            )
        elif self.component_type == SubComponentType.DECODE:
            self.itl_regression = DecodeRegressionModel(
                max_num_fpm_samples=self.config.max_num_fpm_samples,
                min_observations=self.config.load_min_observations,
                bucket_count=self.config.fpm_sample_bucket_size,
            )

    @property
    def last_metrics(self) -> Metrics:
        return self.shared_state.last_metrics

    @last_metrics.setter
    def last_metrics(self, value: Metrics) -> None:
        self.shared_state.last_metrics = value

    async def _init_worker_info(
        self, require_prefill: bool, require_decode: bool
    ) -> None:
        """Initialize WorkerInfo and model name in a single step."""
        connector = getattr(self, "connector", None)
        self.prefill_worker_info, self.decode_worker_info = resolve_worker_info(
            backend=self.config.backend,
            require_prefill=require_prefill,
            require_decode=require_decode,
            connector=connector,
            config_model_name=getattr(self.config, "model_name", ""),
            no_operation=self.config.no_operation,
        )
        # model_name is resolved and written into both WorkerInfo objects
        self.model_name = (
            self.decode_worker_info.model_name or self.prefill_worker_info.model_name
        )

    async def _async_init(self):
        """Async initialization: connector init, deployment validation, WorkerInfo."""
        if hasattr(self, "connector") and hasattr(self.connector, "_async_init"):
            await self.connector._async_init()

        require_prefill = self.component_type == SubComponentType.PREFILL
        require_decode = self.component_type == SubComponentType.DECODE

        if not self.config.no_operation:
            defaults = WORKER_COMPONENT_NAMES.get(self.config.backend)

            logger.info("Validating deployment...")
            await self.connector.validate_deployment(
                prefill_component_name=(
                    defaults.prefill_worker_k8s_name
                    if require_prefill and defaults
                    else None
                ),
                decode_component_name=(
                    defaults.decode_worker_k8s_name
                    if require_decode and defaults
                    else None
                ),
                require_prefill=require_prefill,
                require_decode=require_decode,
            )
            logger.info("Successfully validated the deployment")

            _initialize_gpu_counts(
                self.config,
                self.connector,
                require_prefill=require_prefill,
                require_decode=require_decode,
            )

            await self.connector.wait_for_deployment_ready(include_planner=False)

        await self._init_worker_info(
            require_prefill=require_prefill,
            require_decode=require_decode,
        )

        if self.runtime is not None:
            await self._init_fpm_subscriber()

        await self._bootstrap_regression()

    async def _bootstrap_regression(self) -> None:
        """Fetch pre-deployment FPM data and bootstrap the regression model."""
        worker_info = (
            self.prefill_worker_info
            if self.component_type == SubComponentType.PREFILL
            else self.decode_worker_info
        )
        try:
            fpms = await fetch_pre_deployment_metrics(
                runtime=self.runtime,
                namespace=self.namespace,
                worker_info=worker_info,
                profile_results_dir=self.config.profile_results_dir,
                component_type=self.component_type,
            )
            if self.component_type == SubComponentType.PREFILL:
                self.ttft_regression.load_benchmark_fpms(fpms)
            elif self.component_type == SubComponentType.DECODE:
                self.itl_regression.load_benchmark_fpms(fpms)
            logger.info(
                f"Bootstrapped {self.component_type.value} regression with "
                f"{len(fpms)} pre-deployment FPMs"
            )
        except Exception as e:
            if self.enable_throughput:
                raise
            logger.warning(
                f"No pre-deployment data for {self.component_type.value} regression: {e}. "
                "Load-based scaling will learn from live FPM only."
            )

    async def _init_fpm_subscriber(self) -> None:
        """Create and start the FPM subscriber for load-based scaling."""
        from dynamo.llm import FpmEventSubscriber

        worker_info = (
            self.prefill_worker_info
            if self.component_type == SubComponentType.PREFILL
            else self.decode_worker_info
        )
        if not worker_info.component_name or not worker_info.endpoint:
            logger.warning(
                "WorkerInfo missing component_name or endpoint, "
                "cannot create FPM subscriber"
            )
            return

        assert self.runtime is not None
        endpoint = self.runtime.endpoint(
            f"{self.namespace}.{worker_info.component_name}.{worker_info.endpoint}"
        )
        self.fpm_subscriber = FpmEventSubscriber(endpoint)
        self.fpm_subscriber.start_tracking()
        logger.info(
            f"FPM tracker started for {worker_info.component_name}.{worker_info.endpoint}"
        )

    def _get_fpm_stats(self) -> "dict[tuple[str, int], ForwardPassMetrics]":
        """Get decoded FPM stats from the subscriber, keyed by (worker_id, dp_rank)."""
        from dynamo.common.forward_pass_metrics import decode as decode_fpm

        if self.fpm_subscriber is None:
            return {}
        raw_stats = self.fpm_subscriber.get_recent_stats()
        result = {}
        for key, raw_bytes in raw_stats.items():
            fpm = decode_fpm(raw_bytes)
            if fpm is not None:
                result[key] = fpm
        return result

    async def _get_or_create_client(self, component_name: str, endpoint_name: str):
        """Create a client for the given component and endpoint, with a brief sleep for state sync."""
        assert self.runtime is not None, "Runtime is not initialized"
        client = await self.runtime.endpoint(
            f"{self.namespace}.{component_name}.{endpoint_name}"
        ).client()
        # TODO: remove this sleep after rust client() is blocking until watching state
        await asyncio.sleep(0.1)
        return client

    async def get_workers_info(
        self, require_prefill: bool = True, require_decode: bool = True
    ) -> tuple[int, int, bool]:
        """
        Get worker counts for prefill and decode components.

        Returns:
            tuple[int, int, bool]: (num_p_workers, num_d_workers, is_stable)
            - is_stable: False if rollout in progress (scaling should be skipped)
        """
        num_p_workers = 0
        num_d_workers = 0

        # For Kubernetes, use DGD status instead of runtime client
        if hasattr(self, "connector") and isinstance(
            self.connector, KubernetesConnector
        ):
            (
                prefill_count,
                decode_count,
                is_stable,
            ) = self.connector.get_actual_worker_counts(
                prefill_component_name=(
                    self.prefill_worker_info.k8s_name if require_prefill else None
                ),
                decode_component_name=(
                    self.decode_worker_info.k8s_name if require_decode else None
                ),
            )
            num_p_workers = prefill_count if require_prefill else 0
            num_d_workers = decode_count if require_decode else 0
            return num_p_workers, num_d_workers, is_stable

        # Fall back to runtime client for non-Kubernetes environments
        if self.runtime is None:
            raise RuntimeError("Runtime is not initialized")

        if require_prefill:
            try:
                if self.prefill_client is None:
                    assert self.prefill_worker_info.component_name is not None
                    assert self.prefill_worker_info.endpoint is not None
                    self.prefill_client = await self._get_or_create_client(
                        self.prefill_worker_info.component_name,
                        self.prefill_worker_info.endpoint,
                    )
                num_p_workers = len(self.prefill_client.instance_ids())  # type: ignore
            except Exception:
                num_p_workers = 0
                logger.warning(
                    "No prefill workers found, aggregated mode is not supported yet"
                )

        if require_decode:
            try:
                if self.workers_client is None:
                    assert self.decode_worker_info.component_name is not None
                    assert self.decode_worker_info.endpoint is not None
                    self.workers_client = await self._get_or_create_client(
                        self.decode_worker_info.component_name,
                        self.decode_worker_info.endpoint,
                    )
                num_d_workers = len(self.workers_client.instance_ids())  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to get decode worker endpoints: {e}")

        return num_p_workers, num_d_workers, True  # Always stable for non-K8s

    async def observe_traffic_stats(
        self, require_prefill: bool = True, require_decode: bool = True
    ) -> None:
        """
        Observe metrics from Prometheus and update shared state.
        """
        num_p_workers, num_d_workers, _ = await self.get_workers_info(
            require_prefill=require_prefill, require_decode=require_decode
        )

        self.shared_state.num_p_workers = num_p_workers
        self.shared_state.num_d_workers = num_d_workers
        logger.debug(
            f"Number of prefill workers: {num_p_workers}, number of decode workers: {num_d_workers}"
        )

        # Update Prometheus metrics if server is running
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.num_p_workers.set(num_p_workers)
            self.prometheus_metrics.num_d_workers.set(num_d_workers)

            # Calculate and accumulate GPU hours for this interval
            # TODO: track startup and shutdown times to get more accurate GPU hours
            interval_gpu_hours = (
                (
                    num_p_workers * (self.config.prefill_engine_num_gpu or 0)
                    + num_d_workers * (self.config.decode_engine_num_gpu or 0)
                )
                * self.config.throughput_adjustment_interval
                / 3600
            )
            self.shared_state.cumulative_gpu_hours += interval_gpu_hours
            self.prometheus_metrics.gpu_hours.set(
                self.shared_state.cumulative_gpu_hours
            )

        # Prometheus returns seconds, convert to milliseconds
        assert (
            self.model_name is not None
        ), "model_name must be set before observing traffic stats"

        interval_str = f"{self.config.throughput_adjustment_interval}s"
        self.last_metrics.ttft = (
            self.prometheus_traffic_client.get_avg_time_to_first_token(
                interval_str,
                self.model_name,
            )
            * 1000
        )
        self.last_metrics.itl = (
            self.prometheus_traffic_client.get_avg_inter_token_latency(
                interval_str,
                self.model_name,
            )
            * 1000
        )
        self.last_metrics.num_req = (
            self.prometheus_traffic_client.get_avg_request_count(
                interval_str,
                self.model_name,
            )
        )
        self.last_metrics.request_duration = (
            self.prometheus_traffic_client.get_avg_request_duration(
                interval_str,
                self.model_name,
            )
        )
        self.last_metrics.isl = (
            self.prometheus_traffic_client.get_avg_input_sequence_tokens(
                interval_str,
                self.model_name,
            )
        )
        self.last_metrics.osl = (
            self.prometheus_traffic_client.get_avg_output_sequence_tokens(
                interval_str,
                self.model_name,
            )
        )

        logger.info(
            f"Observed num_req: {self.last_metrics.num_req:.2f} isl: {self.last_metrics.isl:.2f} osl: {self.last_metrics.osl:.2f}"
        )
        logger.info(
            f"Observed ttft: {self.last_metrics.ttft:.2f}ms itl: {self.last_metrics.itl:.2f}ms"
        )

        # Update observed metrics in Prometheus
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.observed_ttft.set(self.last_metrics.ttft)
            self.prometheus_metrics.observed_itl.set(self.last_metrics.itl)
            self.prometheus_metrics.observed_request_rate.set(
                self.last_metrics.num_req / self.config.throughput_adjustment_interval
            )
            self.prometheus_metrics.observed_request_duration.set(
                self.last_metrics.request_duration
            )
            self.prometheus_metrics.observed_isl.set(self.last_metrics.isl)
            self.prometheus_metrics.observed_osl.set(self.last_metrics.osl)

        self.update_predictors_from_metrics(self.last_metrics)

    def update_predictors_from_metrics(self, metrics: Metrics) -> None:
        if metrics.num_req is not None:
            self.num_req_predictor.add_data_point(metrics.num_req)
        if metrics.isl is not None:
            self.isl_predictor.add_data_point(metrics.isl)
        if metrics.osl is not None:
            self.osl_predictor.add_data_point(metrics.osl)

    def predict_load(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        try:
            next_num_req = self.num_req_predictor.predict_next()
            next_isl = self.isl_predictor.predict_next()
            next_osl = self.osl_predictor.predict_next()
            logger.info(
                f"Predicted load: num_req={next_num_req:.2f}, isl={next_isl:.2f}, osl={next_osl:.2f}"
            )
            return next_num_req, next_isl, next_osl
        except Exception as e:
            logger.error(f"Failed to predict load: {e}")
            return None, None, None

    def plan_adjustment(self) -> Optional[int]:
        if not self.last_metrics.is_valid():
            logger.info(
                "Metrics contain None or NaN values (no active requests), skipping adjustment"
            )
            return None

        next_num_req, next_isl, next_osl = self.predict_load()
        if next_num_req is None or next_isl is None or next_osl is None:
            return None

        # Update predicted load metrics in Prometheus
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.predicted_request_rate.set(
                next_num_req / self.config.throughput_adjustment_interval
            )
            self.prometheus_metrics.predicted_isl.set(next_isl)
            self.prometheus_metrics.predicted_osl.set(next_osl)

        try:
            return self._compute_replica_requirements(next_num_req, next_isl, next_osl)
        except Exception as e:
            logger.error(f"Failed to compute number of replicas: {e}")
            return None

    def update_predicted_replicas_metric(self, desired_replicas: int) -> None:
        raise NotImplementedError

    def _compute_replica_requirements(
        self, next_num_req: float, next_isl: float, next_osl: float
    ) -> Optional[int]:
        raise NotImplementedError

    def _component_name(self) -> str:
        if self.component_type == SubComponentType.PREFILL:
            assert self.prefill_worker_info.k8s_name is not None
            return self.prefill_worker_info.k8s_name
        assert self.decode_worker_info.k8s_name is not None
        return self.decode_worker_info.k8s_name

    def _engine_num_gpu(self) -> int:
        if self.component_type == SubComponentType.PREFILL:
            assert self.config.prefill_engine_num_gpu is not None
            return self.config.prefill_engine_num_gpu
        assert self.config.decode_engine_num_gpu is not None
        return self.config.decode_engine_num_gpu

    def apply_component_budget(self, desired_replicas: int) -> int:
        return _apply_component_gpu_budget(
            max(desired_replicas, self.config.min_endpoint),
            self._engine_num_gpu(),
            self.config,
        )

    async def _apply_scaling(self, desired_replicas: int) -> None:
        if self.config.no_operation:
            return
        target_replicas = [
            TargetReplica(
                sub_component_type=self.component_type,
                component_name=self._component_name(),
                desired_replicas=desired_replicas,
            )
        ]
        await self.connector.set_component_replicas(target_replicas, blocking=False)

    _apply_scaling_blocking = _apply_scaling

    @staticmethod
    def _reconcile_fpm_worker_count(
        fpm_stats: "dict[tuple[str, int], ForwardPassMetrics]",
        dgd_count: int,
        label: str,
    ) -> bool:
        """Validate that FPM coverage matches DGD worker count, accounting for DP.

        With attention DP, each worker emits FPM per dp_rank. We check that
        the number of unique worker IDs matches DGD, and that all workers
        have the same number of dp_ranks (complete coverage).

        Returns True if counts match, False otherwise.
        """
        workers_to_dp: dict[str, set[int]] = {}
        for wid, dp in fpm_stats:
            workers_to_dp.setdefault(wid, set()).add(dp)

        fpm_worker_count = len(workers_to_dp)
        if fpm_worker_count != dgd_count:
            logger.warning(
                f"Worker count mismatch: DGD reports {dgd_count}, "
                f"FPM reports {fpm_worker_count} workers for {label}. "
                "Skipping scaling."
            )
            return False

        dp_sizes = {len(dps) for dps in workers_to_dp.values()}
        if len(dp_sizes) > 1:
            logger.warning(
                f"Inconsistent DP ranks across workers for {label}: "
                f"{dict(workers_to_dp)}. Skipping scaling."
            )
            return False

        dp_size = dp_sizes.pop() if dp_sizes else 1
        expected_total = dgd_count * dp_size
        actual_total = len(fpm_stats)
        if actual_total != expected_total:
            logger.warning(
                f"Incomplete FPM coverage for {label}: expected "
                f"{dgd_count} workers × {dp_size} dp_ranks = {expected_total}, "
                f"got {actual_total}. Skipping scaling."
            )
            return False

        if dp_size > 1:
            logger.info(
                f"FPM {label}: {fpm_worker_count} workers × {dp_size} dp_ranks "
                f"= {actual_total} engines"
            )
        return True

    @staticmethod
    def _log_fpm(wid: str, dp: int, fpm: "ForwardPassMetrics", label: str) -> None:
        sched = fpm.scheduled_requests
        queued = fpm.queued_requests
        logger.info(
            f"FPM {label} engine {wid}:dp{dp}: "
            f"wall_time={fpm.wall_time:.4f}s, "
            f"sched(prefill_tok={sched.sum_prefill_tokens}, "
            f"prefill_req={sched.num_prefill_requests}, "
            f"decode_kv={sched.sum_decode_kv_tokens}, "
            f"decode_req={sched.num_decode_requests}), "
            f"queued(prefill_tok={queued.sum_prefill_tokens}, "
            f"decode_kv={queued.sum_decode_kv_tokens})"
        )

    def observe_fpm_load_stats(
        self,
    ) -> "dict[tuple[str, int], ForwardPassMetrics]":
        """Get latest FPM stats and feed observations into the regression model.

        Returns:
            The decoded FPM stats dict for use by load_plan_adjustment().
        """
        fpm_stats = self._get_fpm_stats()
        if not fpm_stats:
            logger.warning(
                f"No FPM data available for {self.component_type.value} (tracker empty)"
            )
            return {}

        for (wid, dp), fpm in fpm_stats.items():
            self._log_fpm(wid, dp, fpm, self.component_type.value)
            if self.component_type == SubComponentType.PREFILL:
                self.ttft_regression.add_observation(fpm)
            elif self.component_type == SubComponentType.DECODE:
                self.itl_regression.add_observation(fpm)

        logger.info(
            f"FPM load stats: {len(fpm_stats)} engines observed for "
            f"{self.component_type.value}"
        )
        return fpm_stats

    def _load_based_scaling_decision_from_estimates(
        self,
        estimates: list[float],
        sla: float,
        num_workers: int,
        label: str,
    ) -> Optional[int]:
        """Shared scale-up/down logic from per-engine latency estimates (ms).

        Args:
            estimates: per-engine estimated latencies in ms.
            sla: target SLA in ms (e.g. config.ttft or config.itl).
            num_workers: current worker count for this component.
            label: human-readable label for log messages (e.g. "prefill TTFT").

        Returns:
            Desired replica count, or None if no scaling action needed.
        """
        if not estimates:
            return None

        sensitivity = self.config.load_scaling_down_sensitivity / 100.0

        logger.info(
            f"Load-based {label}: workers={num_workers}, sla={sla:.1f}ms, "
            f"estimates={[f'{t:.1f}' for t in estimates]}"
        )

        if all(t > sla for t in estimates):
            logger.info(
                f"Load-based {label}: ALL engines above SLA ({sla:.1f}ms), "
                f"scaling up to {num_workers + 1}"
            )
            return num_workers + 1

        if num_workers > 1:
            threshold = sla * sensitivity
            if all(t < threshold for t in estimates):
                desired = max(num_workers - 1, self.config.min_endpoint)
                if desired == num_workers:
                    logger.info(
                        f"Load-based {label}: ALL engines below threshold "
                        f"({threshold:.1f}ms), but at min_endpoint ({self.config.min_endpoint})"
                    )
                else:
                    logger.info(
                        f"Load-based {label}: ALL engines below threshold "
                        f"({threshold:.1f}ms), scaling down to {desired}"
                    )
                return desired

        return None

    def load_plan_adjustment(self) -> Optional[int]:
        """Load-based scaling decision. Override in subclasses."""
        raise NotImplementedError

    async def _throughput_loop(
        self, require_prefill: bool, require_decode: bool
    ) -> None:
        """Throughput-based scaling loop (existing behavior, extracted from run())."""
        while True:
            current_time = time.time()

            if (
                current_time - self.shared_state.last_adjustment_time
                >= self.config.throughput_adjustment_interval
            ):
                self.shared_state.last_adjustment_time = time.time()
                logger.info("New throughput adjustment interval started!")

                await self.observe_traffic_stats(
                    require_prefill=require_prefill, require_decode=require_decode
                )
                desired_replicas = self.plan_adjustment()
                if desired_replicas is not None:
                    if self.enable_load:
                        # When load-based is also enabled: just set lower bound
                        if self.component_type == SubComponentType.PREFILL:
                            self.shared_state.throughput_lower_bound_p = (
                                desired_replicas
                            )
                        else:
                            self.shared_state.throughput_lower_bound_d = (
                                desired_replicas
                            )
                        logger.info(
                            f"Throughput lower bound set to {desired_replicas} for {self.component_type.value}"
                        )
                    else:
                        # Throughput-only: apply scaling directly
                        desired_replicas = self.apply_component_budget(desired_replicas)
                        self.update_predicted_replicas_metric(desired_replicas)
                        # Throughput planner does not needs blocking scaling because it monitors
                        # and predicts the load, not relying on the current status of the engine.
                        await self._apply_scaling(desired_replicas)

            await asyncio.sleep(self.config.throughput_adjustment_interval / 10)

    async def _load_and_fpm_update_loop(
        self, require_prefill: bool, require_decode: bool
    ) -> None:
        """FPM observation and (optionally) load-based scaling loop.

        Runs every load_adjustment_interval.  Always updates the FPM
        regression model with live observations.  When load-based scaling
        is enabled, also makes scaling decisions immediately after the
        FPM update.
        """
        pending_desired: Optional[int] = None
        while True:
            await asyncio.sleep(self.config.load_adjustment_interval)
            logger.info("New load/FPM update interval started!")

            num_p, num_d, is_stable = await self.get_workers_info(
                require_prefill=require_prefill, require_decode=require_decode
            )
            self.shared_state.num_p_workers = num_p
            self.shared_state.num_d_workers = num_d

            fpm_stats = self.observe_fpm_load_stats()
            if not fpm_stats:
                continue

            if not self.enable_load:
                continue

            if pending_desired is not None:
                dgd_count = (
                    num_p if self.component_type == SubComponentType.PREFILL else num_d
                )
                if dgd_count == pending_desired:
                    logger.info(
                        f"Scaling to {pending_desired} complete, resuming decisions"
                    )
                    pending_desired = None
                else:
                    logger.info(
                        f"Scaling in progress ({dgd_count} -> {pending_desired}), "
                        "observing only"
                    )
                    continue

            dgd_count = (
                num_p if self.component_type == SubComponentType.PREFILL else num_d
            )
            if not self._reconcile_fpm_worker_count(
                fpm_stats, dgd_count, self.component_type.value
            ):
                continue

            desired_replicas = self.load_plan_adjustment()

            if desired_replicas is not None:
                if self.enable_throughput:
                    if self.component_type == SubComponentType.PREFILL:
                        lower_bound = self.shared_state.throughput_lower_bound_p
                    else:
                        lower_bound = self.shared_state.throughput_lower_bound_d
                    desired_replicas = max(desired_replicas, lower_bound)
                desired_replicas = self.apply_component_budget(desired_replicas)
                self.update_predicted_replicas_metric(desired_replicas)
                pending_desired = desired_replicas
                await self._apply_scaling_blocking(desired_replicas)

    async def run(self):
        """Main scaling loop. Call _async_init() before this."""
        require_prefill = self.component_type == SubComponentType.PREFILL
        require_decode = self.component_type == SubComponentType.DECODE

        self.shared_state.last_adjustment_time = time.time()
        self.shared_state.last_load_adjustment_time = time.time()

        loops = []
        if self.enable_throughput:
            loops.append(self._throughput_loop(require_prefill, require_decode))
        loops.append(self._load_and_fpm_update_loop(require_prefill, require_decode))

        await asyncio.gather(*loops)
