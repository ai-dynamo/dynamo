# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from prometheus_client import Gauge, start_http_server

from dynamo.planner import (
    KubernetesConnector,
    SubComponentType,
    TargetReplica,
    VirtualConnector,
)
from dynamo.planner.defaults import WORKER_COMPONENT_NAMES
from dynamo.planner.global_planner_connector import GlobalPlannerConnector
from dynamo.planner.utils.exceptions import DeploymentValidationError
from dynamo.planner.utils.load_predictor import LOAD_PREDICTORS
from dynamo.planner.utils.perf_interpolation import (
    DecodeInterpolator,
    PrefillInterpolator,
)
from dynamo.planner.utils.planner_config import PlannerConfig
from dynamo.planner.utils.pre_swept_results_utils import PreSweptResultsHelper
from dynamo.planner.utils.prometheus import (
    CachedLoadMetrics,
    DirectRouterMetricsClient,
    Metrics,
    PrometheusAPIClient,
)
from dynamo.planner.utils.trace_data_extractor import extract_metrics_from_mooncake
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


class PlannerPrometheusMetrics:
    """Container for all Planner Prometheus metrics."""

    def __init__(self, prefix: str = "planner"):
        # Worker counts
        self.num_p_workers = Gauge(
            f"{prefix}:num_p_workers", "Number of prefill workers"
        )
        self.num_d_workers = Gauge(
            f"{prefix}:num_d_workers", "Number of decode workers"
        )

        # Observed metrics
        self.observed_ttft = Gauge(
            f"{prefix}:observed_ttft", "Observed time to first token (ms)"
        )
        self.observed_itl = Gauge(
            f"{prefix}:observed_itl", "Observed inter-token latency (ms)"
        )
        self.observed_request_rate = Gauge(
            f"{prefix}:observed_request_rate", "Observed request rate (req/s)"
        )
        self.observed_request_duration = Gauge(
            f"{prefix}:observed_request_duration", "Observed request duration (s)"
        )
        self.observed_isl = Gauge(
            f"{prefix}:observed_isl", "Observed input sequence length"
        )
        self.observed_osl = Gauge(
            f"{prefix}:observed_osl", "Observed output sequence length"
        )

        # Correction factors
        self.p_correction_factor = Gauge(
            f"{prefix}:p_correction_factor", "Prefill correction factor"
        )
        self.d_correction_factor = Gauge(
            f"{prefix}:d_correction_factor", "Decode correction factor"
        )

        # Predicted metrics
        self.predicted_request_rate = Gauge(
            f"{prefix}:predicted_request_rate", "Predicted request rate (req/s)"
        )
        self.predicted_isl = Gauge(
            f"{prefix}:predicted_isl", "Predicted input sequence length"
        )
        self.predicted_osl = Gauge(
            f"{prefix}:predicted_osl", "Predicted output sequence length"
        )
        self.predicted_num_p = Gauge(
            f"{prefix}:predicted_num_p", "Predicted number of prefill replicas"
        )
        self.predicted_num_d = Gauge(
            f"{prefix}:predicted_num_d", "Predicted number of decode replicas"
        )

        # Cumulative GPU usage
        self.gpu_hours = Gauge(f"{prefix}:gpu_hours", "Cumulative GPU hours used")


@dataclass
class PlannerSharedState:
    last_metrics: Metrics = field(default_factory=Metrics)
    num_p_workers: int = 0
    num_d_workers: int = 0
    cumulative_gpu_hours: float = 0.0
    last_adjustment_time: float = 0.0
    # Lower bounds from throughput-based scaling (used when both modes enabled)
    throughput_lower_bound_p: int = 1
    throughput_lower_bound_d: int = 1
    # Separate timestamp for load-based adjustment loop
    last_load_adjustment_time: float = 0.0


def _apply_global_gpu_budget(
    next_num_p: int, next_num_d: int, config: PlannerConfig
) -> tuple[int, int]:
    """Apply GPU budget constraint to both prefill and decode replicas.

    When total GPUs required (num_p * prefill_gpus + num_d * decode_gpus) exceeds the
    budget, scale down both proportionally using scale = budget / total_required. Prefill
    replicas are clamped to [min_endpoint, max_prefill] where max_prefill reserves enough
    GPUs for min_endpoint decode replicas. Remaining budget is then allocated to decode.
    Returns (0, 0) if budget cannot satisfy min_endpoint for both components.
    """
    if config.max_gpu_budget < 0:
        return next_num_p, next_num_d
    assert config.prefill_engine_num_gpu is not None
    assert config.decode_engine_num_gpu is not None
    total_gpu_required = (
        next_num_p * config.prefill_engine_num_gpu
        + next_num_d * config.decode_engine_num_gpu
    )
    if total_gpu_required <= config.max_gpu_budget:
        return next_num_p, next_num_d
    min_required = (
        config.min_endpoint * config.prefill_engine_num_gpu
        + config.min_endpoint * config.decode_engine_num_gpu
    )
    if config.max_gpu_budget < min_required:
        logger.warning(
            f"max_gpu_budget ({config.max_gpu_budget}) is below the minimum required "
            f"for min_endpoint ({min_required}); enforcing zero replicas"
        )
        return 0, 0
    scale = config.max_gpu_budget / total_gpu_required
    max_prefill = math.floor(
        (config.max_gpu_budget - config.min_endpoint * config.decode_engine_num_gpu)
        / config.prefill_engine_num_gpu
    )
    next_num_p = max(
        config.min_endpoint, min(max_prefill, math.floor(next_num_p * scale))
    )
    remaining = config.max_gpu_budget - next_num_p * config.prefill_engine_num_gpu
    next_num_d = max(
        config.min_endpoint, math.floor(remaining / config.decode_engine_num_gpu)
    )
    logger.warning(
        f"Total number of GPUs required ({total_gpu_required}) exceeds the max GPU budget ({config.max_gpu_budget}), "
        f"scaling down to {next_num_p} prefill and {next_num_d} decode replicas"
    )
    return next_num_p, next_num_d


def _apply_component_gpu_budget(
    desired_replicas: int, engine_num_gpu: int, config: PlannerConfig
) -> int:
    """Apply GPU budget constraint to a single component (prefill-only or decode-only).

    When total GPUs required (replicas * gpus_per_replica) exceeds the budget, scale down
    using scale = budget / total_required, floored and clamped to at least min_endpoint.
    Returns 0 if budget cannot satisfy min_endpoint replicas.
    """
    if config.max_gpu_budget < 0:
        return desired_replicas
    total_gpu_required = desired_replicas * engine_num_gpu
    if total_gpu_required <= config.max_gpu_budget:
        return desired_replicas
    min_required = config.min_endpoint * engine_num_gpu
    if config.max_gpu_budget < min_required:
        logger.warning(
            f"max_gpu_budget ({config.max_gpu_budget}) is below the minimum required "
            f"for min_endpoint ({min_required}); enforcing zero replicas"
        )
        return 0
    scale = config.max_gpu_budget / total_gpu_required
    next_num = max(config.min_endpoint, math.floor(desired_replicas * scale))
    logger.warning(
        f"Total number of GPUs required ({total_gpu_required}) exceeds the max GPU budget ({config.max_gpu_budget}), "
        f"scaling down to {next_num} replicas"
    )
    return next_num


def _initialize_gpu_counts(
    config: PlannerConfig,
    connector,
    require_prefill: bool,
    require_decode: bool,
) -> None:
    """Initialize GPU counts from DGD (Kubernetes) or config (virtual).

    In Kubernetes mode: reads from DGD, falls back to CLI flags if not found
    (useful for mockers that don't specify GPU resources).
    In virtual mode: requires CLI flags, errors if not provided.

    Raises:
        DeploymentValidationError: If GPU counts cannot be determined
    """
    # Try to read from DGD in Kubernetes mode
    if hasattr(connector, "get_gpu_counts"):
        try:
            prefill_gpu, decode_gpu = connector.get_gpu_counts(
                require_prefill=require_prefill,
                require_decode=require_decode,
            )
            config.prefill_engine_num_gpu = prefill_gpu
            config.decode_engine_num_gpu = decode_gpu
            logger.info(
                f"Detected GPU counts from DGD: prefill={prefill_gpu}, decode={decode_gpu}"
            )
            return
        except Exception as e:
            # Fall back to CLI flags (e.g., for mockers without GPU resources in DGD)
            logger.warning(
                f"Could not read GPU counts from DGD ({e}), falling back to CLI flags"
            )

    # Use CLI flags (virtual mode, or K8s fallback when DGD lacks GPU resources)
    errors = []
    if require_prefill and config.prefill_engine_num_gpu is None:
        errors.append("Missing prefill_engine_num_gpu in config")
    if require_decode and config.decode_engine_num_gpu is None:
        errors.append("Missing decode_engine_num_gpu in config")
    if errors:
        raise DeploymentValidationError(errors)
    logger.info(
        f"Using GPU counts from CLI: prefill={config.prefill_engine_num_gpu}, "
        f"decode={config.decode_engine_num_gpu}"
    )


class BasePlanner:
    component_type: SubComponentType

    def __init__(
        self,
        runtime: Optional[DistributedRuntime],
        config: PlannerConfig,
        dryrun: bool = False,
        shared_state: Optional[PlannerSharedState] = None,
        prometheus_metrics: Optional[PlannerPrometheusMetrics] = None,
        prometheus_traffic_client: Optional[PrometheusAPIClient] = None,
        prometheus_engine_client: Optional[DirectRouterMetricsClient] = None,
        connector=None,
        start_prometheus_server: bool = True,
    ):
        self.config = config
        self.dryrun = dryrun
        self.shared_state = shared_state or PlannerSharedState()

        # Rely on getting model name from connector
        self.model_name: Optional[str] = None

        if not self.dryrun:
            self.runtime = runtime
            self.namespace = config.namespace

            if not config.no_operation:
                # Initialize connector based on environment
                if config.environment == "global-planner":
                    assert config.global_planner_namespace is not None
                    self.connector = GlobalPlannerConnector(
                        runtime,
                        self.namespace,
                        config.global_planner_namespace,
                        "GlobalPlanner",
                        config.model_name,
                    )
                elif config.environment == "kubernetes":
                    self.connector = KubernetesConnector(
                        self.namespace, self.model_name
                    )
                elif config.environment == "virtual":
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
                )
            )

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

        # Load-based scaling flags.
        # Argument validation (flag resolution, constraint checks, correction factor
        # auto-disable) is handled by validate_sla_planner_args() in planner_argparse.
        self.enable_load = config.enable_load_scaling
        self.enable_throughput = config.enable_throughput_scaling

        # Only create interpolators when throughput-based scaling is enabled
        # (they require profiling data that isn't needed for load-based-only mode)
        if self.enable_throughput:
            if "use-pre-swept-results" in config.profile_results_dir:
                config_list = config.profile_results_dir.split(":")
                configs = {
                    "gpu_type": config_list[1],
                    "model": config_list[2],
                    "framework": config_list[3],
                    "framework_version": config_list[4],
                    "tp": int(config_list[5]),
                    "dp": int(config_list[6]),
                    "pp": int(config_list[7]),
                    "block_size": int(config_list[8]),
                    "max_batch_size": int(config_list[9]),
                    "gpu_count": int(config_list[10]),
                }
                if self.dryrun:
                    pre_swept_results_helper = PreSweptResultsHelper(
                        configs["gpu_type"], configs["framework"], configs["model"]
                    )
                    raw_data = pre_swept_results_helper.select_data("prefill", configs)
                    self.prefill_interpolator = PrefillInterpolator(raw_data=raw_data)
                    raw_data = pre_swept_results_helper.select_data("decode", configs)
                    self.decode_interpolator = DecodeInterpolator(raw_data=raw_data)
                else:
                    raise ValueError(
                        "Cannot set profile_results_dir to 'use-pre-swept-results' in non-dryrun mode"
                    )
            else:
                self.prefill_interpolator = PrefillInterpolator(
                    config.profile_results_dir
                )
                self.decode_interpolator = DecodeInterpolator(
                    config.profile_results_dir
                )

        self.prefill_component_name = WORKER_COMPONENT_NAMES[
            self.config.backend
        ].prefill_worker_k8s_name
        self.decode_component_name = WORKER_COMPONENT_NAMES[
            self.config.backend
        ].decode_worker_k8s_name

        if not self.dryrun:
            self.prefill_client = None
            self.workers_client = None

            self.prometheus_port = config.metric_reporting_prometheus_port

            if prometheus_metrics is None:
                self.prometheus_metrics = PlannerPrometheusMetrics()
            else:
                self.prometheus_metrics = prometheus_metrics

            # Start Prometheus HTTP server if port is specified
            if start_prometheus_server and self.prometheus_port != 0:
                try:
                    start_http_server(self.prometheus_port)
                    logger.info(
                        f"Started Prometheus metrics server on port {self.prometheus_port}"
                    )
                except Exception as e:
                    logger.error(f"Failed to start Prometheus metrics server: {e}")
        else:
            self.prometheus_port = 0
            self.prometheus_metrics = prometheus_metrics

        self.p_correction_factor = 1.0
        self.d_correction_factor = 1.0
        if self.dryrun:
            self.no_correction = True
        else:
            self.no_correction = config.no_correction

        if self.enable_load:
            if prometheus_engine_client is not None:
                self.prometheus_engine_client = prometheus_engine_client
            else:
                # Auto-discover frontend metrics URL in Kubernetes mode
                if not config.load_router_metrics_url and isinstance(
                    getattr(self, "connector", None), KubernetesConnector
                ):
                    config.load_router_metrics_url = (
                        self.connector.get_frontend_metrics_url()
                    )
                    if not config.load_router_metrics_url:
                        raise ValueError(
                            "Could not auto-discover frontend metrics URL from DGD. "
                            "No service with componentType 'frontend' found. "
                            "Please set load_router_metrics_url in the config."
                        )
                    else:
                        logger.info(
                            f"Auto-discovered frontend metrics URL: {config.load_router_metrics_url}"
                        )

                self.prometheus_engine_client = DirectRouterMetricsClient(
                    config.load_router_metrics_url, config.namespace
                )
            self.cached_load_metrics = CachedLoadMetrics()

            from dynamo.planner.utils.load_based_regression import (
                LoadBasedRegressionModel,
            )

            if self.component_type == SubComponentType.PREFILL:
                self.ttft_regression = LoadBasedRegressionModel(
                    window_size=self.config.load_learning_window,
                    min_observations=self.config.load_min_observations,
                )
            elif self.component_type == SubComponentType.DECODE:
                self.itl_regression = LoadBasedRegressionModel(
                    window_size=self.config.load_learning_window,
                    min_observations=self.config.load_min_observations,
                )

    @property
    def last_metrics(self) -> Metrics:
        return self.shared_state.last_metrics

    @last_metrics.setter
    def last_metrics(self, value: Metrics) -> None:
        self.shared_state.last_metrics = value

    async def _async_init(self):
        """Async initialization for components that need it"""
        if (
            not self.dryrun
            and hasattr(self, "connector")
            and hasattr(self.connector, "_async_init")
        ):
            await self.connector._async_init()

    async def _get_model_name(self, require_prefill: bool, require_decode: bool) -> str:
        model_name = self.connector.get_model_name(
            require_prefill=require_prefill, require_decode=require_decode
        )
        if asyncio.iscoroutine(model_name):
            model_name = await model_name
        return model_name

    async def _get_or_create_client(self, component_name: str, endpoint_name: str):
        """Create a client for the given component and endpoint, with a brief sleep for state sync."""
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
                    self.prefill_component_name if require_prefill else None
                ),
                decode_component_name=(
                    self.decode_component_name if require_decode else None
                ),
            )
            num_p_workers = prefill_count if require_prefill else 0
            num_d_workers = decode_count if require_decode else 0
            return num_p_workers, num_d_workers, is_stable

        # Fall back to runtime client for non-Kubernetes environments
        if self.runtime is None:
            raise RuntimeError("Runtime is not initialized")

        worker_names = WORKER_COMPONENT_NAMES[self.config.backend]

        if require_prefill:
            try:
                if self.prefill_client is None:
                    self.prefill_client = await self._get_or_create_client(
                        worker_names.prefill_worker_component_name,
                        worker_names.prefill_worker_endpoint,
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
                    self.workers_client = await self._get_or_create_client(
                        worker_names.decode_worker_component_name,
                        worker_names.decode_worker_endpoint,
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
        self.last_metrics.ttft = (
            self.prometheus_traffic_client.get_avg_time_to_first_token(
                f"{self.config.throughput_adjustment_interval}s",
                self.model_name,
            )
            * 1000
        )
        self.last_metrics.itl = (
            self.prometheus_traffic_client.get_avg_inter_token_latency(
                f"{self.config.throughput_adjustment_interval}s",
                self.model_name,
            )
            * 1000
        )
        self.last_metrics.num_req = (
            self.prometheus_traffic_client.get_avg_request_count(
                f"{self.config.throughput_adjustment_interval}s",
                self.model_name,
            )
        )
        self.last_metrics.request_duration = (
            self.prometheus_traffic_client.get_avg_request_duration(
                f"{self.config.throughput_adjustment_interval}s",
                self.model_name,
            )
        )
        self.last_metrics.isl = (
            self.prometheus_traffic_client.get_avg_input_sequence_tokens(
                f"{self.config.throughput_adjustment_interval}s",
                self.model_name,
            )
        )
        self.last_metrics.osl = (
            self.prometheus_traffic_client.get_avg_output_sequence_tokens(
                f"{self.config.throughput_adjustment_interval}s",
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
        self.num_req_predictor.add_data_point(metrics.num_req)
        self.isl_predictor.add_data_point(metrics.isl)
        self.osl_predictor.add_data_point(metrics.osl)

    def predict_load(self):
        try:
            # predict the next load
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

    def dryrun_observe_traffic_stats(
        self, num_req: int, isl_avg: float, osl_avg: float
    ):
        self.num_req_predictor.add_data_point(num_req)
        self.isl_predictor.add_data_point(isl_avg)
        self.osl_predictor.add_data_point(osl_avg)

    def plan_adjustment(self) -> Optional[int]:
        # Skip adjustment if no traffic
        if not self.last_metrics.is_valid():
            logger.info(
                "Metrics contain None or NaN values (no active requests), skipping adjustment"
            )
            return None

        if not self.no_correction:
            try:
                if not self._update_correction_factor():
                    return None
            except Exception as e:
                logger.error(f"Failed to correct prediction factors: {e}")
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
    ) -> int:
        raise NotImplementedError

    def _update_correction_factor(self) -> bool:
        raise NotImplementedError

    def _component_name(self) -> str:
        if self.component_type == SubComponentType.PREFILL:
            return self.prefill_component_name
        return self.decode_component_name

    def _engine_num_gpu(self) -> int:
        if self.component_type == SubComponentType.PREFILL:
            assert self.config.prefill_engine_num_gpu is not None
            return self.config.prefill_engine_num_gpu
        assert self.config.decode_engine_num_gpu is not None
        return self.config.decode_engine_num_gpu

    def apply_component_budget(self, desired_replicas: int) -> int:
        return _apply_component_gpu_budget(
            desired_replicas, self._engine_num_gpu(), self.config
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

    async def _apply_scaling_blocking(self, desired_replicas: int) -> None:
        """Apply scaling with blocking=True (wait for deployment ready)."""
        if self.config.no_operation:
            return
        target_replicas = [
            TargetReplica(
                sub_component_type=self.component_type,
                component_name=self._component_name(),
                desired_replicas=desired_replicas,
            )
        ]
        await self.connector.set_component_replicas(target_replicas, blocking=True)

    async def observe_engine_load_stats(self) -> None:
        """Query DirectRouterMetricsClient for per-worker metrics, update regression."""
        worker_type = self.component_type.value  # "prefill" or "decode"
        result = self.prometheus_engine_client.get_recent_and_averaged_metrics(
            worker_type
        )
        if result is None:
            logger.warning(
                f"No per-worker metrics available yet for {worker_type} (buffer empty)"
            )
            return

        recent, per_worker_averaged, cluster_averaged = result
        self.cached_load_metrics = CachedLoadMetrics(
            recent=recent,
            per_worker_averaged=per_worker_averaged,
            cluster_averaged=cluster_averaged,
        )

        if self.component_type == SubComponentType.PREFILL:
            for wid, m in recent.items():
                active_prefill = m.get("active_prefill_tokens", 0.0)
                last_isl = m.get("last_isl", 0.0)
                last_ttft = m.get("last_ttft", 0.0)
                if last_ttft > 0 and last_isl > 0:
                    x = active_prefill + last_isl
                    # last_ttft is in seconds from Prometheus, convert to ms
                    y = last_ttft * 1000
                    logger.info(
                        f"{SubComponentType.PREFILL.value} Worker {wid} observed status: TTFT {y:.2f}ms @ prefill tokens {x:.2f}"
                    )
                    self.ttft_regression.add_observation(x, y)

        elif self.component_type == SubComponentType.DECODE:
            for wid, m in recent.items():
                active_decode = m.get("active_decode_blocks", 0.0)
                last_itl = m.get("last_itl", 0.0)
                if last_itl > 0 and active_decode > 0:
                    x = active_decode
                    # last_itl is in seconds from Prometheus, convert to ms
                    y = last_itl * 1000
                    logger.info(
                        f"{SubComponentType.DECODE.value} Worker {wid} observed status: ITL {y:.2f}ms @ decode blocks {x:.2f}"
                    )
                    self.itl_regression.add_observation(x, y)

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

    async def _load_loop(self, require_prefill: bool, require_decode: bool) -> None:
        """Load-based scaling loop at shorter interval."""
        while True:
            await asyncio.sleep(self.config.load_adjustment_interval)
            logger.info("New load-based adjustment interval started!")

            # Query DGD for fresh worker counts
            num_p, num_d, _ = await self.get_workers_info(
                require_prefill=require_prefill, require_decode=require_decode
            )
            self.shared_state.num_p_workers = num_p
            self.shared_state.num_d_workers = num_d

            # Observe per-worker metrics from router
            await self.observe_engine_load_stats()

            # Reconcile DGD worker count with router Prometheus count
            prom_count = len(self.cached_load_metrics.recent)
            dgd_count = (
                num_p if self.component_type == SubComponentType.PREFILL else num_d
            )
            if prom_count != dgd_count:
                logger.warning(
                    f"Worker count mismatch: DGD reports {dgd_count} workers, "
                    f"router metrics reports {prom_count} workers. "
                    "Skipping load-based scaling adjustment."
                )
                continue

            desired_replicas = self.load_plan_adjustment()

            if desired_replicas is not None:
                # Enforce lower bound from throughput-based
                if self.enable_throughput:
                    if self.component_type == SubComponentType.PREFILL:
                        lower_bound = self.shared_state.throughput_lower_bound_p
                    else:
                        lower_bound = self.shared_state.throughput_lower_bound_d
                    desired_replicas = max(desired_replicas, lower_bound)
                desired_replicas = self.apply_component_budget(desired_replicas)
                self.update_predicted_replicas_metric(desired_replicas)
                # Load-based planner needs blocking scaling because it only checks
                # the current status of the engine, not the predicted load.
                # We need to wait for the deployment to be steady before making another one.
                await self._apply_scaling_blocking(desired_replicas)

    async def run(self):
        """Main loop for the planner"""
        require_prefill = self.component_type == SubComponentType.PREFILL
        require_decode = self.component_type == SubComponentType.DECODE

        if not self.config.no_operation:
            logger.info("Validating deployment...")
            await self.connector.validate_deployment(
                prefill_component_name=(
                    self.prefill_component_name if require_prefill else None
                ),
                decode_component_name=(
                    self.decode_component_name if require_decode else None
                ),
                require_prefill=require_prefill,
                require_decode=require_decode,
            )
            logger.info("Successfully validated the deployment")

            # Initialize GPU counts
            _initialize_gpu_counts(
                self.config,
                self.connector,
                require_prefill=require_prefill,
                require_decode=require_decode,
            )

            await self.connector.wait_for_deployment_ready()

        # Model name discovery runs in all modes (needed for metrics collection)
        if not self.config.no_operation:
            model_name = await self._get_model_name(
                require_prefill=require_prefill, require_decode=require_decode
            )
            logger.info(f"Detected model name from deployment: {model_name}")
            self.model_name = model_name.lower()
        else:
            model_name = getattr(self.config, "model_name", None)
            if not model_name:
                raise ValueError(
                    "Model name is required in no-operation mode. "
                    "Please set model_name in the config."
                )
            self.model_name = model_name.lower()

        self.shared_state.last_adjustment_time = time.time()
        self.shared_state.last_load_adjustment_time = time.time()

        # Build list of concurrent loops based on enabled scaling modes
        loops = []
        if self.enable_throughput:
            loops.append(self._throughput_loop(require_prefill, require_decode))
        if self.enable_load:
            loops.append(self._load_loop(require_prefill, require_decode))
            loops.append(
                self.prometheus_engine_client.run_sampling_loop(
                    self.config.load_metric_samples,
                    self.config.load_adjustment_interval,
                )
            )

        await asyncio.gather(*loops)
