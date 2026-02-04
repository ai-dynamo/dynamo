# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
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
from dynamo.planner.utils.exceptions import DeploymentValidationError
from dynamo.planner.utils.load_predictor import LOAD_PREDICTORS
from dynamo.planner.utils.perf_interpolation import (
    DecodeInterpolator,
    PrefillInterpolator,
)
from dynamo.planner.utils.pre_swept_results_utils import PreSweptResultsHelper
from dynamo.planner.utils.prometheus import PrometheusAPIClient
from dynamo.planner.utils.trace_data_extractor import extract_metrics_from_mooncake
from dynamo.runtime import DistributedRuntime
from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()
logger = logging.getLogger(__name__)


@dataclass
class Metrics:
    ttft: Optional[float] = None
    itl: Optional[float] = None
    num_req: Optional[float] = None
    isl: Optional[float] = None
    osl: Optional[float] = None
    request_duration: Optional[float] = None
    p_load: Optional[float] = None
    d_load: Optional[float] = None

    def is_valid(self) -> bool:
        """Check if all metrics are valid (not None and not NaN)."""
        return (
            self.ttft is not None
            and self.itl is not None
            and self.isl is not None
            and self.osl is not None
            and not math.isnan(self.ttft)
            and not math.isnan(self.itl)
            and not math.isnan(self.isl)
            and not math.isnan(self.osl)
        )


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
    p_endpoints: list = field(default_factory=list)
    d_endpoints: list = field(default_factory=list)
    cumulative_gpu_hours: float = 0.0
    last_adjustment_time: float = 0.0


def _apply_global_gpu_budget(
    next_num_p: int, next_num_d: int, args: argparse.Namespace
) -> tuple[int, int]:
    """Apply GPU budget constraint to both prefill and decode replicas.

    When total GPUs required (num_p * prefill_gpus + num_d * decode_gpus) exceeds the
    budget, scale down both proportionally using scale = budget / total_required. Prefill
    replicas are clamped to [min_endpoint, max_prefill] where max_prefill reserves enough
    GPUs for min_endpoint decode replicas. Remaining budget is then allocated to decode.
    Returns (0, 0) if budget cannot satisfy min_endpoint for both components.
    """
    if args.max_gpu_budget < 0:
        return next_num_p, next_num_d
    total_gpu_required = (
        next_num_p * args.prefill_engine_num_gpu
        + next_num_d * args.decode_engine_num_gpu
    )
    if total_gpu_required <= args.max_gpu_budget:
        return next_num_p, next_num_d
    min_required = (
        args.min_endpoint * args.prefill_engine_num_gpu
        + args.min_endpoint * args.decode_engine_num_gpu
    )
    if args.max_gpu_budget < min_required:
        logger.warning(
            f"max_gpu_budget ({args.max_gpu_budget}) is below the minimum required "
            f"for min_endpoint ({min_required}); enforcing zero replicas"
        )
        return 0, 0
    scale = args.max_gpu_budget / total_gpu_required
    max_prefill = math.floor(
        (args.max_gpu_budget - args.min_endpoint * args.decode_engine_num_gpu)
        / args.prefill_engine_num_gpu
    )
    next_num_p = max(
        args.min_endpoint, min(max_prefill, math.floor(next_num_p * scale))
    )
    remaining = args.max_gpu_budget - next_num_p * args.prefill_engine_num_gpu
    next_num_d = max(
        args.min_endpoint, math.floor(remaining / args.decode_engine_num_gpu)
    )
    logger.warning(
        f"Total number of GPUs required ({total_gpu_required}) exceeds the max GPU budget ({args.max_gpu_budget}), "
        f"scaling down to {next_num_p} prefill and {next_num_d} decode replicas"
    )
    return next_num_p, next_num_d


def _apply_component_gpu_budget(
    desired_replicas: int, engine_num_gpu: int, args: argparse.Namespace
) -> int:
    """Apply GPU budget constraint to a single component (prefill-only or decode-only).

    When total GPUs required (replicas * gpus_per_replica) exceeds the budget, scale down
    using scale = budget / total_required, floored and clamped to at least min_endpoint.
    Returns 0 if budget cannot satisfy min_endpoint replicas.
    """
    if args.max_gpu_budget < 0:
        return desired_replicas
    total_gpu_required = desired_replicas * engine_num_gpu
    if total_gpu_required <= args.max_gpu_budget:
        return desired_replicas
    min_required = args.min_endpoint * engine_num_gpu
    if args.max_gpu_budget < min_required:
        logger.warning(
            f"max_gpu_budget ({args.max_gpu_budget}) is below the minimum required "
            f"for min_endpoint ({min_required}); enforcing zero replicas"
        )
        return 0
    scale = args.max_gpu_budget / total_gpu_required
    next_num = max(args.min_endpoint, math.floor(desired_replicas * scale))
    logger.warning(
        f"Total number of GPUs required ({total_gpu_required}) exceeds the max GPU budget ({args.max_gpu_budget}), "
        f"scaling down to {next_num} replicas"
    )
    return next_num


def _initialize_gpu_counts(
    args: argparse.Namespace,
    connector,
    require_prefill: bool,
    require_decode: bool,
) -> None:
    """Initialize GPU counts from DGD (Kubernetes) or CLI args (virtual).

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
            args.prefill_engine_num_gpu = prefill_gpu
            args.decode_engine_num_gpu = decode_gpu
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
    if require_prefill and args.prefill_engine_num_gpu is None:
        errors.append("Missing --prefill-engine-num-gpu flag")
    if require_decode and args.decode_engine_num_gpu is None:
        errors.append("Missing --decode-engine-num-gpu flag")
    if errors:
        raise DeploymentValidationError(errors)
    logger.info(
        f"Using GPU counts from CLI: prefill={args.prefill_engine_num_gpu}, "
        f"decode={args.decode_engine_num_gpu}"
    )


class BasePlanner:
    component_type: SubComponentType

    def __init__(
        self,
        runtime: Optional[DistributedRuntime],
        args: argparse.Namespace,
        dryrun: bool = False,
        shared_state: Optional[PlannerSharedState] = None,
        prometheus_metrics: Optional[PlannerPrometheusMetrics] = None,
        prometheus_api_client: Optional[PrometheusAPIClient] = None,
        connector=None,
        start_prometheus_server: bool = True,
    ):
        self.args = args
        self.dryrun = dryrun
        self.shared_state = shared_state or PlannerSharedState()

        # Rely on getting model name from connector
        self.model_name: Optional[str] = None

        if not self.dryrun:
            self.runtime = runtime
            self.namespace = args.namespace

            if not args.no_operation:
                if connector is not None:
                    self.connector = connector
                elif args.environment == "kubernetes":
                    self.connector = KubernetesConnector(
                        self.namespace, self.model_name
                    )
                elif args.environment == "virtual":
                    self.connector = VirtualConnector(
                        runtime,
                        self.namespace,
                        args.model_name,
                    )
                else:
                    raise ValueError(f"Invalid environment: {args.environment}")

            self.prometheus_api_client = prometheus_api_client or PrometheusAPIClient(
                args.metric_pulling_prometheus_endpoint,
                args.namespace,
            )

        predictor_cls = LOAD_PREDICTORS[args.load_predictor]
        # Predictors read configuration from `args` directly.
        self.num_req_predictor = predictor_cls(args)
        self.isl_predictor = predictor_cls(args)
        self.osl_predictor = predictor_cls(args)

        # Optional warmup: preload predictors with historical observations from a
        # mooncake-style JSONL trace (request_count/avg_isl/avg_osl per interval).
        if getattr(args, "load_predictor_warmup_trace", None):
            warmup_trace = args.load_predictor_warmup_trace
            try:
                metrics = extract_metrics_from_mooncake(
                    warmup_trace, args.adjustment_interval
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

        if "use-pre-swept-results" in args.profile_results_dir:
            config_list = args.profile_results_dir.split(":")
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
            self.prefill_interpolator = PrefillInterpolator(args.profile_results_dir)
            self.decode_interpolator = DecodeInterpolator(args.profile_results_dir)

        self.prefill_component_name = WORKER_COMPONENT_NAMES[
            self.args.backend
        ].prefill_worker_k8s_name
        self.decode_component_name = WORKER_COMPONENT_NAMES[
            self.args.backend
        ].decode_worker_k8s_name

        if not self.dryrun:
            self.prefill_client = None
            self.workers_client = None

            self.prometheus_port = args.metric_reporting_prometheus_port

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
            self.no_correction = args.no_correction

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
        client = (
            await self.runtime.namespace(self.namespace)
            .component(component_name)
            .endpoint(endpoint_name)
            .client()
        )
        # TODO: remove this sleep after rust client() is blocking until watching state
        await asyncio.sleep(0.1)
        return client

    async def get_workers_info(
        self, require_prefill: bool = True, require_decode: bool = True
    ):
        if self.runtime is None:
            raise RuntimeError("Runtime is not initialized")

        p_endpoints = []
        d_endpoints = []
        worker_names = WORKER_COMPONENT_NAMES[self.args.backend]

        if require_prefill:
            try:
                if self.prefill_client is None:
                    self.prefill_client = await self._get_or_create_client(
                        worker_names.prefill_worker_component_name,
                        worker_names.prefill_worker_endpoint,
                    )
                p_endpoints = self.prefill_client.instance_ids()  # type: ignore
            except Exception:
                p_endpoints = []
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
                d_endpoints = self.workers_client.instance_ids()  # type: ignore
            except Exception as e:
                raise RuntimeError(f"Failed to get decode worker endpoints: {e}")

        return p_endpoints, d_endpoints

    async def observe_metrics(
        self, require_prefill: bool = True, require_decode: bool = True
    ):
        p_endpoints, d_endpoints = await self.get_workers_info(
            require_prefill=require_prefill, require_decode=require_decode
        )
        self.shared_state.p_endpoints = p_endpoints
        self.shared_state.d_endpoints = d_endpoints
        logger.debug(
            f"Number of prefill workers: {len(p_endpoints)}, number of decode workers: {len(d_endpoints)}"
        )

        # Update Prometheus metrics if server is running
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.num_p_workers.set(len(p_endpoints))
            self.prometheus_metrics.num_d_workers.set(len(d_endpoints))

            # Calculate and accumulate GPU hours for this interval
            # TODO: track startup and shutdown times to get more accurate GPU hours
            interval_gpu_hours = (
                (
                    len(p_endpoints) * self.args.prefill_engine_num_gpu
                    + len(d_endpoints) * self.args.decode_engine_num_gpu
                )
                * self.args.adjustment_interval
                / 3600
            )
            self.shared_state.cumulative_gpu_hours += interval_gpu_hours
            self.prometheus_metrics.gpu_hours.set(
                self.shared_state.cumulative_gpu_hours
            )

        # Prometheus returns seconds, convert to milliseconds
        self.last_metrics.ttft = (
            self.prometheus_api_client.get_avg_time_to_first_token(
                f"{self.args.adjustment_interval}s",
                self.model_name,
            )
            * 1000
        )
        self.last_metrics.itl = (
            self.prometheus_api_client.get_avg_inter_token_latency(
                f"{self.args.adjustment_interval}s",
                self.model_name,
            )
            * 1000
        )
        self.last_metrics.num_req = self.prometheus_api_client.get_avg_request_count(
            f"{self.args.adjustment_interval}s",
            self.model_name,
        )
        self.last_metrics.request_duration = (
            self.prometheus_api_client.get_avg_request_duration(
                f"{self.args.adjustment_interval}s",
                self.model_name,
            )
        )
        self.last_metrics.isl = (
            self.prometheus_api_client.get_avg_input_sequence_tokens(
                f"{self.args.adjustment_interval}s",
                self.model_name,
            )
        )
        self.last_metrics.osl = (
            self.prometheus_api_client.get_avg_output_sequence_tokens(
                f"{self.args.adjustment_interval}s",
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
                self.last_metrics.num_req / self.args.adjustment_interval
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

    def dryrun_observe_metrics(self, num_req: int, isl_avg: float, osl_avg: float):
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
                next_num_req / self.args.adjustment_interval
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
            return self.args.prefill_engine_num_gpu
        return self.args.decode_engine_num_gpu

    def apply_component_budget(self, desired_replicas: int) -> int:
        return _apply_component_gpu_budget(
            desired_replicas, self._engine_num_gpu(), self.args
        )

    async def _apply_scaling(self, desired_replicas: int) -> None:
        if self.args.no_operation:
            return
        target_replicas = [
            TargetReplica(
                sub_component_type=self.component_type,
                component_name=self._component_name(),
                desired_replicas=desired_replicas,
            )
        ]
        await self.connector.set_component_replicas(target_replicas, blocking=False)

    async def run(self):
        """Main loop for the planner"""
        if not self.args.no_operation:
            logger.info("Validating deployment...")
            require_prefill = self.component_type == SubComponentType.PREFILL
            require_decode = self.component_type == SubComponentType.DECODE
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
                self.args,
                self.connector,
                require_prefill=require_prefill,
                require_decode=require_decode,
            )

            await self.connector.wait_for_deployment_ready()

            model_name = await self._get_model_name(
                require_prefill=require_prefill, require_decode=require_decode
            )
            logger.info(f"Detected model name from deployment: {model_name}")
            self.model_name = (
                model_name.lower()
            )  # normalize model name to lowercase (MDC)

        self.shared_state.last_adjustment_time = time.time()

        while True:
            current_time = time.time()

            if (
                current_time - self.shared_state.last_adjustment_time
                >= self.args.adjustment_interval
            ):
                self.shared_state.last_adjustment_time = time.time()
                logger.info("New adjustment interval started!")

                await self.observe_metrics(
                    require_prefill=require_prefill, require_decode=require_decode
                )
                desired_replicas = self.plan_adjustment()
                if desired_replicas is not None:
                    desired_replicas = self.apply_component_budget(desired_replicas)
                    self.update_predicted_replicas_metric(desired_replicas)
                    await self._apply_scaling(desired_replicas)

            # sleep for a while to avoid busy-waiting but not too long to miss the next adjustment
            await asyncio.sleep(self.args.adjustment_interval / 10)


class PrefillPlanner(BasePlanner):
    component_type = SubComponentType.PREFILL

    def _update_correction_factor(self) -> bool:
        expect_ttft = self.prefill_interpolator.interpolate_ttft(self.last_metrics.isl)
        self.p_correction_factor = self.last_metrics.ttft / expect_ttft
        logger.info(f"Correction factor (prefill TTFT): {self.p_correction_factor:.3f}")
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.p_correction_factor.set(self.p_correction_factor)
        return True

    def _compute_replica_requirements(
        self, next_num_req: float, next_isl: float, next_osl: float
    ) -> int:
        pred_prefill_throughput = (
            next_num_req
            * next_isl
            / self.args.adjustment_interval
            * min(1, self.p_correction_factor)
        )
        p_thpt_per_gpu = self.prefill_interpolator.interpolate_thpt_per_gpu(next_isl)
        next_num_p = math.ceil(
            pred_prefill_throughput / p_thpt_per_gpu / self.args.prefill_engine_num_gpu
        )
        next_num_p = max(next_num_p, self.args.min_endpoint)
        logger.info(
            f"Prefill calculation: {pred_prefill_throughput:.2f}(p_thpt) / "
            f"{p_thpt_per_gpu * self.args.prefill_engine_num_gpu:.2f}(p_engine_cap) = "
            f"{next_num_p}(num_p)"
        )
        return next_num_p

    def update_predicted_replicas_metric(self, desired_replicas: int) -> None:
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.predicted_num_p.set(desired_replicas)


class DecodePlanner(BasePlanner):
    component_type = SubComponentType.DECODE

    def _update_correction_factor(self) -> bool:
        if not self.shared_state.d_endpoints:
            logger.warning(
                "No decode workers found for correction factor, skipping correction update"
            )
            return True
        expect_itl = self.decode_interpolator.interpolate_itl(
            concurrency=self.last_metrics.num_req  # type: ignore
            / len(self.shared_state.d_endpoints)
            * self.last_metrics.request_duration  # type: ignore
            / self.args.adjustment_interval,
            context_length=self.last_metrics.isl + self.last_metrics.osl / 2,  # type: ignore
        )
        self.d_correction_factor = self.last_metrics.itl / expect_itl
        logger.info(f"Correction factor (decode ITL): {self.d_correction_factor:.3f}")
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.d_correction_factor.set(self.d_correction_factor)
        return True

    def _compute_replica_requirements(
        self, next_num_req: float, next_isl: float, next_osl: float
    ) -> int:
        if self.d_correction_factor <= 0:
            logger.warning(
                f"d_correction_factor is {self.d_correction_factor}, using default value of 1.0"
            )
            corrected_itl = self.args.itl
        else:
            corrected_itl = self.args.itl / self.d_correction_factor
        (
            pred_decode_thpt_per_gpu,
            _,
            _,
        ) = self.decode_interpolator.find_best_throughput_per_gpu(
            itl=corrected_itl, context_length=next_isl + next_osl / 2
        )
        pred_decode_throughput = next_num_req * next_osl / self.args.adjustment_interval
        next_num_d = math.ceil(
            pred_decode_throughput
            / pred_decode_thpt_per_gpu
            / self.args.decode_engine_num_gpu
        )
        next_num_d = max(next_num_d, self.args.min_endpoint)
        logger.info(
            f"Decode calculation: {pred_decode_throughput:.2f}(d_thpt) / "
            f"{pred_decode_thpt_per_gpu * self.args.decode_engine_num_gpu:.2f}(d_engine_cap) = "
            f"{next_num_d}(num_d)"
        )
        return next_num_d

    def update_predicted_replicas_metric(self, desired_replicas: int) -> None:
        if self.prometheus_port != 0 and self.prometheus_metrics is not None:
            self.prometheus_metrics.predicted_num_d.set(desired_replicas)


class DisaggPlanner:
    def __init__(
        self, runtime: Optional[DistributedRuntime], args: argparse.Namespace
    ) -> None:
        self.args = args
        self.shared_state = PlannerSharedState()
        prometheus_metrics = PlannerPrometheusMetrics()

        self.prefill_planner = PrefillPlanner(
            runtime,
            args,
            shared_state=self.shared_state,
            prometheus_metrics=prometheus_metrics,
            start_prometheus_server=True,
        )
        self.decode_planner = DecodePlanner(
            runtime,
            args,
            shared_state=self.shared_state,
            prometheus_metrics=prometheus_metrics,
            prometheus_api_client=getattr(
                self.prefill_planner, "prometheus_api_client", None
            ),
            connector=getattr(self.prefill_planner, "connector", None),
            start_prometheus_server=False,
        )

    async def _async_init(self):
        # Prefill/Decode share the same connector instance in disagg mode.
        await self.prefill_planner._async_init()

    async def run(self):
        if not self.args.no_operation:
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
                self.args,
                self.prefill_planner.connector,
                require_prefill=True,
                require_decode=True,
            )

            await self.prefill_planner.connector.wait_for_deployment_ready()

            model_name = await self.prefill_planner._get_model_name(
                require_prefill=True, require_decode=True
            )
            logger.info(f"Detected model name from deployment: {model_name}")
            model_name = model_name.lower()
            self.prefill_planner.model_name = model_name
            self.decode_planner.model_name = model_name

        self.shared_state.last_adjustment_time = time.time()

        while True:
            current_time = time.time()

            if (
                current_time - self.shared_state.last_adjustment_time
                >= self.args.adjustment_interval
            ):
                self.shared_state.last_adjustment_time = time.time()
                logger.info("New adjustment interval started!")

                await self.prefill_planner.observe_metrics(
                    require_prefill=True, require_decode=True
                )
                self.decode_planner.update_predictors_from_metrics(
                    self.shared_state.last_metrics
                )
                next_num_p = self.prefill_planner.plan_adjustment()
                next_num_d = self.decode_planner.plan_adjustment()
                if next_num_p is None or next_num_d is None:
                    continue

                next_num_p, next_num_d = _apply_global_gpu_budget(
                    next_num_p, next_num_d, self.args
                )
                self.prefill_planner.update_predicted_replicas_metric(next_num_p)
                self.decode_planner.update_predicted_replicas_metric(next_num_d)

                if not self.args.no_operation:
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

            # sleep for a while to avoid busy-waiting but not too long to miss the next adjustment
            await asyncio.sleep(self.args.adjustment_interval / 10)


async def start_sla_planner(runtime: DistributedRuntime, args: argparse.Namespace):
    mode = getattr(args, "mode", "disagg")
    if mode == "disagg":
        planner = DisaggPlanner(runtime, args)
    elif mode == "prefill":
        planner = PrefillPlanner(runtime, args)
    elif mode == "decode":
        planner = DecodePlanner(runtime, args)
    else:
        raise ValueError(f"Invalid planner mode: {mode}")
    await planner._async_init()
    await planner.run()
