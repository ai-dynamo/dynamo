# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import datetime
import json
import logging
import math
import os
import threading
import time
from http.server import ThreadingHTTPServer
from typing import TYPE_CHECKING, Optional, Union

from prometheus_client.exposition import MetricsHandler

from dynamo.planner.config.backend_components import WORKER_COMPONENT_NAMES
from dynamo.planner.config.defaults import ScalingMode, SubComponentType, TargetReplica
from dynamo.planner.config.planner_config import PlannerConfig
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.connectors.virtual import VirtualConnector
from dynamo.planner.core.budget import (
    _apply_component_gpu_budget,
    _initialize_gpu_counts,
)
from dynamo.planner.core.load.predictors import LOAD_PREDICTORS
from dynamo.planner.core.state import PlannerSharedState
from dynamo.planner.core.throughput.interpolation import (
    DecodeInterpolator,
    PrefillInterpolator,
)
from dynamo.planner.core.throughput.pre_swept_results import PreSweptResultsHelper
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

# Module-level advisory status state (updated by BasePlanner, read by HTTP handler)
_advisory_status_state: dict = {}


class _PlannerHTTPHandler(MetricsHandler):
    """HTTP handler: /metrics for Prometheus, /advisory/status for advisory state."""

    def do_GET(self):
        if self.path == "/advisory/status":
            body = json.dumps(_advisory_status_state).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()

    def log_message(self, format, *args):
        pass  # Suppress access log noise


def _start_planner_http_server(port: int) -> None:
    """Start the Planner HTTP server (Prometheus metrics + /advisory/status)."""
    server = ThreadingHTTPServer(("", port), _PlannerHTTPHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    logger.info(f"Started Planner HTTP server on port {port}")


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
        connector: Optional[ConnectorType] = None,
        start_prometheus_server: bool = True,
        component_type: Optional[SubComponentType] = None,
    ):
        if component_type is not None:
            self.component_type = component_type

        self.config = config
        self.dryrun = dryrun
        self.shared_state = shared_state or PlannerSharedState()

        # Rely on getting model name from connector
        self.model_name: Optional[str] = None

        if not self.dryrun:
            self.runtime = runtime
            self.namespace = config.namespace
            self.connector: ConnectorType

            if config.effective_scaling_mode != ScalingMode.NOOP:
                # Initialize connector based on environment
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
                    self.connector = KubernetesConnector(
                        self.namespace, self.model_name
                    )
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

        # WorkerInfo: finalized by _init_worker_info() at the start of run().
        # Empty placeholders until then.
        self.prefill_worker_info = WorkerInfo()
        self.decode_worker_info = WorkerInfo()

        self.prometheus_metrics: PlannerPrometheusMetrics | None = None
        if not self.dryrun:
            self.prefill_client = None
            self.workers_client = None

            self.prometheus_port = config.metric_reporting_prometheus_port

            if prometheus_metrics is None:
                self.prometheus_metrics = PlannerPrometheusMetrics()
            else:
                self.prometheus_metrics = prometheus_metrics

            # Start Planner HTTP server if port is specified
            if start_prometheus_server and self.prometheus_port != 0:
                try:
                    _start_planner_http_server(self.prometheus_port)
                except Exception as e:
                    logger.error(f"Failed to start Planner HTTP server: {e}")

            # Startup self-check log
            if start_prometheus_server:
                port_source = (
                    "env PLANNER_PROMETHEUS_PORT"
                    if os.environ.get("PLANNER_PROMETHEUS_PORT")
                    else "config"
                )
                file_output_status = (
                    f"enabled → {config.log_dir}/advisory_history.jsonl"
                    if config.advisory_file_output and config.log_dir
                    else "disabled"
                )
                logger.info(
                    "[ADVISORY] Self-check:\n"
                    f"  - Scaling mode: {config.effective_scaling_mode.value}\n"
                    f"  - Metrics server: port {self.prometheus_port} (source: {port_source})\n"
                    f"  - Metrics endpoint: /metrics (15 advisory metrics registered)\n"
                    f"  - Advisory status endpoint: /advisory/status\n"
                    f"  - Advisory file output: {file_output_status}"
                )
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
            from dynamo.planner.core.load.fpm_regression import (
                DecodeRegressionModel,
                PrefillRegressionModel,
            )

            self.fpm_subscriber: "Optional[FpmEventSubscriber]" = None

            if self.component_type == SubComponentType.PREFILL:
                self.ttft_regression = PrefillRegressionModel(
                    window_size=self.config.load_learning_window,
                    min_observations=self.config.load_min_observations,
                )
            elif self.component_type == SubComponentType.DECODE:
                self.itl_regression = DecodeRegressionModel(
                    window_size=self.config.load_learning_window,
                    min_observations=self.config.load_min_observations,
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
            no_operation=self.config.effective_scaling_mode == ScalingMode.NOOP,
        )
        # model_name is resolved and written into both WorkerInfo objects
        self.model_name = (
            self.decode_worker_info.model_name or self.prefill_worker_info.model_name
        )

    async def _async_init(self):
        """Async initialization: connector init, deployment validation, WorkerInfo."""
        if (
            not self.dryrun
            and hasattr(self, "connector")
            and hasattr(self.connector, "_async_init")
        ):
            await self.connector._async_init()

        require_prefill = self.component_type == SubComponentType.PREFILL
        require_decode = self.component_type == SubComponentType.DECODE

        if not self.dryrun and not (
            self.config.effective_scaling_mode == ScalingMode.NOOP
        ):
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

        # Start FPM tracking if load-based scaling is enabled.
        # The subscriber auto-discovers FPM publishers for this component.
        if self.enable_load and self.runtime is not None:
            await self._init_fpm_subscriber()

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

    def dryrun_observe_traffic_stats(
        self, num_req: int, isl_avg: float, osl_avg: float
    ):
        self.num_req_predictor.add_data_point(num_req)
        self.isl_predictor.add_data_point(isl_avg)
        self.osl_predictor.add_data_point(osl_avg)

    def plan_adjustment(self) -> Optional[int]:
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
        if self.config.effective_scaling_mode != ScalingMode.ACTIVE:
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
        """Apply scaling without blocking so the loop continues observing metrics."""
        if self.config.effective_scaling_mode != ScalingMode.ACTIVE:
            return
        target_replicas = [
            TargetReplica(
                sub_component_type=self.component_type,
                component_name=self._component_name(),
                desired_replicas=desired_replicas,
            )
        ]
        await self.connector.set_component_replicas(target_replicas, blocking=False)

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

    # -------------------------------------------------------------------------
    # Advisory Engine methods (active + advisory modes)
    # -------------------------------------------------------------------------

    @staticmethod
    def _safe_gauge_set(gauge, value) -> None:
        """NaN/None-safe Prometheus gauge write."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            gauge.set(float("nan"))
        else:
            gauge.set(value)

    def _emit_advisory_metrics(
        self,
        recommended_p: int,
        recommended_d: int,
        source: str,
    ) -> None:
        """Emit advisory metrics (Layer 1 + 2 + JSONL).

        source: "throughput" or "load"
        """
        if self.prometheus_metrics is None or self.prometheus_port == 0:
            return

        # Metrics validity gate
        if not self.shared_state.last_metrics.is_valid():
            logger.info(
                "[ADVISORY] Skipping advisory output: "
                "metrics contain None/NaN (no active traffic or Prometheus unreachable)"
            )
            return

        # Non-negative guard
        recommended_p = max(0, recommended_p)
        recommended_d = max(0, recommended_d)

        current_p = self.shared_state.num_p_workers
        current_d = self.shared_state.num_d_workers
        delta_p = recommended_p - current_p
        delta_d = recommended_d - current_d

        # Anomaly detection
        threshold = self.config.advisory_anomaly_threshold
        if abs(delta_p) > threshold or abs(delta_d) > threshold:
            logger.warning(
                f"[ADVISORY] Unusually large delta: delta_p={delta_p}, delta_d={delta_d} "
                f"(threshold={threshold}). Possible metrics jump or configuration error."
            )

        # Determine aggregate action and reason code
        if delta_p > 0 or delta_d > 0:
            action = 1
            reason_code = 1 if source == "throughput" else 3
        elif delta_p < 0 or delta_d < 0:
            action = -1
            reason_code = 2 if source == "throughput" else 4
        else:
            action = 0
            reason_code = 6

        m = self.prometheus_metrics
        # Update counters
        if action == 1:
            m.advisory_scaleup_total.inc()
        elif action == -1:
            m.advisory_scaledown_total.inc()
        else:
            m.advisory_hold_total.inc()

        # Update gauges
        m.advisory_recommended_p.set(recommended_p)
        m.advisory_recommended_d.set(recommended_d)
        m.advisory_current_p.set(current_p)
        m.advisory_current_d.set(current_d)
        m.advisory_delta_p.set(delta_p)
        m.advisory_delta_d.set(delta_d)
        m.advisory_scaling_action.set(action)
        m.advisory_action_reason.set(reason_code)

        # SLA estimation (Layer 3)
        sla = self._estimate_sla_with_replicas(recommended_p, recommended_d)
        self._safe_gauge_set(m.advisory_est_ttft, sla.get("est_ttft"))
        self._safe_gauge_set(m.advisory_est_itl, sla.get("est_itl"))
        est_ttft = sla.get("est_ttft")
        est_itl = sla.get("est_itl")
        self._safe_gauge_set(
            m.advisory_ttft_headroom,
            self.config.ttft - est_ttft if est_ttft is not None else None,
        )
        self._safe_gauge_set(
            m.advisory_itl_headroom,
            self.config.itl - est_itl if est_itl is not None else None,
        )

        action_str = {1: "scale_up", 0: "hold", -1: "scale_down"}[action]

        # Update HTTP advisory status snapshot
        _advisory_status_state.update(
            {
                "scaling_mode": self.config.effective_scaling_mode.value,
                "last_update": datetime.datetime.utcnow().isoformat() + "Z",
                "current": {"prefill": current_p, "decode": current_d},
                "recommended": {"prefill": recommended_p, "decode": recommended_d},
                "delta": {"prefill": delta_p, "decode": delta_d},
                "action": action_str,
                "reason": source,
                "sla_estimation": {
                    "est_ttft_ms": est_ttft,
                    "est_itl_ms": est_itl,
                    "ttft_headroom_ms": (
                        self.config.ttft - est_ttft if est_ttft is not None else None
                    ),
                    "itl_headroom_ms": (
                        self.config.itl - est_itl if est_itl is not None else None
                    ),
                },
            }
        )

        # Structured logs for active + advisory modes (Layer 2)
        if self.config.effective_scaling_mode in (
            ScalingMode.ACTIVE,
            ScalingMode.ADVISORY,
        ):
            path = self._build_path_recommendation(
                current_p, current_d, recommended_p, recommended_d
            )
            log_data = {
                "event": "advisory_recommendation",
                "scaling_mode": self.config.effective_scaling_mode.value,
                "source": source,
                "action": action_str,
                "current": {"prefill": current_p, "decode": current_d},
                "recommended_final": {
                    "prefill": recommended_p,
                    "decode": recommended_d,
                },
                "path": path,
                "est_ttft_ms": est_ttft,
                "est_itl_ms": est_itl,
                "ttft_headroom_ms": (
                    self.config.ttft - est_ttft if est_ttft is not None else None
                ),
                "itl_headroom_ms": (
                    self.config.itl - est_itl if est_itl is not None else None
                ),
                "note": (
                    "Estimation based on profiling data; "
                    "actual values may differ due to runtime conditions"
                ),
            }
            logger.info("[ADVISORY] Recommendation", extra=log_data)

            if self.config.advisory_file_output:
                self._write_advisory_jsonl(
                    {
                        "ts": int(time.time()),
                        "mode": self.config.effective_scaling_mode.value,
                        "action": action_str,
                        "source": source,
                        "current": {"p": current_p, "d": current_d},
                        "recommended": {"p": recommended_p, "d": recommended_d},
                        "est_ttft": est_ttft,
                        "est_itl": est_itl,
                        "reason_code": reason_code,
                    }
                )

    def _estimate_sla_with_replicas(self, num_p: int, num_d: int) -> dict:
        """Estimate TTFT and ITL for the given replica counts using profiling data."""
        if not self.enable_throughput:
            if self.prometheus_metrics and self.prometheus_port != 0:
                self._safe_gauge_set(
                    self.prometheus_metrics.advisory_est_ttft, float("nan")
                )
                self._safe_gauge_set(
                    self.prometheus_metrics.advisory_est_itl, float("nan")
                )
                self._safe_gauge_set(
                    self.prometheus_metrics.advisory_ttft_headroom, float("nan")
                )
                self._safe_gauge_set(
                    self.prometheus_metrics.advisory_itl_headroom, float("nan")
                )
            logger.info(
                "[ADVISORY] SLA estimation unavailable: "
                "throughput scaling disabled (no profiling data)"
            )
            return {"est_ttft": None, "est_itl": None}

        try:
            isl = self.shared_state.last_metrics.isl or 3000.0
            osl = self.shared_state.last_metrics.osl or 150.0

            est_ttft = None
            if num_p > 0 and hasattr(self, "prefill_interpolator"):
                est_ttft = float(self.prefill_interpolator.interpolate_ttft(isl))

            est_itl = None
            if num_d > 0 and hasattr(self, "decode_interpolator"):
                _, est_itl_raw, _ = (
                    self.decode_interpolator.find_best_throughput_per_gpu(
                        self.config.itl, osl
                    )
                )
                est_itl = float(est_itl_raw)

            logger.debug(
                f"[ADVISORY] SLA estimate with P={num_p} D={num_d}: "
                f"TTFT={est_ttft}ms ITL={est_itl}ms"
            )
            return {"est_ttft": est_ttft, "est_itl": est_itl}
        except Exception as e:
            logger.warning(f"[ADVISORY] SLA estimation failed: {e}")
            return {"est_ttft": None, "est_itl": None}

    def _build_path_recommendation(
        self, current_p: int, current_d: int, target_p: int, target_d: int
    ) -> list:
        """Build incremental path from current to target replicas."""
        path = [f"{current_p}P{current_d}D"]
        max_step = self.config.advisory_max_step_size
        p, d = current_p, current_d
        while p != target_p or d != target_d:
            step_p = min(max_step, abs(target_p - p)) * (1 if target_p > p else -1)
            step_d = min(max_step, abs(target_d - d)) * (1 if target_d > d else -1)
            p += step_p
            d += step_d
            if p != target_p or d != target_d:
                path.append(f"{p}P{d}D (observe 1 interval)")
            else:
                path.append(f"{p}P{d}D")
        return path

    def _write_advisory_jsonl(self, data: dict) -> None:
        """Append one JSON line to the advisory JSONL file."""
        if not self.config.log_dir:
            return

        os.makedirs(self.config.log_dir, exist_ok=True)
        filepath = os.path.join(self.config.log_dir, "advisory_history.jsonl")
        try:
            with open(filepath, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            logger.warning(f"[ADVISORY] Failed to write advisory JSONL: {e}")

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
                        # Emit advisory metrics (active + advisory modes)
                        if self.config.effective_scaling_mode in (
                            ScalingMode.ACTIVE,
                            ScalingMode.ADVISORY,
                        ):
                            if self.component_type == SubComponentType.PREFILL:
                                self._emit_advisory_metrics(
                                    desired_replicas,
                                    self.shared_state.num_d_workers,
                                    "throughput",
                                )
                            else:
                                self._emit_advisory_metrics(
                                    self.shared_state.num_p_workers,
                                    desired_replicas,
                                    "throughput",
                                )
                        # Throughput planner does not needs blocking scaling because it monitors
                        # and predicts the load, not relying on the current status of the engine.
                        await self._apply_scaling(desired_replicas)

            await asyncio.sleep(self.config.throughput_adjustment_interval / 10)

    async def _load_loop(self, require_prefill: bool, require_decode: bool) -> None:
        """Load-based scaling loop at shorter interval.

        Uses FPM stats from the event plane (via FpmEventSubscriber) instead
        of scraping the router's /metrics endpoint.
        """
        pending_desired: Optional[int] = None
        while True:
            await asyncio.sleep(self.config.load_adjustment_interval)
            logger.info("New load-based adjustment interval started!")

            # Query DGD for fresh worker counts
            num_p, num_d, is_stable = await self.get_workers_info(
                require_prefill=require_prefill, require_decode=require_decode
            )
            self.shared_state.num_p_workers = num_p
            self.shared_state.num_d_workers = num_d

            # Always observe FPM stats and update regression, even during scaling.
            fpm_stats = self.observe_fpm_load_stats()
            if not fpm_stats:
                continue

            # If a previous scaling action is still in progress, skip decisions.
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
                # Enforce lower bound from throughput-based
                if self.enable_throughput:
                    if self.component_type == SubComponentType.PREFILL:
                        lower_bound = self.shared_state.throughput_lower_bound_p
                    else:
                        lower_bound = self.shared_state.throughput_lower_bound_d
                    desired_replicas = max(desired_replicas, lower_bound)
                desired_replicas = self.apply_component_budget(desired_replicas)
                self.update_predicted_replicas_metric(desired_replicas)
                # Emit advisory metrics (active + advisory modes)
                if self.config.effective_scaling_mode in (
                    ScalingMode.ACTIVE,
                    ScalingMode.ADVISORY,
                ):
                    if self.component_type == SubComponentType.PREFILL:
                        self._emit_advisory_metrics(
                            desired_replicas,
                            self.shared_state.num_d_workers,
                            "load",
                        )
                    else:
                        self._emit_advisory_metrics(
                            self.shared_state.num_p_workers,
                            desired_replicas,
                            "load",
                        )
                if self.config.effective_scaling_mode == ScalingMode.ACTIVE:
                    pending_desired = desired_replicas
                    await self._apply_scaling_blocking(desired_replicas)

    async def run(self):
        """Main scaling loop. Call _async_init() before this."""
        require_prefill = self.component_type == SubComponentType.PREFILL
        require_decode = self.component_type == SubComponentType.DECODE

        self.shared_state.last_adjustment_time = time.time()
        self.shared_state.last_load_adjustment_time = time.time()

        # Build list of concurrent loops based on enabled scaling modes.
        # FPM tracking (started in _async_init) replaces the former
        # DirectRouterMetricsClient.run_sampling_loop().
        loops = []
        if self.enable_throughput:
            loops.append(self._throughput_loop(require_prefill, require_decode))
        if self.enable_load:
            loops.append(self._load_loop(require_prefill, require_decode))

        await asyncio.gather(*loops)
