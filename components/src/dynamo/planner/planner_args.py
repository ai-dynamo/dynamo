# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SLA Planner configuration ArgGroup and Config."""

import logging
from typing import Optional

from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.config_base import ConfigBase
from dynamo.common.configuration.utils import add_argument, add_negatable_bool_argument
from dynamo.planner.defaults import SLAPlannerDefaults

logger = logging.getLogger(__name__)


class PlannerArgGroup(ArgGroup):
    """SLA Planner configuration parameters."""

    name = "sla-planner"

    def add_arguments(self, parser) -> None:
        """Add SLA planner arguments to parser."""
        g = parser.add_argument_group("SLA Planner Options")

        add_argument(
            g,
            flag_name="--environment",
            env_var="PLANNER_ENVIRONMENT",
            default=SLAPlannerDefaults.environment,
            help="Environment type",
            choices=["kubernetes", "virtual"],
        )
        add_argument(
            g,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default=SLAPlannerDefaults.namespace,
            help="Dynamo namespace",
        )
        add_argument(
            g,
            flag_name="--backend",
            env_var="PLANNER_BACKEND",
            default=SLAPlannerDefaults.backend,
            help="Backend type",
            choices=["vllm", "sglang", "trtllm", "mocker"],
        )
        add_argument(
            g,
            flag_name="--mode",
            env_var="PLANNER_MODE",
            default=SLAPlannerDefaults.mode,
            help="Planner mode: disagg (prefill+decode), prefill-only, decode-only, or agg (aggregated)",
            choices=["disagg", "prefill", "decode", "agg"],
        )
        add_argument(
            g,
            flag_name="--no-operation",
            env_var="PLANNER_NO_OPERATION",
            default=SLAPlannerDefaults.no_operation,
            help="Enable no-operation mode",
            arg_type=None,
            action="store_true",
        )
        add_argument(
            g,
            flag_name="--log-dir",
            env_var="PLANNER_LOG_DIR",
            default=SLAPlannerDefaults.log_dir,
            help="Log directory path",
        )
        add_argument(
            g,
            flag_name="--adjustment-interval",
            env_var="PLANNER_ADJUSTMENT_INTERVAL",
            default=SLAPlannerDefaults.adjustment_interval,
            help="Adjustment interval in seconds",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--max-gpu-budget",
            env_var="PLANNER_MAX_GPU_BUDGET",
            default=SLAPlannerDefaults.max_gpu_budget,
            help="Maximum GPU budget (-1 for no budget enforcement)",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--min-endpoint",
            env_var="PLANNER_MIN_ENDPOINT",
            default=SLAPlannerDefaults.min_endpoint,
            help="Minimum number of endpoints",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--decode-engine-num-gpu",
            env_var="PLANNER_DECODE_ENGINE_NUM_GPU",
            default=None,
            help="Number of GPUs per decode engine. In Kubernetes mode, this is auto-detected "
            "from DGD resources but can be overridden (e.g., for mockers without GPU resources).",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--prefill-engine-num-gpu",
            env_var="PLANNER_PREFILL_ENGINE_NUM_GPU",
            default=None,
            help="Number of GPUs per prefill engine. In Kubernetes mode, this is auto-detected "
            "from DGD resources but can be overridden (e.g., for mockers without GPU resources).",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--profile-results-dir",
            env_var="PLANNER_PROFILE_RESULTS_DIR",
            default=SLAPlannerDefaults.profile_results_dir,
            help="Profile results directory or 'use-pre-swept-results:<gpu_type>:<framework>:<model>:<tp>:<dp>:<pp>:<block_size>:<max_batch_size>:<gpu_count>' to use pre-swept results from pre_swept_results directory",
        )
        add_argument(
            g,
            flag_name="--ttft",
            env_var="PLANNER_TTFT",
            default=SLAPlannerDefaults.ttft,
            help="Time to first token (float, in milliseconds)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--itl",
            env_var="PLANNER_ITL",
            default=SLAPlannerDefaults.itl,
            help="Inter-token latency (float, in milliseconds)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--load-predictor",
            env_var="PLANNER_LOAD_PREDICTOR",
            default=SLAPlannerDefaults.load_predictor,
            help="Load predictor type (constant, arima, kalman, prophet)",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--load-predictor-log1p",
            env_var="PLANNER_LOAD_PREDICTOR_LOG1P",
            default=SLAPlannerDefaults.load_predictor_log1p,
            help="Model log1p(y) instead of y in the selected load predictor (ARIMA/Kalman/Prophet)",
        )
        add_argument(
            g,
            flag_name="--prophet-window-size",
            env_var="PLANNER_PROPHET_WINDOW_SIZE",
            default=SLAPlannerDefaults.prophet_window_size,
            help="Prophet history window size",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--load-predictor-warmup-trace",
            env_var="PLANNER_LOAD_PREDICTOR_WARMUP_TRACE",
            default=None,
            help="Optional path to a mooncake-style JSONL trace file used to warm up load predictors before observing live traffic",
        )
        add_argument(
            g,
            flag_name="--kalman-q-level",
            env_var="PLANNER_KALMAN_Q_LEVEL",
            default=SLAPlannerDefaults.kalman_q_level,
            help="Kalman process noise for level (higher = more responsive)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--kalman-q-trend",
            env_var="PLANNER_KALMAN_Q_TREND",
            default=SLAPlannerDefaults.kalman_q_trend,
            help="Kalman process noise for trend (higher = faster trend changes)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--kalman-r",
            env_var="PLANNER_KALMAN_R",
            default=SLAPlannerDefaults.kalman_r,
            help="Kalman measurement noise (lower = remember less / react more to new measurements)",
            arg_type=float,
        )
        add_argument(
            g,
            flag_name="--kalman-min-points",
            env_var="PLANNER_KALMAN_MIN_POINTS",
            default=SLAPlannerDefaults.kalman_min_points,
            help="Minimum number of points before Kalman predictor returns forecasts",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--metric-pulling-prometheus-endpoint",
            env_var="PROMETHEUS_ENDPOINT",
            default=SLAPlannerDefaults.metric_pulling_prometheus_endpoint,
            help="Prometheus endpoint URL for pulling dynamo deployment metrics",
        )
        add_argument(
            g,
            flag_name="--metric-reporting-prometheus-port",
            env_var="PLANNER_PROMETHEUS_PORT",
            default=SLAPlannerDefaults.metric_reporting_prometheus_port,
            help="Port for exposing planner's own metrics to Prometheus",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--no-correction",
            env_var="PLANNER_NO_CORRECTION",
            default=SLAPlannerDefaults.no_correction,
            help="Disable correction factor",
            arg_type=None,
            action="store_true",
        )
        add_argument(
            g,
            flag_name="--model-name",
            env_var="PLANNER_MODEL_NAME",
            default=None,
            help="Model name of deployment (only required for virtual environment)",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-throughput-scaling",
            env_var="PLANNER_ENABLE_THROUGHPUT_SCALING",
            default=SLAPlannerDefaults.enable_throughput_scaling,
            help="Enable throughput-based scaling",
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-loadbased-scaling",
            env_var="PLANNER_ENABLE_LOADBASED_SCALING",
            default=SLAPlannerDefaults.enable_loadbased_scaling,
            help="Enable load-based scaling",
        )

        # Load-based scaling settings
        add_argument(
            g,
            flag_name="--loadbased-router-metrics-url",
            env_var="PLANNER_LOADBASED_ROUTER_METRICS_URL",
            default=SLAPlannerDefaults.loadbased_router_metrics_url,
            help="URL to router's /metrics endpoint for direct load metric queries (default: auto-discovered from the DGD)",
        )
        add_argument(
            g,
            flag_name="--loadbased-adjustment-interval",
            env_var="PLANNER_LOADBASED_ADJUSTMENT_INTERVAL",
            default=SLAPlannerDefaults.loadbased_adjustment_interval,
            help="Load-based adjustment interval in seconds (must be < --adjustment-interval)",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--loadbased-learning-window",
            env_var="PLANNER_LOADBASED_LEARNING_WINDOW",
            default=SLAPlannerDefaults.loadbased_learning_window,
            help="Sliding window size for load-based regression (number of observations)",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--loadbased-scaling-down-sensitivity",
            env_var="PLANNER_LOADBASED_SCALING_DOWN_SENSITIVITY",
            default=SLAPlannerDefaults.loadbased_scaling_down_sensitivity,
            help="Scale-down sensitivity 0-100 (0=never scale down, 100=aggressive)",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--loadbased-metric-samples",
            env_var="PLANNER_LOADBASED_METRIC_SAMPLES",
            default=SLAPlannerDefaults.loadbased_metric_samples,
            help="Number of metric samples to average per load-based adjustment interval",
            arg_type=int,
        )
        add_argument(
            g,
            flag_name="--loadbased-min-observations",
            env_var="PLANNER_LOADBASED_MIN_OBSERVATIONS",
            default=SLAPlannerDefaults.loadbased_min_observations,
            help="Minimum regression observations before load-based scaling starts (cold start)",
            arg_type=int,
        )


class PlannerConfig(ConfigBase):
    """Configuration for SLA Planner (parsed from CLI / env)."""

    namespace: str
    environment: str
    backend: str
    mode: str
    no_operation: bool
    log_dir: Optional[str] = None
    adjustment_interval: int
    max_gpu_budget: int
    min_endpoint: int
    decode_engine_num_gpu: Optional[int] = None
    prefill_engine_num_gpu: Optional[int] = None
    profile_results_dir: str
    ttft: float
    itl: float
    load_predictor: str
    load_predictor_log1p: bool
    prophet_window_size: int
    load_predictor_warmup_trace: Optional[str] = None
    kalman_q_level: float
    kalman_q_trend: float
    kalman_r: float
    kalman_min_points: int
    metric_pulling_prometheus_endpoint: str
    metric_reporting_prometheus_port: int
    no_correction: bool
    model_name: Optional[str] = None
    enable_throughput_scaling: bool
    enable_loadbased_scaling: bool
    loadbased_router_metrics_url: Optional[str] = None
    loadbased_adjustment_interval: int
    loadbased_learning_window: int
    loadbased_scaling_down_sensitivity: int
    loadbased_metric_samples: int
    loadbased_min_observations: int

    def validate(self) -> None:
        """Validate and normalize SLA planner arguments.

        Resolves conflicting flags, checks required arguments, and enforces
        constraints between related arguments. Should be called after parsing
        and before constructing any planner.

        Raises:
        ValueError: If argument constraints are violated
        """

        enable_throughput = getattr(self, "enable_throughput_scaling", True)
        enable_loadbased = getattr(self, "enable_loadbased_scaling", False)

        if not enable_throughput and not enable_loadbased:
            raise ValueError(
                "At least one scaling mode must be enabled "
                "(--enable-throughput-scaling or --enable-loadbased-scaling)"
            )

        if enable_loadbased:
            environment = getattr(self, "environment", "kubernetes")
            if (
                not getattr(self, "loadbased_router_metrics_url", None)
                and environment != "kubernetes"
            ):
                raise ValueError(
                    "--loadbased-router-metrics-url is required when "
                    "load-based scaling is enabled outside kubernetes mode"
                )

            if enable_throughput:
                if self.loadbased_adjustment_interval >= self.adjustment_interval:
                    raise ValueError(
                        f"--loadbased-adjustment-interval ({self.loadbased_adjustment_interval}s) "
                        f"must be shorter than --adjustment-interval ({self.adjustment_interval}s). "
                        "Load-based scaling is the fast reactive loop; throughput-based is the "
                        "slow predictive loop."
                    )

            if not getattr(self, "no_correction", False):
                logger.warning(
                    "Correction factor is automatically disabled when load-based "
                    "scaling is enabled. Load-based scaling already accounts for "
                    "actual latency conditions."
                )
                self.no_correction = True
