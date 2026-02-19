# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from dynamo.planner.defaults import SLAPlannerDefaults

logger = logging.getLogger(__name__)


class PlannerPreDeploymentSweepMode(str, Enum):
    None_ = "none"
    Rapid = "rapid"
    Thorough = "thorough"


class PlannerConfig(BaseModel):
    """Pydantic configuration for the Dynamo Planner.

    Replaces the argparse-based CLI. All fields mirror the former CLI flags
    with defaults sourced from SLAPlannerDefaults.
    """

    plannerPreDeploymentSweeping: Optional[PlannerPreDeploymentSweepMode] = Field(
        default=PlannerPreDeploymentSweepMode.Rapid,
        description='PlannerPreDeploymentSweeping controls pre-deployment sweeping mode for planner in-depth profiling. "none" means no pre-deployment sweep (only load-based scaling). "rapid" uses AI Configurator to simulate engine performance. "thorough" uses real GPUs to measure engine performance (takes several hours).',
    )

    environment: Literal[
        "kubernetes", "virtual", "global-planner"
    ] = SLAPlannerDefaults.environment
    namespace: str = Field(
        default_factory=lambda: os.environ.get("DYN_NAMESPACE", "dynamo")
    )
    backend: Literal["vllm", "sglang", "trtllm", "mocker"] = SLAPlannerDefaults.backend
    mode: Literal["disagg", "prefill", "decode", "agg"] = SLAPlannerDefaults.mode

    no_operation: bool = SLAPlannerDefaults.no_operation
    log_dir: Optional[str] = SLAPlannerDefaults.log_dir
    throughput_adjustment_interval: int = (
        SLAPlannerDefaults.throughput_adjustment_interval
    )
    max_gpu_budget: int = SLAPlannerDefaults.max_gpu_budget
    min_endpoint: int = SLAPlannerDefaults.min_endpoint

    decode_engine_num_gpu: Optional[int] = None
    prefill_engine_num_gpu: Optional[int] = None

    profile_results_dir: str = SLAPlannerDefaults.profile_results_dir

    ttft: float = SLAPlannerDefaults.ttft
    itl: float = SLAPlannerDefaults.itl

    # Load predictor settings
    load_predictor: str = SLAPlannerDefaults.load_predictor
    load_predictor_log1p: bool = SLAPlannerDefaults.load_predictor_log1p
    prophet_window_size: int = SLAPlannerDefaults.prophet_window_size
    load_predictor_warmup_trace: Optional[str] = None

    # Kalman filter settings
    kalman_q_level: float = SLAPlannerDefaults.kalman_q_level
    kalman_q_trend: float = SLAPlannerDefaults.kalman_q_trend
    kalman_r: float = SLAPlannerDefaults.kalman_r
    kalman_min_points: int = SLAPlannerDefaults.kalman_min_points

    # Prometheus settings
    metric_pulling_prometheus_endpoint: str = Field(
        default_factory=lambda: os.environ.get(
            "PROMETHEUS_ENDPOINT",
            "http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090",
        )
    )
    metric_reporting_prometheus_port: int = Field(
        default_factory=lambda: int(os.environ.get("PLANNER_PROMETHEUS_PORT", 0))
    )

    no_correction: bool = SLAPlannerDefaults.no_correction
    model_name: Optional[str] = None

    # Global planner environment
    global_planner_namespace: Optional[str] = None

    # Scaling mode flags
    enable_throughput_scaling: bool = SLAPlannerDefaults.enable_throughput_scaling
    enable_load_scaling: bool = SLAPlannerDefaults.enable_load_scaling

    # Load-based scaling settings
    load_router_metrics_url: Optional[str] = SLAPlannerDefaults.load_router_metrics_url
    load_adjustment_interval: int = SLAPlannerDefaults.load_adjustment_interval
    load_learning_window: int = SLAPlannerDefaults.load_learning_window
    load_scaling_down_sensitivity: int = (
        SLAPlannerDefaults.load_scaling_down_sensitivity
    )
    load_metric_samples: int = SLAPlannerDefaults.load_metric_samples
    load_min_observations: int = SLAPlannerDefaults.load_min_observations

    @model_validator(mode="after")
    def _validate_config(self) -> "PlannerConfig":
        # global-planner environment requires a namespace
        if self.environment == "global-planner" and not self.global_planner_namespace:
            raise ValueError(
                "global_planner_namespace is required when environment='global-planner'. "
                "Please specify the namespace where GlobalPlanner is running."
            )

        # At least one scaling mode must be enabled
        if not self.enable_throughput_scaling and not self.enable_load_scaling:
            raise ValueError(
                "At least one scaling mode must be enabled "
                "(enable_throughput_scaling or enable_load_scaling)"
            )

        if self.enable_load_scaling:
            # Router metrics URL is required outside kubernetes mode
            if not self.load_router_metrics_url and self.environment != "kubernetes":
                raise ValueError(
                    "load_router_metrics_url is required when "
                    "load-based scaling is enabled outside kubernetes mode"
                )

            # Load-based interval must be shorter than throughput interval
            if self.enable_throughput_scaling:
                if self.load_adjustment_interval >= self.throughput_adjustment_interval:
                    raise ValueError(
                        f"load_adjustment_interval ({self.load_adjustment_interval}s) "
                        f"must be shorter than throughput_adjustment_interval ({self.throughput_adjustment_interval}s). "
                        "Load-based scaling is the fast reactive loop; throughput-based is the "
                        "slow predictive loop."
                    )

            # Auto-disable correction factor when load-based scaling is enabled
            if not self.no_correction:
                logger.warning(
                    "Correction factor is automatically disabled when load-based "
                    "scaling is enabled. Load-based scaling already accounts for "
                    "actual latency conditions."
                )
                self.no_correction = True

        return self

    @classmethod
    def from_config_arg(cls, config_arg: str) -> "PlannerConfig":
        """Create a PlannerConfig from a CLI --config argument.

        Auto-detects whether the argument is a file path (JSON/YAML) or an
        inline JSON string, loads it, and validates.
        """
        path = Path(config_arg)
        if path.is_file():
            return cls._load_from_file(path)

        # Try parsing as inline JSON
        try:
            data = json.loads(config_arg)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--config value is neither a valid file path nor valid JSON: {e}"
            ) from e

        return cls.model_validate(data)

    @classmethod
    def _load_from_file(cls, path: Path) -> "PlannerConfig":
        suffix = path.suffix.lower()
        text = path.read_text()

        if suffix in (".yaml", ".yml"):
            data = yaml.safe_load(text)
        elif suffix == ".json":
            data = json.loads(text)
        else:
            # Try JSON first, then YAML
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                try:
                    data = yaml.safe_load(text)
                except ImportError:
                    raise ValueError(
                        f"Could not parse config file '{path}'. "
                        "For YAML support, install pyyaml."
                    )

        return cls.model_validate(data)


if __name__ == "__main__":
    from pathlib import Path

    schema = PlannerConfig.model_json_schema()

    output_path = Path(__file__).parent / "planner_config_json_schema.json"
    output_path.write_text(json.dumps(schema, indent=2))
    print(f"PlannerConfig JSON schema written to {output_path}")
