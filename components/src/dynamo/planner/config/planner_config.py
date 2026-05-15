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
import math
import os
from enum import Enum
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.defaults import SLAPlannerDefaults
from dynamo.planner.plugins.types import HoldPolicy

logger = logging.getLogger(__name__)


class PlannerPreDeploymentSweepMode(str, Enum):
    None_ = "none"
    Rapid = "rapid"
    Thorough = "thorough"


class ExternalPluginEntry(BaseModel):
    """One entry in the static external-plugin registration list.

    The planner reads this list at startup and calls
    ``await registry.register(RegisterRequest(...))`` for each entry —
    same code path an external plugin would hit through the gRPC
    gateway, so behaviour is identical to dynamic registration.

    Sourced from PlannerConfig (which itself comes from a ConfigMap in
    K8s). The plugin process must already be running and reachable at
    ``endpoint`` when the planner starts; if it isn't, the entry's
    register fails and is logged but the planner keeps booting (a bad
    plugin entry must NOT take down the planner).
    """

    plugin_id: str = Field(
        ...,
        min_length=1,
        description="Unique identifier; must not collide with builtin "
        "plugin_ids (e.g. ``builtin_load_propose``).",
    )
    plugin_type: Literal["predict", "propose", "reconcile", "constrain"] = Field(
        ...,
        description="Stage this plugin participates in.",
    )
    priority: int = Field(
        ...,
        description=(
            "Stage priority — smaller number = more authoritative in this "
            "stage. The number's *meaning* is uniform but the *mechanism* "
            "by which it takes effect differs between the merge stages "
            "(parallel) and PREDICT (sequential chain):\n"
            "  • PROPOSE / RECONCILE / CONSTRAIN: plugins run in parallel; "
            "    smallest-priority SET wins on conflict (type-aware merge). "
            "    AT_LEAST / AT_MOST clamps stack regardless of priority — "
            "    AT_LEAST = max of floors, AT_MOST = min of ceilings.\n"
            "  • PREDICT: plugins run sequentially in priority-DESCENDING "
            "    order (largest priority first); smallest-priority plugin "
            "    runs LAST and gets the final word on each prediction "
            "    field via partial-merge. Only the smallest-priority "
            "    plugin may set ``final=True`` to terminate the chain — "
            "    otherwise chain_augment emits a misuse warning + "
            "    Prometheus counter and the rest of the chain is skipped."
        ),
    )
    endpoint: str = Field(
        ...,
        min_length=1,
        description=(
            "Wire endpoint where the plugin is reachable. Must start with "
            "``grpc://host:port`` (TCP). ``inproc://`` is rejected by "
            "``register()`` since static-config plugins are out-of-process "
            "by definition."
        ),
    )
    auth_token: str = Field(
        default="",
        description="Bearer token validated by the registry's "
        "``AuthValidator``. For ``static_secret`` source, this is the "
        "shared secret. For ``k8s_sa`` / ``spiffe_jwt``, this is the "
        "projected SA token / SPIFFE JWT-SVID. Production deployments "
        "should template this from a mounted Secret rather than "
        "hard-coding in the ConfigMap.",
    )
    protocol_version: str = Field(
        default="1.0",
        description="Plugin protocol version. Must match planner's "
        "supported range (``[1.0, 1.0]`` today).",
    )
    version: str = Field(
        default="v1",
        description="Plugin's own version string — surfaced in "
        "ListPlugins for debugging / canary identification.",
    )
    execution_interval_seconds: float = Field(
        default=0.0,
        ge=0,
        description="0.0 means ``run every tick``; positive value "
        "throttles to ``every N seconds`` (PluginScheduler enforces).",
    )
    hold_policy: HoldPolicy = Field(
        default=HoldPolicy.HOLD_LAST,
        description="What to do when this plugin is throttled by "
        "execution_interval. ``HOLD_LAST`` reuses the cached result "
        "(typical for static-config plugins); ``ACCEPT_WHEN_IDLE`` "
        "treats it as no-opinion when not due.",
    )
    needs: list[str] = Field(
        default_factory=list,
        description="Capability list (consumed by type-aware merge); "
        "empty in v1 (no plugin yet uses needs declaration).",
    )

    @field_validator("hold_policy", mode="before")
    @classmethod
    def _coerce_hold_policy(cls, v):
        # IntEnum doesn't auto-accept string names from JSON/YAML
        # config (Pydantic just sees ``"HOLD_LAST"`` and tries the int
        # path). ConfigMap authors think in names, so accept either:
        # ``"HOLD_LAST"`` / ``"ACCEPT_WHEN_IDLE"`` (case-insensitive)
        # OR the raw integer the IntEnum already accepts.
        if isinstance(v, str):
            try:
                return HoldPolicy[v.upper()]
            except KeyError:
                raise ValueError(
                    f"hold_policy must be one of {[p.name for p in HoldPolicy]}, "
                    f"got {v!r}"
                )
        return v


class GatewayConfig(BaseModel):
    """Plugin-registry gRPC gateway config.

    When ``enabled=True``, the planner stands up a gRPC server hosting
    the public ``PluginRegistry`` service so external plugin processes
    can register / heartbeat / unregister themselves over the network.
    See `docs/components/planner/external-plugins.md`.

    Default ``enabled=False`` keeps existing deployments unchanged.
    Operators opt in explicitly.
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Open the gRPC gateway at ``listen``. Required for "
            "self-registering plugins. Static-config plugins"
            "registered via ``external_plugins`` do NOT need this."
        ),
    )
    listen: str = Field(
        default="unix:///var/run/dynamo/planner/registry.sock",
        description=(
            "Bind address. ``unix:/path`` for same-Pod sidecar plugins "
            "(recommended; Pod boundary is the trust boundary). The "
            "default matches ``EndpointsConfig.uds_socket_path`` in "
            "``plugins/registry/config.py`` — keep these aligned. "
            "``host:port`` for cross-Pod (e.g. ``0.0.0.0:9099``) — "
            "production cross-Pod deployments should pair with mTLS, "
            "but the gateway itself currently only takes the bind "
            "address; mTLS credentials are wired by the planner "
            "startup helper."
        ),
    )


class SchedulingConfig(BaseModel):
    """Planner-level scheduling config.

    Controls which tick engine drives the planner and how long each
    tick may run. Backwards compatible: all fields have safe defaults,
    so existing deployments see no behaviour change until
    ``use_orchestrator=True`` is set explicitly.

    Distinct from the per-stage ``SchedulingConfig`` in
    ``plugins/registry/config.py`` (registry internals); only this one
    lives on ``PlannerConfig`` and is read by ``NativePlannerBase``.
    """

    use_orchestrator: bool = Field(
        default=False,
        description=(
            "Feature flag: when True, the planner drives ticks through "
            "``LocalPlannerOrchestrator`` + real builtin plugins; when "
            "False (default), uses the legacy ``PlannerStateMachine`` "
            "path. Both paths are wired in ``NativePlannerBase`` via "
            "``EngineProtocol``. Defaulted OFF so upgrade ≠ cutover — "
            "operations control the enable timing."
        ),
    )
    request_timeout_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Per-plugin RPC timeout (orchestrator path only).",
    )
    allow_insecure_grpc: bool = Field(
        default=False,
        description=(
            "When True, the orchestrator's transport factory accepts "
            "``grpc://`` endpoints WITHOUT mTLS. Default False rejects "
            "them at register time so production deployments can't "
            "accidentally talk to a plugin in cleartext. Flip to True "
            "ONLY in dev / staging clusters where you're using "
            "loopback or a hardware-isolated network. Logs a WARNING "
            "on every transport build when enabled. UDS (``unix://``) "
            "and inproc endpoints are unaffected."
        ),
    )
    tick_max_duration_seconds: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Outermost deadline wrapping the entire 4-stage pipeline "
            "(orchestrator path only)."
        ),
    )
    external_plugins: list[ExternalPluginEntry] = Field(
        default_factory=list,
        description=(
            "Static external plugin registration list. Each entry "
            "is registered at planner startup via the same code path "
            "the gRPC gateway would use — so behaviour is "
            "identical between static-config and self-register models. "
            "Per-entry register failures are logged but do not crash "
            "the planner. Only used when ``use_orchestrator=True``; "
            "ignored on the legacy PSM path."
        ),
    )
    gateway: GatewayConfig = Field(
        default_factory=GatewayConfig,
        description=(
            "gRPC registration gateway config. Default disabled. "
            "Only used when ``use_orchestrator=True``."
        ),
    )


class PlannerConfig(BaseModel):
    """Pydantic configuration for the Dynamo Planner.

    Replaces the argparse-based CLI. All fields mirror the former CLI flags
    with defaults sourced from SLAPlannerDefaults.
    """

    pre_deployment_sweeping_mode: Optional[PlannerPreDeploymentSweepMode] = Field(
        default=PlannerPreDeploymentSweepMode.Rapid,
        description='Controls pre-deployment sweeping mode for planner in-depth profiling. "none" means no pre-deployment sweep (only load-based scaling). "rapid" uses AI Configurator to simulate engine performance. "thorough" uses real GPUs to measure engine performance (takes several hours).',
    )

    environment: Literal[
        "kubernetes", "virtual", "global-planner"
    ] = SLAPlannerDefaults.environment
    namespace: str = Field(
        default_factory=lambda: os.environ.get("DYN_NAMESPACE", "dynamo"),
        exclude=True,
    )
    backend: Literal["vllm", "sglang", "trtllm", "mocker"] = SLAPlannerDefaults.backend
    mode: Literal["disagg", "prefill", "decode", "agg"] = SLAPlannerDefaults.mode
    optimization_target: Literal["throughput", "latency", "sla"] = Field(
        default="throughput",
        description=(
            "Scaling optimization target. "
            "'throughput' (default) and 'latency' use static thresholds on queue "
            "depth and KV cache utilization — no SLA targets or profiling needed. "
            "'sla' uses regression-based scaling that targets specific ttft/itl values."
        ),
    )

    log_dir: Optional[str] = SLAPlannerDefaults.log_dir
    throughput_adjustment_interval: int = (
        SLAPlannerDefaults.throughput_adjustment_interval
    )
    max_gpu_budget: int = SLAPlannerDefaults.max_gpu_budget
    min_endpoint: int = SLAPlannerDefaults.min_endpoint

    decode_engine_num_gpu: Optional[int] = None
    prefill_engine_num_gpu: Optional[int] = None

    profile_results_dir: str = SLAPlannerDefaults.profile_results_dir

    aic_interpolation: Optional[AICInterpolationSpec] = Field(
        default=None,
        description=(
            "AIConfigurator interpolation spec populated by the profiler in "
            "rapid mode. When set, the planner runs the AIC sweep in-process "
            "at bootstrap and uses the resulting FPMs to seed the regression "
            "models (priority 2 between the get_perf_metrics endpoint and "
            "the legacy profile_results_dir file loader)."
        ),
    )

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
        ),
        exclude=True,
    )
    metric_reporting_prometheus_port: int = Field(
        default_factory=lambda: int(os.environ.get("PLANNER_PROMETHEUS_PORT", 0))
    )
    throughput_metrics_source: Literal[
        "frontend", "router"
    ] = SLAPlannerDefaults.throughput_metrics_source

    model_name: Optional[str] = None

    # Global planner environment
    global_planner_namespace: Optional[str] = None

    # Scaling mode flags
    enable_throughput_scaling: bool = SLAPlannerDefaults.enable_throughput_scaling
    enable_load_scaling: bool = SLAPlannerDefaults.enable_load_scaling

    # Load-based scaling settings
    load_adjustment_interval: int = Field(
        default=SLAPlannerDefaults.load_adjustment_interval,
        description=(
            "Interval in seconds for FPM regression model updates AND load-based "
            "scaling decisions. Even when only throughput-based scaling is enabled, "
            "live FPM observations are fed into the regression at this interval to "
            "keep the performance model accurate. Must be shorter than "
            "throughput_adjustment_interval."
        ),
    )
    max_num_fpm_samples: int = SLAPlannerDefaults.max_num_fpm_samples
    fpm_sample_bucket_size: int = SLAPlannerDefaults.fpm_sample_bucket_size
    load_scaling_down_sensitivity: int = (
        SLAPlannerDefaults.load_scaling_down_sensitivity
    )
    load_metric_samples: int = SLAPlannerDefaults.load_metric_samples
    load_min_observations: int = SLAPlannerDefaults.load_min_observations

    # Advisory mode: compute and log decisions without executing scaling
    advisory: bool = SLAPlannerDefaults.advisory

    # Diagnostics report settings
    report_interval_hours: Optional[float] = Field(
        default=24.0,
        description=(
            "Generate an HTML diagnostics report every N hours (simulated time). "
            "Set to None to disable periodic report generation."
        ),
    )
    report_output_dir: str = Field(
        default="./planner_reports",
        description="Directory for HTML diagnostics reports.",
    )
    report_filename: Optional[str] = Field(
        default=None,
        description=(
            "Fixed filename for HTML diagnostics reports. "
            "When set, reports are written to report_output_dir/report_filename "
            "instead of the default timestamped name."
        ),
    )
    live_dashboard_port: int = Field(
        default=8080,
        description=(
            "Port for the live diagnostics dashboard HTTP server. "
            "Set to 0 to disable. When enabled, visit http://host:port/ "
            "to view a real-time Plotly report of accumulated snapshots."
        ),
    )

    scheduling: SchedulingConfig = Field(
        default_factory=SchedulingConfig,
        description=(
            "Tick-engine scheduling config — see ``SchedulingConfig`` "
            "docstring. Default uses the legacy PSM path; set "
            "``scheduling.use_orchestrator=true`` to opt into the "
            "orchestrator path."
        ),
    )

    @model_validator(mode="after")
    def _validate_config(self) -> "PlannerConfig":
        if self.ttft <= 0:
            raise ValueError(f"ttft must be > 0, got {self.ttft}")

        if self.report_interval_hours is not None:
            if (
                not math.isfinite(self.report_interval_hours)
                or self.report_interval_hours <= 0
            ):
                raise ValueError(
                    "report_interval_hours must be a positive finite number or None"
                )

        sqrt = math.isqrt(self.fpm_sample_bucket_size)
        if sqrt * sqrt != self.fpm_sample_bucket_size:
            raise ValueError(
                f"fpm_sample_bucket_size must be a perfect square, "
                f"got {self.fpm_sample_bucket_size}"
            )

        if self.environment == "global-planner" and not self.global_planner_namespace:
            raise ValueError(
                "global_planner_namespace is required when environment='global-planner'. "
                "Please specify the namespace where GlobalPlanner is running."
            )

        # Easy mode: force load scaling on, throughput scaling off
        if self.optimization_target != "sla":
            self.enable_load_scaling = True
            self.enable_throughput_scaling = False
            if (
                self.ttft != SLAPlannerDefaults.ttft
                or self.itl != SLAPlannerDefaults.itl
            ):
                logger.warning(
                    "optimization_target=%s ignores ttft/itl values; "
                    "set optimization_target='sla' to use SLA-based scaling",
                    self.optimization_target,
                )

        # At least one scaling mode must be enabled
        if not self.enable_throughput_scaling and not self.enable_load_scaling:
            raise ValueError(
                "At least one scaling mode must be enabled "
                "(enable_throughput_scaling or enable_load_scaling)"
            )

        if self.enable_throughput_scaling:
            if (
                self.pre_deployment_sweeping_mode is None
                or self.pre_deployment_sweeping_mode
                == PlannerPreDeploymentSweepMode.None_
            ):
                raise ValueError(
                    "pre_deployment_sweeping_mode cannot be 'none' when "
                    "enable_throughput_scaling is True. Throughput-based scaling "
                    "requires pre-deployment sweeping to profile engine performance."
                )
            if (
                self.pre_deployment_sweeping_mode == PlannerPreDeploymentSweepMode.Rapid
                and self.aic_interpolation is None
            ):
                logger.warning(
                    "pre_deployment_sweeping_mode='rapid' but aic_interpolation "
                    "is not set; planner will fall back to profile_results_dir "
                    "files if the get_perf_metrics endpoint is unavailable."
                )

        if self.enable_load_scaling:
            if self.enable_throughput_scaling:
                if self.load_adjustment_interval >= self.throughput_adjustment_interval:
                    raise ValueError(
                        f"load_adjustment_interval ({self.load_adjustment_interval}s) "
                        f"must be shorter than throughput_adjustment_interval ({self.throughput_adjustment_interval}s). "
                        "Load-based scaling is the fast reactive loop; throughput-based is the "
                        "slow predictive loop."
                    )

        return self

    @classmethod
    def from_config_arg(cls, config_arg: str) -> "PlannerConfig":
        """Create a PlannerConfig from a CLI --config argument.

        Auto-detects whether the argument is a file path (JSON/YAML) or an
        inline JSON string, loads it, and validates.
        """
        path = Path(config_arg)
        try:
            is_file = path.is_file()
        except OSError:
            # Path component too long (e.g. inline JSON string passed as config arg)
            is_file = False
        if is_file:
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

    def scaling_enabled(self) -> bool:
        return self.enable_throughput_scaling or self.enable_load_scaling


if __name__ == "__main__":
    from pathlib import Path

    schema = PlannerConfig.model_json_schema()

    output_path = Path(__file__).parent / "planner_config_json_schema.json"
    output_path.write_text(json.dumps(schema, indent=2))
    print(f"PlannerConfig JSON schema written to {output_path}")
