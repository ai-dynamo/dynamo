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
from typing import Dict, Literal, Optional
from urllib.parse import parse_qsl

import yaml
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from dynamo.planner.config.aic_interpolation_spec import AICInterpolationSpec
from dynamo.planner.config.defaults import SLAPlannerDefaults
from dynamo.planner.config.parallelization import PickedParallelConfig
from dynamo.planner.plugins.registry.config import PluginRegistrationConfig
from dynamo.planner.plugins.types import HoldPolicy

logger = logging.getLogger(__name__)


def _prometheus_ssl_verify_default() -> bool:
    return os.environ.get("PROMETHEUS_SSL_VERIFY", "false").lower() in (
        "1",
        "true",
        "yes",
    )


class PlannerPreDeploymentSweepMode(str, Enum):
    None_ = "none"
    Rapid = "rapid"
    Thorough = "thorough"


class AICPerfModelSpec(BaseModel):
    """Native AIC model identity used by the Rust engine perf shim.

    Unlike ``AICInterpolationSpec``, this does not describe an AIC sweep.
    It is the forward-pass model/backend/parallelism identity used for
    real-time shim queries. Unsupported native AIC configs are allowed: the
    shim falls back to FPM regression and can still tune from observations.
    """

    hf_id: str = Field(description="HuggingFace model id, e.g. Qwen/Qwen3-32B")
    system: str = Field(description="AIC system identifier, e.g. h200_sxm")
    backend: Literal["trtllm", "vllm", "sglang"]
    backend_version: Optional[str] = None

    prefill_pick: Optional[PickedParallelConfig] = None
    decode_pick: Optional[PickedParallelConfig] = None

    model_arch: Optional[str] = None
    weight_dtype: Optional[str] = None
    moe_dtype: Optional[str] = None
    activation_dtype: Optional[str] = None
    kv_cache_dtype: Optional[str] = None


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

    model_config = ConfigDict(extra="forbid")

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
            "  • PREDICT: plugins run sequentially in priority-ASCENDING "
            "    order (smallest priority number runs first). Partial-merge "
            "    is first-writer-wins per prediction field — once a plugin "
            "    sets a field, later (larger-priority) plugins can only "
            "    fill the fields left as None. The smallest-priority "
            "    plugin is therefore the most authoritative: it writes "
            "    first and its values are immutable for the rest of the "
            "    chain. Only the smallest-priority plugin should set "
            "    ``final=True`` to terminate the chain. Setting "
            "    ``final=True`` on a non-smallest-priority plugin still "
            "    breaks the chain at that point (skipping larger-"
            "    priority-number fallback plugins) — which may be "
            "    intentional (cost / policy override) or a config "
            "    mistake; chain_augment cannot tell. The event is "
            "    recorded on ``ChainAugmentOutcome.chain_break_warnings`` "
            "    (surfaced via ``PipelineOutcome.audit_events``) for "
            "    operator audit; a Prometheus counter is deferred to a "
            "    follow-up observability PR."
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
        "``AuthValidator``. PR #1 only ships ``static_secret`` (shared "
        "secret) — populate it from a mounted ``Secret`` rather than "
        "hard-coding in the ConfigMap. K8s SA / SPIFFE JWT support "
        "lands in a follow-up PR.",
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
    requires_produced_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Hard dependency on earlier-stage produced fields. Each "
            "entry is a dot-path into ``PipelineContext`` (e.g. "
            '``"predictions"``, ``"observations.traffic"``). The '
            "scheduler skips this plugin for the current tick if any "
            "listed field is unset on the live context; skipped ticks "
            "do NOT advance the plugin's anchor, so the next tick that "
            "has the field still fires it.  Empty/unset = no gating."
        ),
    )
    observation_window_seconds: float = Field(
        default=0.0,
        ge=0,
        description=(
            "Aggregation window the plugin wants for windowed observation "
            "types in ``needs`` (currently ``observations.traffic``). "
            "0.0 = ``scale_interval`` freshness; ``N > 0`` = Prometheus "
            "aggregates over the last ``N`` seconds.  Enforced at "
            "register-time to be ``>= scale_interval_seconds`` — a "
            "smaller window than the pipeline tick rate is degenerate."
        ),
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
    See ``plugins/registry/README.md`` for the Register/Heartbeat
    protocol and ``plugins/registry/gateway.py`` for the server
    implementation.

    Default ``enabled=False`` keeps existing deployments unchanged.
    Operators opt in explicitly.
    """

    model_config = ConfigDict(extra="forbid")

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
            "Bind address, passed verbatim to gRPC's "
            "``add_insecure_port`` / ``add_secure_port``. Both accept "
            "gRPC's URI scheme: ``unix:/abs/path`` (or "
            "``unix:///abs/path``) for an in-Pod socket file — useful "
            "when plugins register from inside the same Pod and the "
            "Pod boundary is the trust boundary. ``host:port`` (e.g. "
            "``0.0.0.0:9099``) for TCP. mTLS for the cross-Pod TCP "
            "case lands in a follow-up PR; PR #1 callers either bind "
            "on an in-Pod ``unix:`` socket path (Pod-local trust) or "
            "pair TCP with K8s NetworkPolicy / Pod-to-Pod identity."
        ),
    )
    allow_insecure: bool = Field(
        default=False,
        description=(
            "Permit binding a plaintext (no-TLS) gRPC gateway on a TCP "
            "``host:port`` listen. Default False fails closed: a TCP "
            "listen with no server credentials is rejected, because the "
            "gateway receives plugins' shared-secret ``auth_token`` and a "
            "plaintext TCP bind would expose it on the wire. Mirrors the "
            "outbound ``transport.allow_insecure_grpc`` gate. ``unix:`` "
            "(Pod-local) listens are always allowed — the Pod boundary is "
            "the trust boundary. Set True only when TCP plaintext is "
            "acceptable (e.g. a trusted mesh / NetworkPolicy-isolated net)."
        ),
    )


class SchedulingConfig(BaseModel):
    """Planner-level scheduling config.

    Controls which tick engine drives the planner and how long each
    tick may run. Backwards compatible: all fields have safe defaults,
    so existing deployments see no behaviour change until
    ``use_orchestrator=True`` is set explicitly. Read by
    ``NativePlannerBase`` at startup.
    """

    model_config = ConfigDict(extra="forbid")

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
    scale_interval_seconds: float = Field(
        default=5.0,
        gt=0.0,
        description=(
            "Base pipeline cadence for the orchestrator path. Pipeline "
            "fires one tick per ``scale_interval_seconds`` regardless of "
            "individual plugin intervals; per-plugin throttling via "
            "``RegisterRequest.execution_interval_seconds`` then governs "
            "which plugins actually fire each tick. Must be <= every "
            "plugin's ``execution_interval_seconds`` and a divisor of "
            "every plugin's ``observation_window_seconds`` so windows "
            "align to tick boundaries. Ignored when "
            "``use_orchestrator=False`` (PSM path uses its legacy "
            "load_adjustment_interval_seconds / "
            "throughput_adjustment_interval_seconds two-cadence model). "
            "Surface added in PR #10124; full lazy-pull behaviour lands "
            "in the engine_adapter rewrite commit later in this PR."
        ),
    )


class PlannerConfig(BaseModel):
    """Pydantic configuration for the Dynamo Planner.

    Replaces the argparse-based CLI. All fields mirror the former CLI flags
    with defaults sourced from SLAPlannerDefaults.
    """

    pre_deployment_sweeping_mode: Optional[PlannerPreDeploymentSweepMode] = Field(
        default=PlannerPreDeploymentSweepMode.Rapid,
        description=(
            "Controls optional pre-deployment perf-model bootstrap data. "
            '"none" skips bootstrap data, "rapid" uses AI Configurator to '
            'simulate engine performance, and "thorough" uses real GPUs to '
            "measure engine performance (takes several hours)."
        ),
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
    optimization_target: Literal["throughput", "latency", "load", "sla"] = Field(
        default="throughput",
        description=(
            "Scaling optimization target. "
            "'throughput' (default) and 'latency' use static thresholds on queue "
            "depth and KV cache utilization — no SLA targets or profiling needed. "
            "'load' uses user-defined prefill queue token and decode KV "
            "utilization thresholds. "
            "'sla' uses the Rust engine perf model to target specific "
            "ttft_ms/itl_ms values."
        ),
    )

    log_dir: Optional[str] = SLAPlannerDefaults.log_dir
    throughput_adjustment_interval_seconds: int = Field(
        default=SLAPlannerDefaults.throughput_adjustment_interval_seconds,
        validation_alias=AliasChoices(
            "throughput_adjustment_interval_seconds",
            "throughput_adjustment_interval",
        ),
    )
    max_gpu_budget: int = SLAPlannerDefaults.max_gpu_budget
    min_gpu_budget: int = SLAPlannerDefaults.min_gpu_budget
    """Per-DGD GPU floor enforced by the local planner. -1 disables (default).

    When set alongside ``max_gpu_budget`` with ``min == max``, the local
    planner pins the per-DGD total and only redistributes replicas between
    prefill and decode. Tolerance band:
    ``[min_gpu_budget - tolerance, max_gpu_budget + tolerance]`` where
    ``tolerance = max(prefill_engine_num_gpu, decode_engine_num_gpu)`` —
    needed because integer worker steps from pools with different per-replica
    GPU counts can't always exactly cancel.

    This is per-DGD scope. The GlobalPlanner has a separate cluster-wide
    ``min_total_gpus`` flag for cross-DGD enforcement; the two are
    orthogonal and can both be set.
    """
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
    aic_perf_model: Optional[AICPerfModelSpec] = Field(
        default=None,
        description=(
            "Native AIC forward-pass perf model identity for the Rust engine "
            "perf shim. This enables real-time AIC estimates plus online "
            "correction; unsupported native configs automatically fall back to "
            "FPM regression in the shim. This field does not trigger AIC "
            "interpolation sweeps."
        ),
    )

    ttft_ms: float = Field(
        default=SLAPlannerDefaults.ttft_ms,
        validation_alias=AliasChoices("ttft_ms", "ttft"),
    )
    itl_ms: float = Field(
        default=SLAPlannerDefaults.itl_ms,
        validation_alias=AliasChoices("itl_ms", "itl"),
    )

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
    metric_pulling_prometheus_token: Optional[str] = Field(
        default_factory=lambda: os.environ.get("PROMETHEUS_TOKEN"),
        exclude=True,
        description=(
            "Optional bearer token sent as `Authorization: Bearer <token>` on "
            "every PromQL request. Useful for hardened monitoring stacks "
            "(OpenShift thanos-querier, OAuth-proxied Prometheus). Token is "
            "read once at startup."
        ),
    )
    metric_pulling_prometheus_token_file: Optional[str] = Field(
        default_factory=lambda: os.environ.get("PROMETHEUS_TOKEN_FILE"),
        exclude=True,
        description=(
            "Optional path to a file containing a bearer token. When set, "
            "the token is re-read before every PromQL request so rotated "
            "tokens (Kubernetes projected ServiceAccount tokens, OpenShift "
            "OAuth SA tokens) are picked up without restarting the planner."
        ),
    )
    metric_pulling_prometheus_ssl_verify: bool = Field(
        default_factory=_prometheus_ssl_verify_default,
        exclude=True,
        description=(
            "Verify the upstream Prometheus TLS certificate. Default False "
            "preserves the previous PrometheusConnect(disable_ssl=True) "
            "behavior. Set True for hardened monitoring stacks; pair with "
            "an injected CA bundle if the upstream uses a private CA."
        ),
    )
    metric_pulling_prometheus_extra_query_params: Optional[Dict[str, str]] = Field(
        default_factory=lambda: (
            dict(
                parse_qsl(
                    os.environ.get("PROMETHEUS_EXTRA_QUERY_PARAMS", ""),
                    strict_parsing=True,
                )
            )
            or None
        ),
        exclude=True,
        description=(
            "Fixed key/value pairs appended as URL query parameters on every PromQL "
            "request. Set via PROMETHEUS_EXTRA_QUERY_PARAMS as a URL query string, "
            "e.g. `namespace=my-ns&tenant=foo`."
        ),
    )
    metric_pulling_prometheus_ca_bundle: Optional[str] = Field(
        default_factory=lambda: os.environ.get("PROMETHEUS_CA_BUNDLE"),
        exclude=True,
        validate_default=True,
        description=(
            "Path to a CA bundle for verifying the upstream Prometheus TLS certificate. "
            "No-op unless ssl_verify is enabled."
        ),
    )

    @field_validator("metric_pulling_prometheus_ca_bundle", mode="after")
    @classmethod
    def _validate_ca_bundle_path(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not Path(v).is_file():
            raise ValueError(
                f"metric_pulling_prometheus_ca_bundle path does not exist or is not a file: {v!r}. "
                "Check that PROMETHEUS_CA_BUNDLE points to a valid CA bundle file."
            )
        return v

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
    load_adjustment_interval_seconds: int = Field(
        default=SLAPlannerDefaults.load_adjustment_interval_seconds,
        validation_alias=AliasChoices(
            "load_adjustment_interval_seconds", "load_adjustment_interval"
        ),
        description=(
            "Interval for perf-model tuning AND load-based "
            "scaling decisions. Even when only throughput-based scaling is enabled, "
            "live FPM observations are fed into the perf model at this interval to "
            "keep the performance model accurate. Must be shorter than "
            "throughput_adjustment_interval_seconds."
        ),
    )
    max_num_fpm_samples: int = SLAPlannerDefaults.max_num_fpm_samples
    fpm_sample_bucket_size: int = SLAPlannerDefaults.fpm_sample_bucket_size
    load_scaling_down_sensitivity: int = (
        SLAPlannerDefaults.load_scaling_down_sensitivity
    )
    prefill_scale_up_queue_tokens: Optional[int] = Field(
        default=SLAPlannerDefaults.prefill_scale_up_queue_tokens,
        ge=0,
        description=(
            "Prefill queue token count that triggers scale-up when "
            "optimization_target='load'."
        ),
    )
    prefill_scale_down_queue_tokens: Optional[int] = Field(
        default=SLAPlannerDefaults.prefill_scale_down_queue_tokens,
        ge=0,
        description=(
            "Prefill queue token count that allows scale-down when "
            "optimization_target='load'."
        ),
    )
    decode_scale_up_kv_rate: Optional[float] = Field(
        default=SLAPlannerDefaults.decode_scale_up_kv_rate,
        ge=0,
        le=100,
        validation_alias=AliasChoices(
            "decode_scale_up_kv_rate", "decode_sacle_up_kv_rate"
        ),
        description=(
            "Decode KV utilization percentage that triggers scale-up when "
            "optimization_target='load'. Accepts 0-100."
        ),
    )
    decode_scale_down_kv_rate: Optional[float] = Field(
        default=SLAPlannerDefaults.decode_scale_down_kv_rate,
        ge=0,
        le=100,
        description=(
            "Decode KV utilization percentage that allows scale-down when "
            "optimization_target='load'. Accepts 0-100."
        ),
    )
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
    report_write_gzip_log: bool = Field(
        default=True,
        description=(
            "Write a compressed JSONL diagnostics log next to each HTML report. "
            "The gzip sidecar uses the same report basename with .log.jsonl.gz."
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

    plugin_registration: PluginRegistrationConfig = Field(
        default_factory=PluginRegistrationConfig,
        description=(
            "Plugin registry config — auth validators, transport, "
            "heartbeat, in-process plugins, admin RBAC. Default leaves "
            "``auth.trusted_sources`` empty, which falls back to "
            "``AllowUnauthenticatedAuth`` in the orchestrator (DEV ONLY — "
            "logs WARN on startup). Production: set "
            "``auth.trusted_sources=['static_secret']`` and populate "
            "``auth.static_secrets`` from a mounted ``Secret``. "
            "K8s SA / SPIFFE JWT support lands in a follow-up PR."
        ),
    )

    @model_validator(mode="after")
    def _validate_config(self) -> "PlannerConfig":
        if self.ttft_ms <= 0:
            raise ValueError(f"ttft_ms must be > 0, got {self.ttft_ms}")

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

        if self.optimization_target == "load":
            if self.mode in ("disagg", "prefill"):
                if (
                    self.prefill_scale_up_queue_tokens is None
                    or self.prefill_scale_down_queue_tokens is None
                ):
                    raise ValueError(
                        "optimization_target='load' requires "
                        "prefill_scale_up_queue_tokens and "
                        "prefill_scale_down_queue_tokens for prefill scaling"
                    )
                if (
                    self.prefill_scale_up_queue_tokens
                    <= self.prefill_scale_down_queue_tokens
                ):
                    raise ValueError(
                        "prefill_scale_up_queue_tokens must be greater than "
                        "prefill_scale_down_queue_tokens"
                    )
            if self.mode in ("disagg", "decode", "agg"):
                if (
                    self.decode_scale_up_kv_rate is None
                    or self.decode_scale_down_kv_rate is None
                ):
                    raise ValueError(
                        "optimization_target='load' requires "
                        "decode_scale_up_kv_rate and decode_scale_down_kv_rate "
                        "for decode scaling"
                    )
                if self.decode_scale_up_kv_rate <= self.decode_scale_down_kv_rate:
                    raise ValueError(
                        "decode_scale_up_kv_rate must be greater than "
                        "decode_scale_down_kv_rate"
                    )

        # Easy mode: force load scaling on, throughput scaling off
        if self.optimization_target != "sla":
            if self.optimization_target == "load" and self.enable_throughput_scaling:
                logger.warning(
                    "optimization_target='load' disables throughput-based scaling; "
                    "using reactive load-based scaling only"
                )
            self.enable_load_scaling = True
            self.enable_throughput_scaling = False
            if (
                self.ttft_ms != SLAPlannerDefaults.ttft_ms
                or self.itl_ms != SLAPlannerDefaults.itl_ms
            ):
                logger.warning(
                    "optimization_target=%s ignores ttft_ms/itl_ms values; "
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
                logger.warning(
                    "pre_deployment_sweeping_mode is 'none' or unset while "
                    "throughput scaling is enabled; the Rust engine perf model "
                    "will start from native AIC estimates when available or "
                    "from live FPM regression after enough observations."
                )
            if (
                self.pre_deployment_sweeping_mode == PlannerPreDeploymentSweepMode.Rapid
                and self.aic_interpolation is None
            ):
                logger.warning(
                    "pre_deployment_sweeping_mode='rapid' but aic_interpolation "
                    "is not set; planner will use aic_perf_model, live FPM "
                    "tuning, or profile_results_dir files if the "
                    "get_perf_metrics endpoint is unavailable."
                )

        if self.aic_perf_model is not None:
            if (
                self.mode in ("disagg", "prefill")
                and self.aic_perf_model.prefill_pick is None
            ):
                raise ValueError(
                    "aic_perf_model.prefill_pick is required for prefill "
                    f"perf queries in mode={self.mode!r}"
                )
            if (
                self.mode in ("disagg", "decode", "agg")
                and self.aic_perf_model.decode_pick is None
            ):
                raise ValueError(
                    "aic_perf_model.decode_pick is required for decode/agg "
                    f"perf queries in mode={self.mode!r}"
                )

        if self.enable_load_scaling:
            if self.enable_throughput_scaling:
                if (
                    self.load_adjustment_interval_seconds
                    >= self.throughput_adjustment_interval_seconds
                ):
                    raise ValueError(
                        f"load_adjustment_interval_seconds ({self.load_adjustment_interval_seconds}s) "
                        f"must be shorter than throughput_adjustment_interval_seconds ({self.throughput_adjustment_interval_seconds}s). "
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
