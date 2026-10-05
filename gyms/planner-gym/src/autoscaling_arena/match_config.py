# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, backend-aware YAML configuration for Autoscaling Arena matches.

This module deliberately has no Dynamo imports.  Match files can therefore be
validated, expanded, and used by the real/AIPerf backend in an environment that
does not contain the simulation runtime.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import yaml
from autoscaling_arena.substrates import SUBSTRATES
from autoscaling_arena.workloads import WORKLOADS, validate_mooncake_trace

SCHEMA_VERSION = 1

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

SIM_METRICS = frozenset(
    {
        "goodput_per_gpu",
        "goodput_rps",
        "good_rate",
        "good_count",
        "gpu_hours",
        "request_throughput_rps",
        "completed_requests",
        "duration_s",
        "mean_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_itl_ms",
        "p95_e2e_latency_ms",
        "oscillation_count",
        "scale_events",
    }
)
REAL_METRICS = frozenset(
    {
        "goodput_rps",
        "good_rate",
        "good_count",
        "request_throughput_rps",
        "completed_requests",
        "duration_s",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_itl_ms",
        "p99_itl_ms",
        "mean_e2e_ms",
        "p99_e2e_ms",
    }
)
_REAL_METRIC_ALIASES = {
    "good_request_fraction": "good_rate",
    "good_request_count": "good_count",
    "request_count": "completed_requests",
    "benchmark_duration_s": "duration_s",
}

DEFAULT_SIM_METRICS = (
    "goodput_per_gpu",
    "goodput_rps",
    "good_rate",
    "gpu_hours",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_itl_ms",
    "oscillation_count",
    "scale_events",
)
DEFAULT_REAL_METRICS = (
    "goodput_rps",
    "good_rate",
    "request_throughput_rps",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "mean_itl_ms",
    "p99_itl_ms",
)

_STATIC_CONFIG_KEYS = frozenset({"num_prefill", "num_decode", "poll_interval_s"})
_JEV_DEFAULTS = {
    "min_prefill": 1,
    "max_prefill": 16,
    "min_decode": 1,
    "max_decode": 8,
    "poll_interval_s": 15.0,
    "step": 1,
    "model": "jev-1.13.0",
    "timeout_s": 5.0,
    "min_confidence": 0.0,
    "history_ticks": 4,
    "max_calls": 500,
    "failure_mode": "raise",
}
_REACTIVE_CONFIG_KEYS = frozenset(
    {
        "min_prefill",
        "max_prefill",
        "min_decode",
        "max_decode",
        "prefill_queue_up",
        "prefill_queue_down",
        "decode_kv_up",
        "decode_kv_down",
        "agg_queue_up",
        "agg_queue_down",
        "poll_interval_s",
        "step",
    }
)
_KEDA_CONFIG_KEYS = frozenset(
    {
        "poll_interval_s",
        "queue_threshold",
        "kv_threshold",
        "min_prefill",
        "max_prefill",
        "min_decode",
        "max_decode",
        "tolerance",
        "scale_up_stabilization_s",
        "scale_down_stabilization_s",
    }
)
_ENGINE_LAYER_KEYS = frozenset(
    {
        "system",
        "backend",
        "backend_version",
        "ais_backend",
        "ais_backend_version",
        "tp_size",
        "moe_tp_size",
        "moe_ep_size",
        "attention_dp_size",
        "runtime",
        "extra_args",
    }
)
_ENGINE_RUNTIME_KEYS = frozenset(
    {
        "cold_start_delay_s",
        "kv_transfer_bandwidth_gbps",
        "kv_bytes_per_token",
    }
)
_ENGINE_BACKENDS = frozenset({"vllm", "sglang", "trtllm"})
_RESERVED_ENGINE_ARGS = frozenset(
    {
        "ais_backend",
        "ais_system",
        "ais_model_path",
        "ais_tp_size",
        "ais_backend_version",
        "ais_moe_tp_size",
        "ais_moe_ep_size",
        "ais_attention_dp_size",
        "engine_type",
        "ais_perf_config",
        "tensor_parallel_size",
        "dp_size",
        "worker_type",
        "is_prefill",
        "is_decode",
        "startup_time",
        "kv_transfer_bandwidth",
        "kv_bytes_per_token",
    }
)

_STATIC_DEFAULTS = {
    "poll_interval_s": 5.0,
}
_REACTIVE_DEFAULTS = {
    "min_prefill": 1,
    "max_prefill": 16,
    "min_decode": 1,
    "max_decode": 8,
    "prefill_queue_up": 4,
    "prefill_queue_down": 1,
    "decode_kv_up": 0.8,
    "decode_kv_down": 0.3,
    "agg_queue_up": 4,
    "agg_queue_down": 1,
    "poll_interval_s": 15.0,
    "step": 1,
}
_KEDA_DEFAULTS = {
    "poll_interval_s": 15.0,
    "queue_threshold": 5.0,
    "kv_threshold": 0.9,
    "min_prefill": 1,
    "max_prefill": 16,
    "min_decode": 1,
    "max_decode": 8,
    "tolerance": 0.10,
    "scale_up_stabilization_s": 0.0,
    "scale_down_stabilization_s": 300.0,
}


class MatchConfigError(ValueError):
    """A user-facing Match Config validation error."""


@dataclass(frozen=True)
class ReplicaCounts:
    prefill: int
    decode: int


@dataclass(frozen=True)
class ModelConfig:
    """The model requested by replay and used for AIS performance lookup."""

    name: str
    ais_model_path: str


@dataclass(frozen=True)
class EngineRuntimeConfig:
    """Human-facing engine lifecycle and disaggregated-transfer behavior."""

    cold_start_delay_s: Optional[float] = None
    kv_transfer_bandwidth_gbps: Optional[float] = None
    kv_bytes_per_token: Optional[int] = None


@dataclass(frozen=True)
class EngineConfig:
    """One fully-resolved simulation engine role."""

    system: str
    backend: str
    backend_version: Optional[str]
    ais_backend: str
    ais_backend_version: Optional[str]
    tp_size: int
    moe_tp_size: Optional[int]
    moe_ep_size: Optional[int]
    attention_dp_size: Optional[int]
    num_gpus: int
    extra_args: dict[str, Any]
    runtime: EngineRuntimeConfig = field(default_factory=EngineRuntimeConfig)


@dataclass(frozen=True)
class SimEnginesConfig:
    """Topology-specific engines after common fields have been resolved."""

    prefill: Optional[EngineConfig] = None
    decode: Optional[EngineConfig] = None
    aggregate: Optional[EngineConfig] = None


@dataclass(frozen=True)
class SimAutoscalerConfig:
    name: str
    type: str
    start: ReplicaCounts
    config: dict[str, Any]


@dataclass(frozen=True)
class RealAutoscalerConfig:
    name: str
    endpoint: str
    autoscaler_type: Optional[str]
    declared_config: dict[str, Any]


@dataclass(frozen=True)
class EndpointConfig:
    name: str
    url: str
    model: str
    description: str = ""
    endpoint_type: str = "chat"
    declared_deployment: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RouterConfig:
    mode: str = "round_robin"


@dataclass(frozen=True)
class ReplayConfig:
    ais_bootstrap: bool = True
    concurrency: Optional[int] = None
    telemetry_sample_interval_s: Optional[float] = 5.0


@dataclass(frozen=True)
class AIPerfConfig:
    executable: str = "aiperf"
    tokenizer: Optional[str] = None
    streaming: bool = True
    timeout_s: Optional[float] = None
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class SimBackendConfig:
    type: str
    substrate: Optional[str]
    model: ModelConfig
    engines: SimEnginesConfig
    topology: str
    gpu_budget: int
    router: RouterConfig
    planner_config: dict[str, Any]
    replay: ReplayConfig
    autoscalers: tuple[SimAutoscalerConfig, ...]
    planner_config_path: Optional[Path] = None


@dataclass(frozen=True)
class RealBackendConfig:
    type: str
    endpoint_catalog: Optional[Path]
    endpoints: tuple[EndpointConfig, ...]
    aiperf: AIPerfConfig
    autoscalers: tuple[RealAutoscalerConfig, ...]


@dataclass(frozen=True)
class EvaluationConfig:
    workload: str
    seed: int
    max_requests: Optional[int]
    arrival_speedup: float
    trace_path: Optional[Path] = None
    trace_block_size: Optional[int] = None
    trace_presorted: bool = False
    trace_paths: tuple[Path, ...] = ()
    trace_format: str = "mooncake"


@dataclass(frozen=True)
class SLAProfileConfig:
    name: str
    source_name: str
    ttft_ms: Optional[float] = None
    itl_ms: Optional[float] = None
    e2e_ms: Optional[float] = None


@dataclass(frozen=True)
class MetricsConfig:
    rank_by: str
    include: tuple[str, ...]


@dataclass(frozen=True)
class ExecutionConfig:
    repetitions: int = 1
    fail_fast: bool = False
    max_runs: Optional[int] = None


@dataclass(frozen=True)
class PublishDestination:
    type: str
    path: Optional[Path] = None
    overwrite: bool = False


@dataclass(frozen=True)
class PublishConfig:
    artifact_root: Path
    destinations: tuple[PublishDestination, ...]


@dataclass(frozen=True)
class MatchRun:
    """One fully-expanded cell in a Match Config matrix."""

    index: int
    run_id: str
    backend: str
    autoscaler: str
    workload: str
    sla: str
    slo_profile: SLAProfileConfig
    repetition: int
    seed: int
    max_requests: Optional[int]
    arrival_speedup: float
    trace_path: Optional[Path] = None
    trace_block_size: Optional[int] = None
    trace_presorted: bool = False
    trace_paths: tuple[Path, ...] = ()
    trace_format: str = "mooncake"


@dataclass(frozen=True)
class MatchConfig:
    schema_version: int
    name: str
    description: str
    labels: dict[str, str]
    backend: SimBackendConfig | RealBackendConfig
    evaluations: tuple[EvaluationConfig, ...]
    sla_profiles: tuple[SLAProfileConfig, ...]
    metrics: MetricsConfig
    execution: ExecutionConfig
    publish: PublishConfig
    source_path: Path

    @property
    def expected_runs(self) -> int:
        return (
            len(self.backend.autoscalers)
            * len(self.evaluations)
            * len(self.sla_profiles)
            * self.execution.repetitions
        )

    def iter_runs(self) -> Iterator[MatchRun]:
        """Yield a deterministic expansion of the configured comparison matrix."""

        index = 0
        for evaluation in self.evaluations:
            for profile in self.sla_profiles:
                for repetition in range(self.execution.repetitions):
                    seed = evaluation.seed + repetition
                    for autoscaler in self.backend.autoscalers:
                        index += 1
                        identity = (
                            f"{index:04d}-{self.backend.type}-{autoscaler.name}-"
                            f"{evaluation.workload}-{profile.name}-r{repetition}-s{seed}"
                        )
                        yield MatchRun(
                            index=index,
                            run_id=identity,
                            backend=self.backend.type,
                            autoscaler=autoscaler.name,
                            workload=evaluation.workload,
                            sla=profile.name,
                            slo_profile=profile,
                            repetition=repetition,
                            seed=seed,
                            max_requests=evaluation.max_requests,
                            arrival_speedup=evaluation.arrival_speedup,
                            trace_path=evaluation.trace_path,
                            trace_paths=evaluation.trace_paths,
                            trace_format=evaluation.trace_format,
                            trace_block_size=evaluation.trace_block_size,
                            trace_presorted=evaluation.trace_presorted,
                        )

    def to_dict(self) -> dict[str, Any]:
        """Return the resolved, JSON-serializable configuration."""

        return _jsonable(asdict(self))


def load_match_config(path: str | Path) -> MatchConfig:
    """Load, strictly validate, and expand one YAML Match Config."""

    source_path = Path(path).expanduser().resolve()
    try:
        text = source_path.read_text()
    except OSError as exc:
        raise MatchConfigError(f"config: cannot read {source_path}: {exc}") from exc
    data = _load_yaml(text, source=str(source_path))
    return parse_match_config(data, source_path=source_path)


def parse_match_config(data: Any, *, source_path: str | Path) -> MatchConfig:
    """Validate an already-decoded Match Config mapping.

    ``source_path`` determines the base directory for all relative paths.
    """

    source = Path(source_path).expanduser().resolve()
    root = _mapping(data, "config")
    _only_keys(
        root,
        {
            "schema_version",
            "name",
            "description",
            "labels",
            "backend",
            "evaluations",
            "slo_profiles",
            "metrics",
            "execution",
            "publish",
        },
        "config",
    )

    version = _integer(_required(root, "schema_version", "config"), "schema_version")
    if version != SCHEMA_VERSION:
        raise MatchConfigError(
            f"schema_version: expected {SCHEMA_VERSION}, got {version}"
        )
    # The match's display name may be human-readable; all path-bearing child
    # identifiers are validated separately and run IDs never embed this value.
    name = _string(_required(root, "name", "config"), "name")
    description = _string(root.get("description", ""), "description", allow_empty=True)
    labels = _labels(root.get("labels", {}))

    backend_data = _mapping(_required(root, "backend", "config"), "backend")
    backend_type = _string(_required(backend_data, "type", "backend"), "backend.type")
    if backend_type == "sim":
        backend: SimBackendConfig | RealBackendConfig = _parse_sim_backend(
            backend_data, base_dir=source.parent
        )
    elif backend_type == "real":
        backend = _parse_real_backend(backend_data, base_dir=source.parent)
    else:
        raise MatchConfigError("backend.type: expected 'sim' or 'real'")

    evaluations = _parse_evaluations(
        _required(root, "evaluations", "config"), base_dir=source.parent
    )
    if backend_type == "real" and any(
        evaluation.trace_format == "dynamo" for evaluation in evaluations
    ):
        raise MatchConfigError(
            "evaluations.traces: exact Dynamo traces require backend.type 'sim'; "
            "the real backend drives AIPerf with Mooncake JSONL"
        )
    sla_profiles = _parse_sla_profiles(_required(root, "slo_profiles", "config"))
    metrics = _parse_metrics(root.get("metrics", {}), backend_type)
    execution = _parse_execution(root.get("execution", {}))
    publish = _parse_publish(
        _required(root, "publish", "config"), base_dir=source.parent
    )
    protected_inputs = {source}
    for evaluation in evaluations:
        protected_inputs.update(evaluation.trace_paths)
        if evaluation.trace_path is not None:
            protected_inputs.add(evaluation.trace_path)
    if isinstance(backend, RealBackendConfig):
        if backend.endpoint_catalog is not None:
            protected_inputs.add(backend.endpoint_catalog)
    elif backend.planner_config_path is not None:
        protected_inputs.add(backend.planner_config_path)
    if publish.artifact_root in protected_inputs:
        raise MatchConfigError(
            "publish.artifact_root: must not be an input configuration file"
        )
    for destination in publish.destinations:
        if destination.path in protected_inputs:
            raise MatchConfigError(
                "publish destination: must not overwrite an input configuration file"
            )
        if destination.path is not None and (
            publish.artifact_root == destination.path
            or destination.path in publish.artifact_root.parents
        ):
            raise MatchConfigError(
                "publish.artifact_root: must not equal or be nested beneath "
                f"file destination {destination.path}"
            )

    config = MatchConfig(
        schema_version=version,
        name=name,
        description=description,
        labels=labels,
        backend=backend,
        evaluations=evaluations,
        sla_profiles=sla_profiles,
        metrics=metrics,
        execution=execution,
        publish=publish,
        source_path=source,
    )
    if (
        config.execution.max_runs is not None
        and config.expected_runs > config.execution.max_runs
    ):
        raise MatchConfigError(
            "execution.max_runs: expanded matrix has "
            f"{config.expected_runs} runs, exceeding limit "
            f"{config.execution.max_runs}"
        )
    return config


def _parse_sim_backend(data: Mapping[str, Any], *, base_dir: Path) -> SimBackendConfig:
    _only_keys(
        data,
        {
            "type",
            "substrate",
            "model",
            "engines",
            "topology",
            "gpu_budget",
            "router",
            "planner_config",
            "replay",
            "autoscalers",
        },
        "backend",
    )
    topology = _string(data.get("topology", "disagg"), "backend.topology")
    if topology not in {"agg", "disagg"}:
        raise MatchConfigError("backend.topology: expected 'agg' or 'disagg'")

    has_preset = "substrate" in data
    has_explicit = "model" in data or "engines" in data
    if has_preset and has_explicit:
        raise MatchConfigError(
            "backend: choose either the substrate preset shorthand or explicit "
            "model and engines; do not combine them"
        )
    if not has_preset and not has_explicit:
        raise MatchConfigError(
            "backend: specify either substrate or first-class model and engines"
        )

    router_data = _mapping(data.get("router", {}), "backend.router")
    _only_keys(router_data, {"mode"}, "backend.router")
    router_mode = _string(router_data.get("mode", "round_robin"), "backend.router.mode")
    if router_mode not in {"round_robin", "kv_router"}:
        raise MatchConfigError(
            "backend.router.mode: expected 'round_robin' or 'kv_router'"
        )

    planner_config, planner_config_path = _parse_planner_config(
        data.get("planner_config", {}), base_dir=base_dir
    )
    _validate_json_value(planner_config, "backend.planner_config")
    reserved_planner_fields = {
        "advisory",
        "mode",
        "max_gpu_budget",
        "min_gpu_budget",
        "prefill_engine_num_gpu",
        "decode_engine_num_gpu",
        "report_output_dir",
        "report_filename",
    }
    overlap = sorted(reserved_planner_fields.intersection(planner_config))
    if overlap:
        raise MatchConfigError(
            "backend.planner_config: these runner-owned fields must use their "
            f"dedicated Match Config settings: {overlap}"
        )

    replay_data = _mapping(data.get("replay", {}), "backend.replay")
    _only_keys(
        replay_data,
        {
            "ais_bootstrap",
            "concurrency",
            "model_name",
            "telemetry_sample_interval_s",
        },
        "backend.replay",
    )
    ais_bootstrap = _boolean(
        replay_data.get("ais_bootstrap", True), "backend.replay.ais_bootstrap"
    )
    concurrency_raw = replay_data.get("concurrency")
    concurrency = (
        None
        if concurrency_raw is None
        else _integer(concurrency_raw, "backend.replay.concurrency", positive=True)
    )
    telemetry_interval_raw = replay_data.get("telemetry_sample_interval_s", 5.0)
    telemetry_sample_interval_s = (
        None
        if telemetry_interval_raw is None
        else _number(
            telemetry_interval_raw,
            "backend.replay.telemetry_sample_interval_s",
            positive=True,
        )
    )
    legacy_model_name_raw = replay_data.get("model_name")
    legacy_model_name = (
        None
        if legacy_model_name_raw is None
        else _string(legacy_model_name_raw, "backend.replay.model_name")
    )

    if has_preset:
        substrate = _string(data["substrate"], "backend.substrate")
        if substrate not in SUBSTRATES:
            raise MatchConfigError(
                f"backend.substrate: unknown substrate {substrate!r}; "
                f"known: {sorted(SUBSTRATES)}"
            )
        preset = SUBSTRATES[substrate]
        model = ModelConfig(
            name=legacy_model_name or preset.model,
            ais_model_path=preset.model,
        )
        preset_engine = _engine_from_substrate_preset(
            preset, path=f"backend.substrate[{substrate!r}]"
        )
        engines = (
            SimEnginesConfig(
                prefill=preset_engine,
                decode=preset_engine,
            )
            if topology == "disagg"
            else SimEnginesConfig(aggregate=preset_engine)
        )
        gpu_budget_default: Optional[int] = preset.default_gpu_budget
    else:
        substrate = None
        if legacy_model_name is not None:
            raise MatchConfigError(
                "backend.replay.model_name: cannot be combined with "
                "first-class backend.model; use backend.model.name"
            )
        if "model" not in data:
            raise MatchConfigError(
                "backend.model: field is required when substrate is omitted"
            )
        if "engines" not in data:
            raise MatchConfigError(
                "backend.engines: field is required when substrate is omitted"
            )
        model = _parse_model_config(data["model"])
        engines = _parse_engine_roles(data["engines"], topology=topology)
        gpu_budget_default = None

    if "gpu_budget" not in data and gpu_budget_default is None:
        raise MatchConfigError(
            "backend.gpu_budget: field is required for explicit model/engine configs"
        )
    gpu_budget = _integer(
        data.get("gpu_budget", gpu_budget_default),
        "backend.gpu_budget",
        positive=True,
    )

    if topology == "disagg":
        assert engines.prefill is not None and engines.decode is not None
        prefill_num_gpus = engines.prefill.num_gpus
        decode_num_gpus = engines.decode.num_gpus
    else:
        assert engines.aggregate is not None
        prefill_num_gpus = 0
        decode_num_gpus = engines.aggregate.num_gpus

    items = _list(_required(data, "autoscalers", "backend"), "backend.autoscalers")
    if not items:
        raise MatchConfigError("backend.autoscalers: must not be empty")
    autoscalers = tuple(
        _parse_sim_autoscaler(
            item,
            index=i,
            topology=topology,
            gpu_budget=gpu_budget,
            prefill_num_gpus=prefill_num_gpus,
            decode_num_gpus=decode_num_gpus,
        )
        for i, item in enumerate(items)
    )
    _unique_names([entry.name for entry in autoscalers], "backend.autoscalers")

    return SimBackendConfig(
        type="sim",
        substrate=substrate,
        model=model,
        engines=engines,
        topology=topology,
        gpu_budget=gpu_budget,
        router=RouterConfig(mode=router_mode),
        planner_config=planner_config,
        replay=ReplayConfig(
            ais_bootstrap=ais_bootstrap,
            concurrency=concurrency,
            telemetry_sample_interval_s=telemetry_sample_interval_s,
        ),
        autoscalers=autoscalers,
        planner_config_path=planner_config_path,
    )


def _parse_planner_config(
    value: Any, *, base_dir: Path
) -> tuple[dict[str, Any], Optional[Path]]:
    """Load an inline Planner mapping or a reusable YAML/JSON config path."""

    planner_config_path: Optional[Path] = None
    if isinstance(value, str):
        planner_config_path = _resolve_path(
            _string(value, "backend.planner_config"), base_dir
        )
        try:
            loaded = _load_yaml(
                planner_config_path.read_text(), source=str(planner_config_path)
            )
        except OSError as exc:
            raise MatchConfigError(
                "backend.planner_config: cannot read " f"{planner_config_path}: {exc}"
            ) from exc
        data = _mapping(loaded, "backend.planner_config")
    else:
        data = _mapping(value, "backend.planner_config")
    return dict(data), planner_config_path


def _parse_model_config(value: Any) -> ModelConfig:
    data = _mapping(value, "backend.model")
    _only_keys(data, {"name", "ais_model_path"}, "backend.model")
    name = _string(_required(data, "name", "backend.model"), "backend.model.name")
    ais_model_path = _string(
        data.get("ais_model_path", name), "backend.model.ais_model_path"
    )
    return ModelConfig(name=name, ais_model_path=ais_model_path)


def _parse_engine_roles(value: Any, *, topology: str) -> SimEnginesConfig:
    data = _mapping(value, "backend.engines")
    _only_keys(data, {"common", "prefill", "decode", "aggregate"}, "backend.engines")
    common = _parse_engine_layer(data.get("common", {}), "backend.engines.common")
    if topology == "disagg":
        if "aggregate" in data:
            raise MatchConfigError(
                "backend.engines.aggregate: omit this field for disagg topology"
            )
        prefill = _resolve_engine_config(
            _merge_engine_layers(
                common,
                _parse_engine_layer(data.get("prefill", {}), "backend.engines.prefill"),
            ),
            "backend.engines.prefill",
        )
        decode = _resolve_engine_config(
            _merge_engine_layers(
                common,
                _parse_engine_layer(data.get("decode", {}), "backend.engines.decode"),
            ),
            "backend.engines.decode",
        )
        return SimEnginesConfig(prefill=prefill, decode=decode)

    if "prefill" in data or "decode" in data:
        raise MatchConfigError(
            "backend.engines: prefill/decode overrides are only valid for "
            "disagg topology; use aggregate for agg"
        )
    aggregate = _resolve_engine_config(
        _merge_engine_layers(
            common,
            _parse_engine_layer(data.get("aggregate", {}), "backend.engines.aggregate"),
        ),
        "backend.engines.aggregate",
    )
    return SimEnginesConfig(aggregate=aggregate)


def _parse_engine_layer(value: Any, path: str) -> dict[str, Any]:
    data = dict(_mapping(value, path))
    _only_keys(data, _ENGINE_LAYER_KEYS, path)
    for key in ("system", "backend", "ais_backend"):
        if key in data:
            data[key] = _string(data[key], f"{path}.{key}")
    for key in ("backend", "ais_backend"):
        if key in data and data[key] not in _ENGINE_BACKENDS:
            raise MatchConfigError(
                f"{path}.{key}: expected one of {sorted(_ENGINE_BACKENDS)}"
            )
    for key in ("backend_version", "ais_backend_version"):
        if key in data and data[key] is not None:
            data[key] = _string(data[key], f"{path}.{key}")
    for key in (
        "tp_size",
        "moe_tp_size",
        "moe_ep_size",
        "attention_dp_size",
    ):
        if key == "tp_size" and key in data and data[key] is None:
            raise MatchConfigError(f"{path}.tp_size: expected an integer")
        if key in data and data[key] is not None:
            data[key] = _integer(data[key], f"{path}.{key}", positive=True)
    if "runtime" in data:
        runtime_path = f"{path}.runtime"
        runtime = dict(_mapping(data["runtime"], runtime_path))
        _only_keys(runtime, _ENGINE_RUNTIME_KEYS, runtime_path)
        if "cold_start_delay_s" in runtime:
            raw = runtime["cold_start_delay_s"]
            runtime["cold_start_delay_s"] = (
                None
                if raw is None
                else _number(
                    raw, f"{runtime_path}.cold_start_delay_s", nonnegative=True
                )
            )
        if "kv_transfer_bandwidth_gbps" in runtime:
            raw = runtime["kv_transfer_bandwidth_gbps"]
            runtime["kv_transfer_bandwidth_gbps"] = (
                None
                if raw is None
                else _number(
                    raw,
                    f"{runtime_path}.kv_transfer_bandwidth_gbps",
                    positive=True,
                )
            )
        if "kv_bytes_per_token" in runtime:
            raw = runtime["kv_bytes_per_token"]
            runtime["kv_bytes_per_token"] = (
                None
                if raw is None
                else _integer(raw, f"{runtime_path}.kv_bytes_per_token", positive=True)
            )
        data["runtime"] = runtime
    if "extra_args" in data:
        extra_args = dict(_mapping(data["extra_args"], f"{path}.extra_args"))
        _validate_json_value(extra_args, f"{path}.extra_args")
        overlap = sorted(_RESERVED_ENGINE_ARGS.intersection(extra_args))
        if overlap:
            raise MatchConfigError(
                f"{path}.extra_args: runner-owned engine identity/accounting "
                f"fields must use their first-class settings: {overlap}"
            )
        data["extra_args"] = extra_args
    return data


def _merge_engine_layers(
    common: Mapping[str, Any], role: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(common)
    common_args = dict(common.get("extra_args", {}))
    role_args = dict(role.get("extra_args", {}))
    common_runtime = dict(common.get("runtime", {}))
    role_runtime = dict(role.get("runtime", {}))
    merged.update(
        {
            key: value
            for key, value in role.items()
            if key not in {"extra_args", "runtime"}
        }
    )
    merged["extra_args"] = _deep_merge_mappings(common_args, role_args)
    merged["runtime"] = _deep_merge_mappings(common_runtime, role_runtime)
    return merged


def _deep_merge_mappings(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge nested engine-argument mappings; role scalars/lists replace common."""

    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge_mappings(existing, value)
        else:
            merged[key] = value
    return merged


def _resolve_engine_config(data: Mapping[str, Any], path: str) -> EngineConfig:
    if "system" not in data:
        raise MatchConfigError(f"{path}.system: field is required after common merge")
    if "backend" not in data:
        raise MatchConfigError(f"{path}.backend: field is required after common merge")
    tp_size = int(data.get("tp_size", 1))
    attention_dp_size = data.get("attention_dp_size")
    attention_dp = int(attention_dp_size) if attention_dp_size is not None else 1
    num_gpus = tp_size * attention_dp
    moe_tp_size = data.get("moe_tp_size")
    moe_ep_size = data.get("moe_ep_size")
    if (moe_tp_size is None) != (moe_ep_size is None):
        raise MatchConfigError(
            f"{path}: moe_tp_size and moe_ep_size must be specified together"
        )
    if moe_tp_size is not None and int(moe_tp_size) * int(moe_ep_size) != num_gpus:
        raise MatchConfigError(
            f"{path}: moe_tp_size × moe_ep_size must equal the engine world "
            f"size ({num_gpus} GPUs)"
        )
    runtime_data = dict(data.get("runtime", {}))
    kv_transfer_bandwidth = runtime_data.get("kv_transfer_bandwidth_gbps")
    kv_bytes_per_token = runtime_data.get("kv_bytes_per_token")
    if (kv_transfer_bandwidth is None) != (kv_bytes_per_token is None):
        raise MatchConfigError(
            f"{path}.runtime: kv_transfer_bandwidth_gbps and "
            "kv_bytes_per_token must be specified together"
        )
    return EngineConfig(
        system=str(data["system"]),
        backend=str(data["backend"]),
        backend_version=data.get("backend_version"),
        tp_size=tp_size,
        moe_tp_size=moe_tp_size,
        moe_ep_size=moe_ep_size,
        attention_dp_size=attention_dp_size,
        num_gpus=num_gpus,
        extra_args=dict(data.get("extra_args", {})),
        runtime=EngineRuntimeConfig(
            cold_start_delay_s=runtime_data.get("cold_start_delay_s"),
            kv_transfer_bandwidth_gbps=kv_transfer_bandwidth,
            kv_bytes_per_token=kv_bytes_per_token,
        ),
        ais_backend=data.get("ais_backend", str(data["backend"])),
        ais_backend_version=data.get(
            "ais_backend_version", data.get("backend_version")
        ),
    )


def _engine_from_substrate_preset(preset: Any, *, path: str) -> EngineConfig:
    layer = _parse_engine_layer(
        {
            "system": preset.system,
            "backend": preset.backend,
            "backend_version": preset.backend_version,
            "ais_backend": preset.backend,
            "ais_backend_version": preset.backend_version,
            "tp_size": preset.tp_size,
            "moe_tp_size": preset.moe_tp_size,
            "moe_ep_size": preset.moe_ep_size,
            "attention_dp_size": preset.attention_dp_size,
            "extra_args": dict(preset.extra_engine_args),
        },
        path,
    )
    engine = _resolve_engine_config(layer, path)
    if engine.num_gpus != preset.gpus_per_worker:
        raise MatchConfigError(
            f"{path}: preset declares gpus_per_worker={preset.gpus_per_worker} "
            f"but its parallelism resolves to {engine.num_gpus}"
        )
    return engine


def _parse_sim_autoscaler(
    value: Any,
    *,
    index: int,
    topology: str,
    gpu_budget: int,
    prefill_num_gpus: int,
    decode_num_gpus: int,
) -> SimAutoscalerConfig:
    path = f"backend.autoscalers[{index}]"
    data = _mapping(value, path)
    _only_keys(data, {"name", "type", "start", "config"}, path)
    name = _name(_required(data, "name", path), f"{path}.name")
    autoscaler_type = _string(_required(data, "type", path), f"{path}.type")
    if autoscaler_type not in {"planner", "keda", "reactive", "static", "jev"}:
        raise MatchConfigError(
            f"{path}.type: expected planner, keda, reactive, static, or jev"
        )
    adapter_config = dict(_mapping(data.get("config", {}), f"{path}.config"))

    if autoscaler_type == "planner":
        if adapter_config:
            raise MatchConfigError(
                f"{path}.config: Planner settings belong in backend.planner_config"
            )
    else:
        allowed = {
            "static": _STATIC_CONFIG_KEYS,
            "reactive": _REACTIVE_CONFIG_KEYS,
            "keda": _KEDA_CONFIG_KEYS,
            "jev": frozenset(_JEV_DEFAULTS),
        }[autoscaler_type]
        _only_keys(adapter_config, allowed, f"{path}.config")
        if topology == "agg":
            unused = {
                "min_prefill",
                "max_prefill",
                "prefill_queue_up",
                "prefill_queue_down",
                "decode_kv_up",
                "decode_kv_down",
                "kv_threshold",
            }.intersection(adapter_config)
        else:
            unused = {"agg_queue_up", "agg_queue_down"}.intersection(adapter_config)
        if unused:
            raise MatchConfigError(
                f"{path}.config: fields are unused for {topology} topology: "
                f"{sorted(unused)}"
            )
        effective_config = _sim_autoscaler_defaults(autoscaler_type, topology=topology)
        effective_config.update(adapter_config)
        adapter_config = effective_config
        _validate_adapter_values(
            adapter_config, path=f"{path}.config", autoscaler_type=autoscaler_type
        )

    if autoscaler_type == "static":
        if "start" in data:
            raise MatchConfigError(
                f"{path}.start: static fleets start at their fixed config; omit start"
            )
        if "num_decode" not in adapter_config:
            raise MatchConfigError(f"{path}.config.num_decode: field is required")
        num_decode = _integer(
            adapter_config["num_decode"],
            f"{path}.config.num_decode",
            positive=True,
        )
        if topology == "disagg":
            if "num_prefill" not in adapter_config:
                raise MatchConfigError(
                    f"{path}.config.num_prefill: field is required for disagg"
                )
            num_prefill = _integer(
                adapter_config["num_prefill"],
                f"{path}.config.num_prefill",
                positive=True,
            )
        else:
            if "num_prefill" in adapter_config:
                raise MatchConfigError(
                    f"{path}.config.num_prefill: omit this field for agg topology"
                )
            num_prefill = 0
            adapter_config["num_prefill"] = 0
        adapter_config["num_decode"] = num_decode
        adapter_config["num_prefill"] = num_prefill
        start = ReplicaCounts(prefill=num_prefill, decode=num_decode)
    else:
        start_data = _mapping(_required(data, "start", path), f"{path}.start")
        _only_keys(start_data, {"prefill", "decode"}, f"{path}.start")
        decode = _integer(
            _required(start_data, "decode", f"{path}.start"),
            f"{path}.start.decode",
            positive=True,
        )
        if topology == "disagg":
            prefill = _integer(
                _required(start_data, "prefill", f"{path}.start"),
                f"{path}.start.prefill",
                positive=True,
            )
        else:
            if "prefill" in start_data:
                raise MatchConfigError(
                    f"{path}.start.prefill: omit this field for agg topology"
                )
            prefill = 0
        start = ReplicaCounts(prefill=prefill, decode=decode)

    if autoscaler_type in {"keda", "reactive", "jev"}:
        _validate_start_against_adapter(
            start, adapter_config, topology=topology, path=path
        )
    initial_gpus = start.prefill * prefill_num_gpus + start.decode * decode_num_gpus
    if initial_gpus > gpu_budget:
        raise MatchConfigError(
            f"{path}: starting fleet requires {initial_gpus} GPUs, exceeding "
            f"backend.gpu_budget={gpu_budget}"
        )
    if autoscaler_type in {"keda", "reactive", "jev"}:
        max_decode = int(adapter_config.get("max_decode", 8))
        max_prefill = (
            int(adapter_config.get("max_prefill", 16)) if topology == "disagg" else 0
        )
        maximum_gpus = max_prefill * prefill_num_gpus + max_decode * decode_num_gpus
        if maximum_gpus > gpu_budget:
            raise MatchConfigError(
                f"{path}.config: configured/default maximum fleet requires "
                f"{maximum_gpus} GPUs, exceeding backend.gpu_budget={gpu_budget}; "
                "set max_prefill/max_decode to a budget-safe combination"
            )
    return SimAutoscalerConfig(
        name=name, type=autoscaler_type, start=start, config=adapter_config
    )


def _sim_autoscaler_defaults(autoscaler_type: str, *, topology: str) -> dict[str, Any]:
    """Return only constructor defaults that affect the selected topology."""

    if autoscaler_type == "static":
        return dict(_STATIC_DEFAULTS)
    if autoscaler_type == "jev":
        return {
            key: value
            for key, value in _JEV_DEFAULTS.items()
            if topology == "disagg" or key not in {"min_prefill", "max_prefill"}
        }
    if autoscaler_type == "reactive":
        keys = {
            "min_decode",
            "max_decode",
            "poll_interval_s",
            "step",
        }
        if topology == "disagg":
            keys.update(
                {
                    "min_prefill",
                    "max_prefill",
                    "prefill_queue_up",
                    "prefill_queue_down",
                    "decode_kv_up",
                    "decode_kv_down",
                }
            )
        else:
            keys.update({"agg_queue_up", "agg_queue_down"})
        return {key: _REACTIVE_DEFAULTS[key] for key in keys}
    if autoscaler_type == "keda":
        keys = {
            "poll_interval_s",
            "queue_threshold",
            "min_decode",
            "max_decode",
            "tolerance",
            "scale_up_stabilization_s",
            "scale_down_stabilization_s",
        }
        if topology == "disagg":
            keys.update({"kv_threshold", "min_prefill", "max_prefill"})
        return {key: _KEDA_DEFAULTS[key] for key in keys}
    return {}


def _validate_adapter_values(
    config: Mapping[str, Any], *, path: str, autoscaler_type: str
) -> None:
    integer_fields = {
        "min_prefill",
        "max_prefill",
        "min_decode",
        "max_decode",
        "prefill_queue_up",
        "prefill_queue_down",
        "agg_queue_up",
        "agg_queue_down",
        "step",
        "num_prefill",
        "num_decode",
        "history_ticks",
        "max_calls",
    }
    positive_float_fields = {
        "poll_interval_s",
        "queue_threshold",
        "kv_threshold",
        "timeout_s",
    }
    nonnegative_float_fields = {
        "decode_kv_up",
        "decode_kv_down",
        "tolerance",
        "scale_up_stabilization_s",
        "scale_down_stabilization_s",
        "min_confidence",
    }
    for key, value in config.items():
        key_path = f"{path}.{key}"
        if key in integer_fields:
            _integer(
                value,
                key_path,
                positive=key
                not in {
                    "prefill_queue_up",
                    "prefill_queue_down",
                    "agg_queue_up",
                    "agg_queue_down",
                },
                nonnegative=key
                in {
                    "prefill_queue_up",
                    "prefill_queue_down",
                    "agg_queue_up",
                    "agg_queue_down",
                },
            )
        elif key in positive_float_fields:
            _number(value, key_path, positive=True)
        elif key in nonnegative_float_fields:
            _number(value, key_path, nonnegative=True)
    if autoscaler_type == "jev":
        if "model" in config:
            _string(config["model"], f"{path}.model")
        failure_mode = _string(
            config.get("failure_mode", "raise"), f"{path}.failure_mode"
        )
        if failure_mode not in {"raise", "hold"}:
            raise MatchConfigError(f"{path}.failure_mode: expected raise or hold")
    for key in (
        "decode_kv_up",
        "decode_kv_down",
        "tolerance",
        "kv_threshold",
        "min_confidence",
    ):
        if key in config and float(config[key]) > 1.0:
            raise MatchConfigError(f"{path}.{key}: must be <= 1")
    if autoscaler_type == "reactive":
        for upper, lower in (
            ("prefill_queue_up", "prefill_queue_down"),
            ("agg_queue_up", "agg_queue_down"),
            ("decode_kv_up", "decode_kv_down"),
        ):
            if upper in config and lower in config and config[lower] > config[upper]:
                raise MatchConfigError(f"{path}: {lower} must be <= {upper}")


def _validate_start_against_adapter(
    start: ReplicaCounts,
    config: Mapping[str, Any],
    *,
    topology: str,
    path: str,
) -> None:
    min_decode = int(config.get("min_decode", 1))
    max_decode = int(config.get("max_decode", 8))
    if min_decode > max_decode:
        raise MatchConfigError(f"{path}.config: min_decode must be <= max_decode")
    if not min_decode <= start.decode <= max_decode:
        raise MatchConfigError(
            f"{path}.start.decode: must be within [{min_decode}, {max_decode}]"
        )
    if topology == "disagg":
        min_prefill = int(config.get("min_prefill", 1))
        max_prefill = int(config.get("max_prefill", 16))
        if min_prefill > max_prefill:
            raise MatchConfigError(f"{path}.config: min_prefill must be <= max_prefill")
        if not min_prefill <= start.prefill <= max_prefill:
            raise MatchConfigError(
                f"{path}.start.prefill: must be within "
                f"[{min_prefill}, {max_prefill}]"
            )


def _parse_real_backend(
    data: Mapping[str, Any], *, base_dir: Path
) -> RealBackendConfig:
    _only_keys(
        data,
        {"type", "endpoint_catalog", "endpoints", "aiperf", "autoscalers"},
        "backend",
    )
    has_catalog = "endpoint_catalog" in data
    has_inline = "endpoints" in data
    if has_catalog == has_inline:
        raise MatchConfigError(
            "backend: set exactly one of endpoint_catalog or endpoints"
        )

    endpoint_catalog: Optional[Path] = None
    if has_catalog:
        catalog_raw = _string(data["endpoint_catalog"], "backend.endpoint_catalog")
        endpoint_catalog = _resolve_path(catalog_raw, base_dir)
        try:
            catalog_data = _load_yaml(
                endpoint_catalog.read_text(), source=str(endpoint_catalog)
            )
        except OSError as exc:
            raise MatchConfigError(
                f"backend.endpoint_catalog: cannot read {endpoint_catalog}: {exc}"
            ) from exc
    else:
        catalog_data = data["endpoints"]
    endpoints = _parse_endpoints(catalog_data)
    endpoint_names = {endpoint.name for endpoint in endpoints}

    aiperf_data = _mapping(data.get("aiperf", {}), "backend.aiperf")
    _only_keys(
        aiperf_data,
        {"executable", "tokenizer", "streaming", "timeout_s", "extra_args"},
        "backend.aiperf",
    )
    executable = _string(
        aiperf_data.get("executable", "aiperf"),
        "backend.aiperf.executable",
    )
    executable_path = Path(executable).expanduser()
    if (
        executable_path.is_absolute()
        or executable.startswith(".")
        or len(executable_path.parts) > 1
    ):
        executable = str(_resolve_path(executable, base_dir))
    tokenizer_raw = aiperf_data.get("tokenizer")
    tokenizer = (
        None
        if tokenizer_raw is None
        else _string(tokenizer_raw, "backend.aiperf.tokenizer")
    )
    streaming = _boolean(aiperf_data.get("streaming", True), "backend.aiperf.streaming")
    timeout_raw = aiperf_data.get("timeout_s")
    timeout_s = (
        None
        if timeout_raw is None
        else _number(timeout_raw, "backend.aiperf.timeout_s", positive=True)
    )
    extra_args_raw = _list(
        aiperf_data.get("extra_args", []), "backend.aiperf.extra_args"
    )
    extra_args = tuple(
        _string(arg, f"backend.aiperf.extra_args[{i}]")
        for i, arg in enumerate(extra_args_raw)
    )
    runner_owned_args = {
        "--model",
        "--url",
        "--endpoint-type",
        "--input-file",
        "--isl-block-size",
        "--prompt-input-tokens-block-size",
        "--synthetic-input-tokens-block-size",
        "--custom-dataset-type",
        "--artifact-dir",
        "--streaming",
        "--tokenizer",
        "--goodput",
    }
    for index, argument in enumerate(extra_args):
        flag = argument.split("=", 1)[0]
        if flag in runner_owned_args:
            raise MatchConfigError(
                f"backend.aiperf.extra_args[{index}]: {flag} is managed by "
                "the Match Config runner"
            )

    items = _list(_required(data, "autoscalers", "backend"), "backend.autoscalers")
    if not items:
        raise MatchConfigError("backend.autoscalers: must not be empty")
    autoscalers = tuple(
        _parse_real_autoscaler(item, index=i, endpoint_names=endpoint_names)
        for i, item in enumerate(items)
    )
    _unique_names([entry.name for entry in autoscalers], "backend.autoscalers")
    endpoint_refs = [entry.endpoint for entry in autoscalers]
    if len(endpoint_refs) != len(set(endpoint_refs)):
        raise MatchConfigError(
            "backend.autoscalers: each live endpoint may be selected only once"
        )

    return RealBackendConfig(
        type="real",
        endpoint_catalog=endpoint_catalog,
        endpoints=endpoints,
        aiperf=AIPerfConfig(
            executable=executable,
            tokenizer=tokenizer,
            streaming=streaming,
            timeout_s=timeout_s,
            extra_args=extra_args,
        ),
        autoscalers=autoscalers,
    )


def _parse_endpoints(value: Any) -> tuple[EndpointConfig, ...]:
    if isinstance(value, Mapping):
        data = _mapping(value, "backend.endpoint_catalog")
        _only_keys(data, {"endpoints"}, "backend.endpoint_catalog")
        items = _list(
            _required(data, "endpoints", "backend.endpoint_catalog"),
            "backend.endpoint_catalog.endpoints",
        )
    else:
        items = _list(value, "backend.endpoints")
    if not items:
        raise MatchConfigError("backend endpoints: must not be empty")
    endpoints: list[EndpointConfig] = []
    for index, item in enumerate(items):
        path = f"backend.endpoints[{index}]"
        endpoint = _mapping(item, path)
        _only_keys(
            endpoint,
            {
                "name",
                "url",
                "model",
                "description",
                "endpoint_type",
                "declared_deployment",
            },
            path,
        )
        declared_deployment = dict(
            _mapping(
                endpoint.get("declared_deployment", {}),
                f"{path}.declared_deployment",
            )
        )
        _validate_json_value(declared_deployment, f"{path}.declared_deployment")
        endpoints.append(
            EndpointConfig(
                name=_name(_required(endpoint, "name", path), f"{path}.name"),
                url=_string(_required(endpoint, "url", path), f"{path}.url"),
                model=_string(_required(endpoint, "model", path), f"{path}.model"),
                description=_string(
                    endpoint.get("description", ""),
                    f"{path}.description",
                    allow_empty=True,
                ),
                endpoint_type=_string(
                    endpoint.get("endpoint_type", "chat"),
                    f"{path}.endpoint_type",
                ),
                declared_deployment=declared_deployment,
            )
        )
    _unique_names([endpoint.name for endpoint in endpoints], "backend.endpoints")
    targets = [
        (endpoint.url, endpoint.model, endpoint.endpoint_type) for endpoint in endpoints
    ]
    if len(targets) != len(set(targets)):
        raise MatchConfigError(
            "backend.endpoints: duplicate URL/model/endpoint_type target"
        )
    return tuple(endpoints)


def _parse_real_autoscaler(
    value: Any, *, index: int, endpoint_names: set[str]
) -> RealAutoscalerConfig:
    path = f"backend.autoscalers[{index}]"
    data = _mapping(value, path)
    _only_keys(
        data,
        {"name", "endpoint", "autoscaler_type", "declared_config"},
        path,
    )
    name = _name(_required(data, "name", path), f"{path}.name")
    endpoint = _name(_required(data, "endpoint", path), f"{path}.endpoint")
    if endpoint not in endpoint_names:
        raise MatchConfigError(
            f"{path}.endpoint: {endpoint!r} is not present in the endpoint catalog"
        )
    autoscaler_type_raw = data.get("autoscaler_type")
    autoscaler_type = (
        None
        if autoscaler_type_raw is None
        else _string(autoscaler_type_raw, f"{path}.autoscaler_type")
    )
    declared_config = dict(
        _mapping(data.get("declared_config", {}), f"{path}.declared_config")
    )
    _validate_json_value(declared_config, f"{path}.declared_config")
    return RealAutoscalerConfig(
        name=name,
        endpoint=endpoint,
        autoscaler_type=autoscaler_type,
        declared_config=declared_config,
    )


def _parse_evaluations(value: Any, *, base_dir: Path) -> tuple[EvaluationConfig, ...]:
    data = _mapping(value, "evaluations")
    _only_keys(
        data,
        {"workloads", "traces", "suites", "exclude", "defaults", "overrides"},
        "evaluations",
    )
    workload_values = _list(data.get("workloads", []), "evaluations.workloads")
    suite_values = _list(data.get("suites", []), "evaluations.suites")
    traces = _parse_external_traces(data.get("traces", []), base_dir=base_dir)
    if not workload_values and not suite_values and not traces:
        raise MatchConfigError(
            "evaluations: select at least one workload, trace, or suite"
        )

    selected: list[str] = []
    for index, selector in enumerate(workload_values):
        selected.extend(
            _expand_workload_selector(selector, f"evaluations.workloads[{index}]")
        )
    for index, selector in enumerate(suite_values):
        suite = _string(selector, f"evaluations.suites[{index}]")
        if suite not in {"synthetic", "recorded", "all"}:
            raise MatchConfigError(
                f"evaluations.suites[{index}]: expected synthetic, recorded, or all"
            )
        selected.extend(_workload_suite(suite))
    selected = _dedupe(selected)
    selected.extend(traces)

    excluded: list[str] = []
    for index, selector in enumerate(
        _list(data.get("exclude", []), "evaluations.exclude")
    ):
        excluded.extend(
            _expand_workload_selector(
                selector,
                f"evaluations.exclude[{index}]",
                external_names=frozenset(traces),
            )
        )
    excluded_set = set(excluded)
    selected = [name for name in selected if name not in excluded_set]
    if not selected:
        raise MatchConfigError(
            "evaluations: workload selection is empty after exclusions"
        )

    defaults = _parse_evaluation_options(
        data.get("defaults", {}), path="evaluations.defaults"
    )
    overrides_raw = _mapping(data.get("overrides", {}), "evaluations.overrides")
    overrides: dict[str, dict[str, Any]] = {}
    for raw_name, options in overrides_raw.items():
        if not isinstance(raw_name, str):
            raise MatchConfigError(
                "evaluations.overrides: workload keys must be strings"
            )
        if raw_name not in selected:
            raise MatchConfigError(
                f"evaluations.overrides.{raw_name}: workload is not selected"
            )
        if raw_name in overrides:
            raise MatchConfigError(
                f"evaluations.overrides: duplicate workload {raw_name!r}"
            )
        overrides[raw_name] = _parse_evaluation_options(
            options, path=f"evaluations.overrides.{raw_name}", partial=True
        )

    evaluations = []
    for workload in selected:
        options = dict(defaults)
        options.update(overrides.get(workload, {}))
        trace = traces.get(workload)
        if trace is not None and trace["format"] == "dynamo":
            if options["max_requests"] is not None:
                raise MatchConfigError(
                    f"{trace['config_path']}: exact Dynamo traces "
                    "do not support max_requests"
                )
            if options["arrival_speedup"] != 1.0:
                raise MatchConfigError(
                    f"{trace['config_path']}: exact Dynamo traces "
                    "do not support arrival_speedup"
                )
        evaluations.append(
            EvaluationConfig(
                workload=workload,
                seed=options["seed"],
                max_requests=options["max_requests"],
                arrival_speedup=options["arrival_speedup"],
                trace_path=None if trace is None else trace["path"],
                trace_paths=() if trace is None else trace["paths"],
                trace_format="mooncake" if trace is None else trace["format"],
                trace_block_size=None if trace is None else trace["block_size"],
                trace_presorted=False if trace is None else trace["presorted"],
            )
        )
    return tuple(evaluations)


def _parse_external_traces(value: Any, *, base_dir: Path) -> dict[str, dict[str, Any]]:
    items = _list(value, "evaluations.traces")
    traces: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(items):
        item_path = f"evaluations.traces[{index}]"
        data = _mapping(raw, item_path)
        _only_keys(
            data,
            {"name", "format", "path", "paths", "block_size", "presorted"},
            item_path,
        )
        name = _name(_required(data, "name", item_path), f"{item_path}.name")
        if name in set(WORKLOADS).union({"synthetic", "recorded", "all"}):
            raise MatchConfigError(
                f"{item_path}.name: {name!r} conflicts with a built-in workload or suite"
            )
        if name in traces:
            raise MatchConfigError(f"evaluations.traces: duplicate name {name!r}")
        trace_format = _string(data.get("format", "mooncake"), f"{item_path}.format")
        if trace_format not in {"mooncake", "dynamo"}:
            raise MatchConfigError(
                f"{item_path}.format: expected 'mooncake' or 'dynamo'"
            )

        if trace_format == "mooncake":
            if "paths" in data:
                raise MatchConfigError(
                    f"{item_path}.paths: format 'mooncake' requires singular path"
                )
            source = _resolve_path(
                _string(_required(data, "path", item_path), f"{item_path}.path"),
                base_dir,
            )
            _require_trace_file(source, path=f"{item_path}.path")
            block_size = _integer(
                data.get("block_size", 512),
                f"{item_path}.block_size",
                positive=True,
            )
            presorted = _boolean(data.get("presorted", False), f"{item_path}.presorted")
            try:
                validate_mooncake_trace(
                    source, block_size=block_size, presorted=presorted
                )
            except ValueError as exc:
                raise MatchConfigError(
                    f"{item_path}.path: invalid Mooncake trace: {exc}"
                ) from exc
            paths = (source,)
            legacy_path: Optional[Path] = source
        else:
            if "path" in data:
                raise MatchConfigError(
                    f"{item_path}.path: format 'dynamo' requires paths"
                )
            if "presorted" in data:
                raise MatchConfigError(
                    f"{item_path}.presorted: not supported for exact Dynamo traces"
                )
            raw_paths = _list(_required(data, "paths", item_path), f"{item_path}.paths")
            if not raw_paths:
                raise MatchConfigError(f"{item_path}.paths: must not be empty")
            resolved_paths: list[Path] = []
            for path_index, raw_path in enumerate(raw_paths):
                source = _resolve_path(
                    _string(raw_path, f"{item_path}.paths[{path_index}]"),
                    base_dir,
                )
                _require_trace_file(source, path=f"{item_path}.paths[{path_index}]")
                if source in resolved_paths:
                    raise MatchConfigError(
                        f"{item_path}.paths[{path_index}]: duplicate trace file "
                        f"{source}"
                    )
                resolved_paths.append(source)
            paths = tuple(resolved_paths)
            legacy_path = None
            raw_block_size = data.get("block_size")
            block_size = (
                None
                if raw_block_size is None
                else _integer(
                    raw_block_size,
                    f"{item_path}.block_size",
                    positive=True,
                )
            )
            presorted = False

        traces[name] = {
            "config_path": item_path,
            "format": trace_format,
            "path": legacy_path,
            "paths": paths,
            "block_size": block_size,
            "presorted": presorted,
        }
    return traces


def _require_trace_file(source: Path, *, path: str) -> None:
    if not source.is_file():
        raise MatchConfigError(f"{path}: trace file does not exist: {source}")


def _parse_evaluation_options(
    value: Any, *, path: str, partial: bool = False
) -> dict[str, Any]:
    data = _mapping(value, path)
    _only_keys(data, {"seed", "max_requests", "arrival_speedup"}, path)
    defaults = {
        "seed": 0,
        "max_requests": None,
        "arrival_speedup": 1.0,
    }
    out: dict[str, Any] = {} if partial else dict(defaults)
    if "seed" in data:
        out["seed"] = _integer(data["seed"], f"{path}.seed", nonnegative=True)
    if "max_requests" in data:
        raw = data["max_requests"]
        out["max_requests"] = (
            None
            if raw is None
            else _integer(raw, f"{path}.max_requests", positive=True)
        )
    if "arrival_speedup" in data:
        out["arrival_speedup"] = _number(
            data["arrival_speedup"],
            f"{path}.arrival_speedup",
            positive=True,
        )
    return out


def _parse_sla_profiles(value: Any) -> tuple[SLAProfileConfig, ...]:
    items = _list(value, "slo_profiles")
    if not items:
        raise MatchConfigError("slo_profiles: must not be empty")
    expanded: list[SLAProfileConfig] = []
    for index, item in enumerate(items):
        path = f"slo_profiles[{index}]"
        data = _mapping(item, path)
        _only_keys(data, {"name", "ttft_ms", "itl_ms", "e2e_ms"}, path)
        source_name = _name(_required(data, "name", path), f"{path}.name")
        dimensions = {
            field: _threshold_values(data.get(field), f"{path}.{field}")
            for field in ("ttft_ms", "itl_ms", "e2e_ms")
        }
        combinations = list(
            product(
                dimensions["ttft_ms"],
                dimensions["itl_ms"],
                dimensions["e2e_ms"],
            )
        )
        if all(value is None for value in combinations[0]):
            raise MatchConfigError(f"{path}: at least one SLA threshold is required")
        use_suffix = len(combinations) > 1
        for ttft, itl, e2e in combinations:
            profile_name = (
                _sla_variant_name(source_name, ttft, itl, e2e)
                if use_suffix
                else source_name
            )
            expanded.append(
                SLAProfileConfig(
                    name=profile_name,
                    source_name=source_name,
                    ttft_ms=ttft,
                    itl_ms=itl,
                    e2e_ms=e2e,
                )
            )
    _unique_names([profile.name for profile in expanded], "slo_profiles")
    return tuple(expanded)


def _threshold_values(value: Any, path: str) -> tuple[Optional[float], ...]:
    if value is None:
        return (None,)
    values = value if isinstance(value, list) else [value]
    if not values:
        raise MatchConfigError(f"{path}: threshold list must not be empty")
    parsed = tuple(
        _number(item, f"{path}[{i}]", positive=True) for i, item in enumerate(values)
    )
    if len(set(parsed)) != len(parsed):
        raise MatchConfigError(f"{path}: duplicate threshold values")
    return parsed


def _sla_variant_name(
    base: str,
    ttft_ms: Optional[float],
    itl_ms: Optional[float],
    e2e_ms: Optional[float],
) -> str:
    parts = []
    for label, value in (
        ("ttft", ttft_ms),
        ("itl", itl_ms),
        ("e2e", e2e_ms),
    ):
        if value is not None:
            parts.append(f"{label}{_number_slug(value)}ms")
    return f"{base}-{'-'.join(parts)}"


def _parse_metrics(value: Any, backend_type: str) -> MetricsConfig:
    data = _mapping(value, "metrics")
    _only_keys(data, {"rank_by", "include"}, "metrics")
    if backend_type == "sim":
        supported = SIM_METRICS
        default_include = DEFAULT_SIM_METRICS
        default_rank = "goodput_per_gpu"
    else:
        supported = REAL_METRICS
        default_include = DEFAULT_REAL_METRICS
        default_rank = "goodput_rps"
    include_raw = _list(data.get("include", list(default_include)), "metrics.include")
    if not include_raw:
        raise MatchConfigError("metrics.include: must not be empty")
    include = tuple(
        _string(metric, f"metrics.include[{i}]") for i, metric in enumerate(include_raw)
    )
    if backend_type == "real":
        include = tuple(_REAL_METRIC_ALIASES.get(metric, metric) for metric in include)
    if len(include) != len(set(include)):
        raise MatchConfigError("metrics.include: duplicate metric")
    rank_by = _string(data.get("rank_by", default_rank), "metrics.rank_by")
    if backend_type == "real":
        rank_by = _REAL_METRIC_ALIASES.get(rank_by, rank_by)
    for metric in (*include, rank_by):
        if metric not in supported:
            raise MatchConfigError(
                f"metrics: {metric!r} is not available for backend {backend_type!r}; "
                f"known: {sorted(supported)}"
            )
    if rank_by not in include:
        raise MatchConfigError(
            "metrics.rank_by: ranked metric must also appear in metrics.include"
        )
    return MetricsConfig(rank_by=rank_by, include=include)


def _parse_execution(value: Any) -> ExecutionConfig:
    data = _mapping(value, "execution")
    _only_keys(data, {"repetitions", "fail_fast", "max_runs"}, "execution")
    repetitions = _integer(
        data.get("repetitions", 1), "execution.repetitions", positive=True
    )
    fail_fast = _boolean(data.get("fail_fast", False), "execution.fail_fast")
    max_runs_raw = data.get("max_runs")
    max_runs = (
        None
        if max_runs_raw is None
        else _integer(max_runs_raw, "execution.max_runs", positive=True)
    )
    return ExecutionConfig(
        repetitions=repetitions, fail_fast=fail_fast, max_runs=max_runs
    )


def _parse_publish(value: Any, *, base_dir: Path) -> PublishConfig:
    data = _mapping(value, "publish")
    _only_keys(data, {"artifact_root", "destinations"}, "publish")
    artifact_root = _resolve_path(
        _string(
            _required(data, "artifact_root", "publish"),
            "publish.artifact_root",
        ),
        base_dir,
    )
    items = _list(_required(data, "destinations", "publish"), "publish.destinations")
    if not items:
        raise MatchConfigError("publish.destinations: must not be empty")
    destinations: list[PublishDestination] = []
    output_paths: dict[Path, str] = {}
    for index, item in enumerate(items):
        path = f"publish.destinations[{index}]"
        destination = _mapping(item, path)
        destination_type = _string(_required(destination, "type", path), f"{path}.type")
        if destination_type == "console":
            _only_keys(destination, {"type"}, path)
            destinations.append(PublishDestination(type="console"))
        elif destination_type in {"json", "html"}:
            _only_keys(destination, {"type", "path", "overwrite"}, path)
            output_path = _resolve_path(
                _string(_required(destination, "path", path), f"{path}.path"),
                base_dir,
            )
            if output_path in output_paths:
                existing_type = output_paths[output_path]
                raise MatchConfigError(
                    f"{path}.path: duplicate output destination {output_path} "
                    f"(already used by {existing_type})"
                )
            output_paths[output_path] = destination_type
            destinations.append(
                PublishDestination(
                    type=destination_type,
                    path=output_path,
                    overwrite=_boolean(
                        destination.get("overwrite", False),
                        f"{path}.overwrite",
                    ),
                )
            )
        else:
            raise MatchConfigError(
                f"{path}.type: expected 'console', 'json', or 'html'"
            )
    return PublishConfig(artifact_root=artifact_root, destinations=tuple(destinations))


def _load_yaml(text: str, *, source: str) -> Any:
    class UniqueKeyLoader(yaml.SafeLoader):
        pass

    def construct_mapping(loader, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as exc:
                raise MatchConfigError(
                    f"{source}:{key_node.start_mark.line + 1}: mapping key must be scalar"
                ) from exc
            if duplicate:
                raise MatchConfigError(
                    f"{source}:{key_node.start_mark.line + 1}: duplicate key {key!r}"
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    try:
        data = yaml.load(text, Loader=UniqueKeyLoader)
    except MatchConfigError:
        raise
    except yaml.YAMLError as exc:
        raise MatchConfigError(f"{source}: invalid YAML: {exc}") from exc
    if data is None:
        raise MatchConfigError(f"{source}: config is empty")
    return data


def _expand_workload_selector(
    value: Any,
    path: str,
    *,
    external_names: frozenset[str] = frozenset(),
) -> list[str]:
    selector = _string(value, path)
    if selector in {"synthetic", "recorded", "all"}:
        return _workload_suite(selector)
    if selector in external_names:
        return [selector]
    if selector not in WORKLOADS:
        known = sorted(
            set(WORKLOADS).union(external_names).union({"synthetic", "recorded", "all"})
        )
        raise MatchConfigError(
            f"{path}: unknown workload selector {selector!r}; known: {known}"
        )
    return [selector]


def _workload_suite(name: str) -> list[str]:
    if name == "all":
        return list(WORKLOADS)
    synthetic = name == "synthetic"
    return [
        workload_name
        for workload_name, workload in WORKLOADS.items()
        if workload.is_synthetic == synthetic
    ]


def _labels(value: Any) -> dict[str, str]:
    data = _mapping(value, "labels")
    labels: dict[str, str] = {}
    for key, item in data.items():
        if not isinstance(key, str) or not key:
            raise MatchConfigError("labels: keys must be non-empty strings")
        labels[key] = _string(item, f"labels.{key}", allow_empty=True)
    return labels


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MatchConfigError(f"{path}: expected a mapping")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise MatchConfigError(f"{path}: expected a list")
    return value


def _only_keys(
    data: Mapping[str, Any], allowed: set[str] | frozenset[str], path: str
) -> None:
    unknown = sorted(str(key) for key in data if key not in allowed)
    if unknown:
        raise MatchConfigError(f"{path}: unknown fields: {unknown}")


def _required(data: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in data:
        raise MatchConfigError(f"{path}.{key}: field is required")
    return data[key]


def _string(value: Any, path: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise MatchConfigError(f"{path}: expected a string")
    if not allow_empty and not value.strip():
        raise MatchConfigError(f"{path}: must not be empty")
    return value


def _name(value: Any, path: str) -> str:
    name = _string(value, path)
    if len(name) > 64 or not _SAFE_NAME.fullmatch(name) or name in {".", ".."}:
        raise MatchConfigError(
            f"{path}: use a filesystem-safe identifier containing only "
            "letters, numbers, '.', '_' and '-' (maximum 64 characters)"
        )
    return name


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise MatchConfigError(f"{path}: expected true or false")
    return value


def _integer(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatchConfigError(f"{path}: expected an integer")
    if positive and value <= 0:
        raise MatchConfigError(f"{path}: must be > 0")
    if nonnegative and value < 0:
        raise MatchConfigError(f"{path}: must be >= 0")
    return value


def _number(
    value: Any,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatchConfigError(f"{path}: expected a number")
    number = float(value)
    if not math.isfinite(number):
        raise MatchConfigError(f"{path}: must be finite")
    if positive and number <= 0:
        raise MatchConfigError(f"{path}: must be > 0")
    if nonnegative and number < 0:
        raise MatchConfigError(f"{path}: must be >= 0")
    return number


def _unique_names(names: Sequence[str], path: str) -> None:
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise MatchConfigError(f"{path}: duplicate name {name!r}")
        seen.add(name)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _number_slug(value: float) -> str:
    rendered = repr(value)
    if rendered.endswith(".0"):
        rendered = rendered[:-2]
    return rendered.replace("-", "neg").replace("+", "").replace(".", "p")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _validate_json_value(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise MatchConfigError(f"{path}: numeric values must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise MatchConfigError(f"{path}: mapping keys must be strings")
            _validate_json_value(item, f"{path}.{key}")
        return
    raise MatchConfigError(
        f"{path}: value of type {type(value).__name__} is not supported"
    )


__all__ = [
    "AIPerfConfig",
    "EngineConfig",
    "EngineRuntimeConfig",
    "EndpointConfig",
    "EvaluationConfig",
    "ExecutionConfig",
    "MatchConfig",
    "MatchConfigError",
    "MatchRun",
    "MetricsConfig",
    "ModelConfig",
    "PublishConfig",
    "PublishDestination",
    "REAL_METRICS",
    "RealAutoscalerConfig",
    "RealBackendConfig",
    "ReplayConfig",
    "ReplicaCounts",
    "SCHEMA_VERSION",
    "SIM_METRICS",
    "SLAProfileConfig",
    "SimAutoscalerConfig",
    "SimBackendConfig",
    "SimEnginesConfig",
    "load_match_config",
    "parse_match_config",
]
