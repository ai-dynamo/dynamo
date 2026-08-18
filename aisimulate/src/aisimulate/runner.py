# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-free engine replay implementation of the Sweeper Runner contract."""

from __future__ import annotations

import importlib
import json
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Protocol, runtime_checkable

from .aic import materialize_aic_num_gpu_blocks
from .sweeper.provider import JSONValue
from .sweeper.replay import (
    ReplayOutputRequirements,
    ReplayReport,
    ReplaySpec,
    RunnerCapabilities,
)
from .traffic import materialize_configured_traffic

_SUPPORTED_BACKEND_TOPOLOGIES = (
    ("vllm", "agg"),
    ("vllm", "disagg"),
    ("sglang", "agg"),
    ("sglang", "disagg"),
    ("trtllm", "agg"),
)

_AIC_TIMING_FIELD_ALIASES = {
    "backend_version": ("backend_version", "aic_backend_version"),
    "pp": ("aic_pp_size",),
    "moe_tp_size": ("moe_tp_size", "aic_moe_tp_size"),
    "moe_ep_size": ("moe_ep_size", "aic_moe_ep_size"),
    "gemm_dtype": ("gemm_dtype", "aic_gemm_dtype"),
    "moe_dtype": ("moe_dtype", "aic_moe_dtype"),
    "fmha_dtype": ("fmha_dtype", "aic_fmha_dtype"),
    "kv_cache_dtype": ("kv_cache_dtype", "aic_kv_cache_dtype"),
    "comm_dtype": ("comm_dtype", "aic_comm_dtype"),
    "systems_path": ("systems_path",),
}


class RunnerUnavailableError(RuntimeError):
    """The runtime needed for a requested replay stack is not installed."""


class InvalidRunnerError(RuntimeError):
    """An installed replay runtime does not implement the runner contract."""


@runtime_checkable
class EngineReplayRuntime(Protocol):
    """Serialized execution seam used by the engine-only Runner."""

    def run_replay_json(self, execution_spec_json: str) -> str:
        """Run one serialized execution spec and return serialized report JSON."""


@dataclass(frozen=True)
class EngineReplayRunnerFactory:
    """Create reusable engine-only Runners for Sweeper candidates.

    The optional runtime is a test seam. Production factories leave it unset so
    the dataclass remains process-serializable and each worker loads its local
    compiled runtime lazily.
    """

    trace_block_size: int = 512
    runtime: EngineReplayRuntime | None = field(default=None, repr=False, compare=False)

    def capabilities(self) -> RunnerCapabilities:
        return RunnerCapabilities(
            replay_spec_api_version=1,
            supported_backend_topologies=_SUPPORTED_BACKEND_TOPOLOGIES,
            supports_disaggregated_attention_dp=False,
        )

    def create(self, worker_id: int) -> "EngineReplayRunner":
        return EngineReplayRunner(
            worker_id=worker_id,
            capabilities=self.capabilities(),
            trace_block_size=self.trace_block_size,
            runtime=self.runtime,
        )


@dataclass
class EngineReplayRunner:
    """Execute one candidate with the AISimulate Engine/Replayer composition."""

    worker_id: int
    capabilities: RunnerCapabilities
    trace_block_size: int = 512
    runtime: EngineReplayRuntime | None = None

    def _resolve_runtime(self) -> EngineReplayRuntime:
        if self.runtime is not None:
            return self.runtime

        try:
            runtime = importlib.import_module("aisimulate._runtime")
        except ModuleNotFoundError as exc:
            if exc.name != "aisimulate._runtime":
                raise
            raise RunnerUnavailableError(
                "AISimulate engine replay runtime is not installed. Install a "
                "binary aisimulate package or build its native extension."
            ) from exc
        except ImportError as exc:
            raise RunnerUnavailableError(
                "AISimulate engine replay runtime could not be loaded. Reinstall "
                "the binary aisimulate package or rebuild its native extension."
            ) from exc

        if not isinstance(runtime, EngineReplayRuntime):
            raise InvalidRunnerError(
                "aisimulate._runtime must export "
                "run_replay_json(execution_spec_json: str) -> str"
            )
        self.runtime = runtime
        return runtime

    def run(
        self,
        spec: ReplaySpec,
        *,
        output_requirements: ReplayOutputRequirements | None = None,
    ) -> ReplayReport:
        output_requirements = output_requirements or ReplayOutputRequirements()
        self.capabilities.require_compatible(spec)
        execution_spec = _materialize_engine_execution_spec(
            spec,
            trace_block_size=self.trace_block_size,
            record_per_request=output_requirements.capture_per_request,
        )
        execution_spec_json = json.dumps(
            execution_spec,
            allow_nan=False,
            separators=(",", ":"),
        )
        report_json = self._resolve_runtime().run_replay_json(execution_spec_json)
        if not isinstance(report_json, str):
            raise InvalidRunnerError(
                "AISimulate engine replay runtime report must be a JSON string"
            )
        try:
            report = json.loads(report_json)
        except json.JSONDecodeError as exc:
            raise InvalidRunnerError(
                "AISimulate engine replay runtime returned invalid report JSON"
            ) from exc
        if not isinstance(report, Mapping):
            raise InvalidRunnerError(
                "AISimulate engine replay runtime report must be a JSON object"
            )
        return _normalize_engine_replay_report(
            report,
            include_native_report=(
                output_requirements.include_raw_report
                or output_requirements.capture_per_request
            ),
        )

    def close(self) -> None:
        """Release worker-local resources.

        The current extension owns no persistent resources beyond its imported
        module, so closing is intentionally a no-op.
        """


def _materialize_engine_execution_spec(
    spec: ReplaySpec,
    *,
    trace_block_size: int,
    record_per_request: bool,
) -> dict[str, JSONValue]:
    """Translate the public Runner input into the Rust replay wire schema.

    This is the only high-level-to-execution materializer. The compiled runtime
    deliberately knows nothing about CLI, Sweeper, or provider discovery.
    """

    deployment = spec.backend_deployment
    deployment_mode = deployment.deployment_mode
    if deployment_mode == "agg":
        raw_engine_args = _required_engine_args(
            deployment.agg_engine_args, "aggregated"
        )
        engine = _materialize_engine_role(
            deployment.backend,
            deployment.backend_version,
            deployment.parallel_config,
            raw_engine_args,
            "aggregated",
        )
        _require_parallel_match(
            deployment.parallel_config,
            "replicas",
            deployment.num_workers,
            "num_workers",
        )
        topology: dict[str, JSONValue] = {
            "kind": "aggregated",
            "workers": {
                "initial_workers": _positive_int(deployment.num_workers, "num_workers"),
                "startup_delay_ms": _startup_delay_ms(raw_engine_args),
            },
        }
    elif deployment_mode == "disagg":
        if deployment.backend == "trtllm":
            raise ValueError("engine replay does not support TensorRT-LLM disagg")
        raw_prefill = _required_engine_args(deployment.prefill_engine_args, "prefill")
        raw_decode = _required_engine_args(deployment.decode_engine_args, "decode")
        prefill = _materialize_engine_role(
            deployment.backend,
            deployment.backend_version,
            deployment.parallel_config,
            raw_prefill,
            "prefill",
        )
        decode = _materialize_engine_role(
            deployment.backend,
            deployment.backend_version,
            deployment.parallel_config,
            raw_decode,
            "decode",
        )
        for role, role_config in (("prefill", prefill), ("decode", decode)):
            # TODO(#12965): Keep this fail-fast until disaggregated handoff and
            # lifecycle evidence carry a logical-worker plus DP-rank identity.
            if role_config["dp_size"] != 1:
                raise ValueError(
                    "disaggregated engine replay requires "
                    f"{role} dp_size=1; attention-DP handoff identity is not implemented"
                )
        _require_parallel_match(
            deployment.parallel_config,
            "prefill_replicas",
            deployment.num_prefill_workers,
            "num_prefill_workers",
        )
        _require_parallel_match(
            deployment.parallel_config,
            "decode_replicas",
            deployment.num_decode_workers,
            "num_decode_workers",
        )
        prefill_backend = prefill["rank"].get("backend", "vllm")
        decode_backend = decode["rank"].get("backend", "vllm")
        if prefill_backend != decode_backend:
            raise ValueError(
                "disaggregated prefill and decode must use the same backend: "
                f"{prefill_backend!r} != {decode_backend!r}"
            )
        if prefill_backend == "trtllm":
            raise ValueError(
                "engine replay does not support TensorRT-LLM disaggregated mode"
            )
        engine = {"prefill": prefill, "decode": decode}
        topology = {
            "kind": "disaggregated",
            "prefill": {
                "initial_workers": _positive_int(
                    deployment.num_prefill_workers, "num_prefill_workers"
                ),
                "startup_delay_ms": _startup_delay_ms(raw_prefill),
            },
            "decode": {
                "initial_workers": _positive_int(
                    deployment.num_decode_workers, "num_decode_workers"
                ),
                "startup_delay_ms": _startup_delay_ms(raw_decode),
            },
            "handoff_latency_ms": 0.0,
        }
    else:
        raise ValueError(
            "engine replay deployment_mode must be 'agg' or 'disagg', got "
            f"{deployment_mode!r}"
        )

    requests, max_in_flight = _materialize_requests(spec, trace_block_size)
    sla = _materialize_sla(spec)
    max_sim_time_ms = spec.workload.get("max_sim_time_ms")
    if max_sim_time_ms is not None:
        max_sim_time_ms = _nonnegative_time(max_sim_time_ms, "max_sim_time_ms")

    return {
        "version": 1,
        "topology": topology,
        "engine": engine,
        "adapters": {
            "placement": {
                "provider": "round_robin",
                "config": None,
            },
            "scaling": {
                "provider": "none",
                "config": None,
            },
        },
        "max_sim_time_ms": max_sim_time_ms,
        "max_in_flight": max_in_flight,
        "record_per_request": record_per_request,
        "sla": sla,
        "requests": requests,
    }


def _required_engine_args(
    payload: dict[str, JSONValue] | None, role: str
) -> dict[str, JSONValue]:
    if payload is None:
        raise ValueError(f"ReplaySpec is missing {role} engine arguments")
    return dict(payload)


def _startup_delay_ms(payload: Mapping[str, JSONValue]) -> float:
    value = payload.get("startup_time", 0.0)
    return 1_000.0 * _nonnegative_time(value, "startup_time")


def _materialize_requests(
    spec: ReplaySpec, trace_block_size: int
) -> tuple[list[dict[str, JSONValue]], int | None]:
    workload = spec.workload
    trace_path = workload.get("trace_path")
    if trace_path is not None:
        if not isinstance(trace_path, str) or not trace_path:
            raise TypeError("trace_path must be a non-empty string")
        configured_trace_block_size = workload.get("trace_block_size")
        requests = materialize_configured_traffic(
            {
                "trace_path": trace_path,
                "format": workload.get("trace_format", "mooncake"),
                "trace_block_size": _positive_int(
                    trace_block_size
                    if configured_trace_block_size is None
                    else configured_trace_block_size,
                    "trace_block_size",
                ),
                "speedup": _positive_number(
                    workload.get("arrival_speedup_ratio", 1.0),
                    "arrival_speedup_ratio",
                ),
            }
        )
        cap = workload.get("replay_concurrency")
        return requests, (
            _positive_int(cap, "replay_concurrency") if cap is not None else None
        )

    isl = _positive_int(workload.get("isl"), "isl")
    osl = _positive_int(workload.get("osl"), "osl")
    _require_supported_synthetic_workload(workload)
    concurrency = spec.concurrency
    if concurrency is None and workload.get("concurrency") is not None:
        concurrency = _positive_int(workload["concurrency"], "concurrency")
    request_rate = workload.get("request_rate")
    rate = (
        _positive_number(request_rate, "request_rate")
        if request_rate is not None
        else None
    )
    raw_interval = workload.get("arrival_interval_ms")
    arrival_interval_ms = (
        _nonnegative_time(raw_interval, "arrival_interval_ms")
        if raw_interval is not None
        else None
    )
    if rate is not None and arrival_interval_ms is not None:
        raise ValueError(
            "synthetic replay cannot combine request_rate and arrival_interval_ms"
        )
    if concurrency is not None:
        load = float(concurrency)
    elif rate is not None:
        load = rate
    elif arrival_interval_ms is None:
        raise ValueError(
            "synthetic replay requires concurrency, request_rate, or "
            "arrival_interval_ms"
        )
    raw_request_count = workload.get("request_count")
    if raw_request_count is not None:
        request_count = _positive_int(raw_request_count, "request_count")
    else:
        if arrival_interval_ms is not None:
            raise ValueError(
                "synthetic replay with arrival_interval_ms requires request_count"
            )
        ratio = _positive_number(workload.get("num_request_ratio"), "num_request_ratio")
        request_count = max(1, round(ratio * load))

    if rate is not None:
        arrival_seed = workload.get("arrival_seed", 42)
        if (
            not isinstance(arrival_seed, int)
            or isinstance(arrival_seed, bool)
            or arrival_seed < 0
        ):
            raise ValueError("arrival_seed must be a non-negative integer")
        rng = random.Random(arrival_seed)
        arrival_times = [0.0]
        for _ in range(1, request_count):
            arrival_times.append(arrival_times[-1] + rng.expovariate(rate) * 1_000.0)
    else:
        interval = arrival_interval_ms or 0.0
        arrival_times = [index * interval for index in range(request_count)]
    requests = [
        {
            "id": f"synthetic-{index}",
            "arrival_time_ms": arrival_times[index],
            "input_tokens": isl,
            "output_tokens": osl,
            "metadata": None,
        }
        for index in range(request_count)
    ]
    return requests, concurrency


def _materialize_sla(spec: ReplaySpec) -> dict[str, JSONValue]:
    raw = spec.goal.get("sla")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("goal.sla must be a mapping")
    return {
        key: _positive_number(raw[key], f"sla.{key}")
        for key in ("ttft_ms", "itl_ms", "e2e_ms")
        if raw.get(key) is not None
    }


def _materialize_engine_role(
    deployment_backend: str,
    deployment_backend_version: str,
    parallel_config: Mapping[str, JSONValue],
    raw_config: Mapping[str, JSONValue],
    role: str,
) -> dict[str, JSONValue]:
    """Materialize one single-rank or attention-DP generalized engine."""

    role_config = dict(raw_config)
    # The shared CLI/Sweeper form is flat. Nested rank descriptors are already
    # execution-level input and retain the native runtime's compatibility
    # fallback after their structure has been validated below.
    if "rank" not in role_config:
        role_config = materialize_aic_num_gpu_blocks(role_config)
    for name in ("engine_type", "aic_backend"):
        configured = role_config.pop(name, None)
        if configured is not None and configured != deployment_backend:
            raise ValueError(
                f"{role} {name}={configured!r} conflicts with "
                f"BackendDeploymentSpec backend={deployment_backend!r}"
            )
    role_config.pop("worker_type", None)
    role_config.pop("startup_time", None)
    model = role_config.pop("aic_model_path", None)
    system = role_config.pop("aic_system", None)
    raw_dp_size = _pop_matching_aliases(
        role_config, "attention DP", ("dp_size", "aic_attention_dp_size"), 1
    )
    raw_tp_size = _pop_matching_aliases(
        role_config, "tensor parallel", ("tensor_parallel_size", "aic_tp_size"), 1
    )
    dp_size = _positive_int(
        raw_dp_size,
        f"engine provider {role} dp_size",
    )
    tensor_parallel_size = _positive_int(
        raw_tp_size,
        f"engine provider {role} tensor_parallel_size",
    )
    parallel_prefix = "" if role == "aggregated" else f"{role}_"
    _require_parallel_match(
        parallel_config,
        f"{parallel_prefix}tp",
        tensor_parallel_size,
        f"{role} tensor parallel size",
    )
    _require_parallel_match(
        parallel_config,
        f"{parallel_prefix}attention_dp",
        dp_size,
        f"{role} attention DP size",
    )

    nested_rank = role_config.pop("rank", None)
    if nested_rank is not None:
        if role_config:
            unexpected = ", ".join(sorted(role_config))
            raise ValueError(
                f"engine provider {role} config cannot mix nested 'rank' with "
                f"rank fields: {unexpected}"
            )
        if not isinstance(nested_rank, dict):
            raise ValueError(f"engine provider {role} rank config must be a mapping")
        rank: dict[str, JSONValue] = dict(nested_rank)
    else:
        # Preserve the existing convenient flat aggregated form. Disaggregated
        # role mappings may use the same shorthand.
        rank = role_config

    configured_backend = rank.get("backend")
    if configured_backend is not None and not isinstance(configured_backend, str):
        raise ValueError(f"engine provider {role} rank backend must be a string")
    backend = configured_backend or deployment_backend
    if backend not in {"vllm", "sglang", "trtllm"}:
        raise ValueError(
            f"engine replay supports vllm, sglang, and trtllm, got {backend!r}"
        )
    if configured_backend is not None and configured_backend != deployment_backend:
        raise ValueError(
            "engine provider rank backend conflicts with deployment backend: "
            f"{configured_backend!r} != {deployment_backend!r}"
        )
    if "block_size" not in rank:
        # Keep scheduler capacity, AIC compilation, and synthetic-prefix
        # materialization on one explicit backend-native block size.
        rank["block_size"] = {
            "vllm": 64,
            "sglang": 1,
            "trtllm": 32,
        }[backend]
    rank["backend"] = backend

    memory_fraction_overrides: dict[str, JSONValue] = {}
    for memory_field in (
        "gpu_memory_utilization",
        "mem_fraction_static",
        "free_gpu_memory_fraction",
    ):
        if memory_field not in rank:
            continue
        value = rank.pop(memory_field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"engine provider {role} {memory_field} must be between 0 and 1"
            )
        memory_fraction_overrides[memory_field] = float(value)

    aic_timing_overrides: dict[str, JSONValue] = {}
    for target, aliases in _AIC_TIMING_FIELD_ALIASES.items():
        configured = [alias for alias in aliases if alias in rank]
        if len(configured) > 1:
            names = ", ".join(configured)
            raise ValueError(
                f"engine provider {role} config duplicates AIC field {target}: {names}"
            )
        if not configured:
            continue
        value = rank.pop(configured[0])
        if target in {"pp", "moe_tp_size", "moe_ep_size"}:
            value = _positive_int(value, f"engine provider {role} {target}")
        elif not isinstance(value, str) or not value:
            raise ValueError(f"engine provider {role} {target} must be a string")
        aic_timing_overrides[target] = value

    if deployment_backend_version:
        configured_version = aic_timing_overrides.get("backend_version")
        if (
            configured_version is not None
            and configured_version != deployment_backend_version
        ):
            raise ValueError(
                f"engine provider {role} backend version {configured_version!r} "
                "conflicts with BackendDeploymentSpec backend_version="
                f"{deployment_backend_version!r}"
            )
        timing_model = rank.get("timing_model")
        uses_aic_timing = timing_model is None or (
            isinstance(timing_model, dict)
            and timing_model.get("type") == "external"
            and timing_model.get("provider") == "aic"
        )
        if uses_aic_timing:
            timing_config = (
                timing_model.get("config") if isinstance(timing_model, dict) else None
            )
            timing_backend_version = (
                timing_config.get("backend_version")
                if isinstance(timing_config, dict)
                else None
            )
            if (
                timing_backend_version is not None
                and timing_backend_version != deployment_backend_version
            ):
                raise ValueError(
                    f"engine provider {role} timing_model.config.backend_version="
                    f"{timing_backend_version!r} conflicts with "
                    "BackendDeploymentSpec backend_version="
                    f"{deployment_backend_version!r}"
                )
            aic_timing_overrides["backend_version"] = deployment_backend_version

    nextn = _pop_alias(rank, "aic_nextn", ("aic_nextn", "nextn"))
    if nextn is not None:
        nextn = _positive_int(nextn, f"engine provider {role} aic_nextn")
        if nextn > 5:
            raise ValueError(f"engine provider {role} aic_nextn must be in 1..=5")
        rank["aic_nextn"] = nextn

    accept_rates = _pop_alias(
        rank,
        "aic_nextn_accept_rates",
        ("aic_nextn_accept_rates", "nextn_accept_rates"),
    )
    if accept_rates is not None:
        if nextn is None:
            raise ValueError(
                f"engine provider {role} aic_nextn_accept_rates requires aic_nextn"
            )
        if not isinstance(accept_rates, str):
            raise ValueError(
                f"engine provider {role} aic_nextn_accept_rates must be a string"
            )
        rank["aic_nextn_accept_rates"] = accept_rates

    mtp_seed = _pop_alias(rank, "aic_mtp_seed", ("aic_mtp_seed", "mtp_seed"))
    if mtp_seed is not None:
        if (
            not isinstance(mtp_seed, int)
            or isinstance(mtp_seed, bool)
            or not 0 <= mtp_seed <= 0xFFFF_FFFF_FFFF_FFFF
        ):
            raise ValueError(
                f"engine provider {role} aic_mtp_seed must be an unsigned "
                "64-bit integer"
            )
        rank["aic_mtp_seed"] = mtp_seed

    if "timing_model" not in rank:
        if not isinstance(model, str) or not model:
            raise ValueError(f"{role} engine arguments require aic_model_path")
        if not isinstance(system, str) or not system:
            raise ValueError(f"{role} engine arguments require aic_system")
        timing_config: dict[str, JSONValue] = {
            "model": model,
            "backend": backend,
            "system": system,
            "tp": tensor_parallel_size,
            "attention_dp": dp_size,
        }
        block_size = rank.get("block_size")
        if isinstance(block_size, int) and not isinstance(block_size, bool):
            timing_config["kv_block_size"] = block_size
        timing_config.update(memory_fraction_overrides)
        timing_config.update(aic_timing_overrides)
        if nextn is not None:
            timing_config["nextn"] = nextn
        rank["timing_model"] = {
            "type": "external",
            "provider": "aic",
            "config": timing_config,
        }
    elif memory_fraction_overrides or aic_timing_overrides:
        timing_model = rank["timing_model"]
        if (
            not isinstance(timing_model, dict)
            or timing_model.get("type") != "external"
            or timing_model.get("provider") != "aic"
            or not isinstance(timing_model.get("config"), dict)
        ):
            raise ValueError("engine AIC overrides require an AIC timing model")
        timing_model = dict(timing_model)
        timing_config = dict(timing_model["config"])
        timing_overrides = memory_fraction_overrides | aic_timing_overrides
        duplicates = timing_overrides.keys() & timing_config.keys()
        if (
            "backend_version" in duplicates
            and timing_overrides["backend_version"] == timing_config["backend_version"]
        ):
            duplicates.remove("backend_version")
        if duplicates:
            duplicate = ", ".join(sorted(duplicates))
            raise ValueError(
                "engine AIC option is configured both on the rank and "
                f"inside timing_model.config: {duplicate}"
            )
        timing_config.update(timing_overrides)
        timing_model["config"] = timing_config
        rank["timing_model"] = timing_model

    if nextn is not None:
        timing_model = rank["timing_model"]
        if (
            isinstance(timing_model, dict)
            and timing_model.get("type") == "external"
            and timing_model.get("provider") == "aic"
            and isinstance(timing_model.get("config"), dict)
        ):
            timing_model = dict(timing_model)
            timing_config = dict(timing_model["config"])
            configured_nextn = timing_config.get("nextn")
            if configured_nextn is not None and configured_nextn != nextn:
                raise ValueError(
                    f"engine provider {role} aic_nextn={nextn} conflicts with "
                    f"timing_model.config.nextn={configured_nextn!r}"
                )
            timing_config["nextn"] = nextn
            timing_model["config"] = timing_config
            rank["timing_model"] = timing_model

    return {
        "dp_size": dp_size,
        "tensor_parallel_size": tensor_parallel_size,
        "rank": rank,
    }


def _positive_int(value: JSONValue, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_parallel_match(
    parallel_config: Mapping[str, JSONValue],
    field: str,
    actual: int,
    label: str,
) -> None:
    expected = parallel_config.get(field)
    if expected is None:
        return
    expected = _positive_int(expected, f"parallel_config.{field}")
    if expected != actual:
        raise ValueError(
            f"parallel_config.{field}={expected} conflicts with {label}={actual}"
        )


def _require_supported_synthetic_workload(
    workload: Mapping[str, JSONValue],
) -> None:
    unsupported: list[str] = []
    if workload.get("turns_per_session", 1) != 1:
        unsupported.append("turns_per_session")
    if workload.get("shared_prefix_ratio", 0.0) != 0.0:
        unsupported.append("shared_prefix_ratio")
    if workload.get("num_prefix_groups", 0) != 0:
        unsupported.append("num_prefix_groups")
    if workload.get("inter_turn_delay_ms", 0.0) != 0.0:
        unsupported.append("inter_turn_delay_ms")
    if unsupported:
        raise ValueError(
            "engine replay synthetic traffic does not yet support "
            + ", ".join(unsupported)
        )


def _pop_alias(
    config: dict[str, JSONValue],
    target: str,
    aliases: tuple[str, ...],
) -> JSONValue | None:
    configured = [alias for alias in aliases if alias in config]
    if len(configured) > 1:
        names = ", ".join(configured)
        raise ValueError(f"engine config duplicates {target}: {names}")
    if not configured:
        return None
    return config.pop(configured[0])


def _pop_matching_aliases(
    config: dict[str, JSONValue],
    target: str,
    aliases: tuple[str, ...],
    default: JSONValue,
) -> JSONValue:
    values = [config.pop(alias) for alias in aliases if alias in config]
    if not values:
        return default
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"engine config has conflicting {target} aliases")
    return values[0]


def _nonnegative_time(value: JSONValue, name: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and non-negative")
    return float(value)


def _positive_number(value: JSONValue, name: str) -> float:
    number = _nonnegative_time(value, name)
    if number == 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _normalize_engine_replay_report(
    report: Mapping[str, JSONValue], *, include_native_report: bool
) -> ReplayReport:
    """Normalize an execution report to Sweeper's stable scoring metric names."""

    payload = dict(report)
    metrics: dict[str, float] = {}

    def add(name: str, value: object) -> None:
        if isinstance(value, bool) or not isinstance(value, Real):
            return
        number = float(value)
        if not math.isfinite(number):
            raise InvalidRunnerError(f"engine replay metric {name!r} is not finite")
        metrics[name] = number

    # The Rust ReplayReport's flat serialization is the sole wire
    # format. Per-request records remain a Rust replay concern and are not
    # reconstructed into a second Python report model.
    for name, value in payload.items():
        add(name, value)
    metadata = {"native_report": payload} if include_native_report else {}
    return ReplayReport(metrics=metrics, metadata=metadata)
