# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility lowering from a Sweeper candidate to ``GeneratorRequest``.

This is copied from ai-dynamo/aisimulate commit
ac7ac2fa3c5e89b8bd05836da913b52731c4599d. Dynamo currently pins a published
AIC version from before that bridge. The renderer prefers the installed bridge
and uses this copy only until a compatible AI Simulate release can be pinned.

The Sweeper owns search and ranking. The generator owns deterministic lowering
and artifact rendering. This module is the deliberately small boundary between
those components: it consumes the public, JSON-shaped candidate contract and
does not import Sweeper or Dynamo implementation types.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from typing import Any

from aiconfigurator.generator.request.schema import (
    BackendSpec,
    EmitTargets,
    GeneratorRequest,
    ModelFacts,
    ModelSpec,
    Overrides,
    Platform,
    RoleSizing,
    SlaSpec,
    Topology,
)


class SweeperCandidateError(ValueError):
    """A candidate cannot be represented faithfully by the generator."""


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return copy.deepcopy(dict(value))
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return copy.deepcopy(dict(dumped))
    if is_dataclass(value) and not isinstance(value, type):
        dumped = asdict(value)
        if isinstance(dumped, Mapping):
            return copy.deepcopy(dict(dumped))
    raise SweeperCandidateError(f"{label} must be a mapping or a serializable model")


def _required_text(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SweeperCandidateError(f"candidate.config.{key} is required")
    return value.strip()


def _positive_int(value: Any, *, path: str) -> int:
    if isinstance(value, bool):
        raise SweeperCandidateError(f"{path} must be a positive integer")
    try:
        normalized = int(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise SweeperCandidateError(f"{path} must be a positive integer") from exc
    if normalized <= 0 or normalized != value:
        raise SweeperCandidateError(f"{path} must be a positive integer")
    return normalized


def _positive_float(value: Any, *, path: str) -> float:
    if isinstance(value, bool):
        raise SweeperCandidateError(f"{path} must be a positive number")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise SweeperCandidateError(f"{path} must be a positive number") from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise SweeperCandidateError(f"{path} must be a positive number")
    return normalized


def _flatten_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept CLI-shaped nested sections or the request's flat raw form."""

    raw: dict[str, Any] = {}
    for key, value in (overrides or {}).items():
        if isinstance(value, Mapping) and "." not in key:
            for child_key, child_value in value.items():
                raw[f"{key}.{child_key}"] = copy.deepcopy(child_value)
        else:
            raw[key] = copy.deepcopy(value)
    return raw


def _role_sizing(config: Mapping[str, Any], role: str) -> tuple[RoleSizing, int]:
    prefix = "" if role == "agg" else f"{role}_"

    tp = _positive_int(config.get(f"{prefix}tp"), path=f"candidate.config.{prefix}tp")
    pp = _positive_int(config.get(f"{prefix}pp"), path=f"candidate.config.{prefix}pp")
    dp = _positive_int(
        config.get(f"{prefix}attention_dp"),
        path=f"candidate.config.{prefix}attention_dp",
    )
    moe_tp = _positive_int(
        config.get(f"{prefix}moe_tp"),
        path=f"candidate.config.{prefix}moe_tp",
    )
    moe_ep = _positive_int(
        config.get(f"{prefix}moe_ep"),
        path=f"candidate.config.{prefix}moe_ep",
    )
    workers_key = "replicas" if role == "agg" else f"{role}_replicas"
    workers = _positive_int(
        config.get(workers_key), path=f"candidate.config.{workers_key}"
    )

    max_batch_size = _positive_int(
        config.get(f"{role}_max_num_seqs"),
        path=f"candidate.config.{role}_max_num_seqs",
    )
    max_num_tokens = _positive_int(
        config.get(f"{role}_max_num_batched_tokens"),
        path=f"candidate.config.{role}_max_num_batched_tokens",
    )
    tokens_per_block = _positive_int(
        config.get(f"{role}_block_size"),
        path=f"candidate.config.{role}_block_size",
    )
    memory_fraction = _positive_float(
        config.get(f"{role}_gpu_memory_utilization"),
        path=f"candidate.config.{role}_gpu_memory_utilization",
    )
    if memory_fraction > 1:
        raise SweeperCandidateError(
            f"candidate.config.{role}_gpu_memory_utilization must be at most 1"
        )

    extra: dict[str, Any] = {
        "gpus_per_worker": tp * pp * dp,
        "max_num_tokens": max_num_tokens,
        "tokens_per_block": tokens_per_block,
        "kv_cache_free_gpu_memory_fraction": memory_fraction,
        "disable_prefix_cache": not bool(config.get(f"{role}_enable_prefix_caching")),
    }
    context_length = config.get("context_length")
    if context_length is not None:
        extra["max_seq_len"] = _positive_int(
            context_length, path="candidate.config.context_length"
        )

    return (
        RoleSizing(
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            moe_tensor_parallel_size=moe_tp,
            moe_expert_parallel_size=moe_ep,
            max_batch_size=max_batch_size,
            extra=extra,
        ),
        workers,
    )


def _map_adapters(config: Mapping[str, Any], raw: dict[str, Any]) -> None:
    adapters = config.get("adapters") or {}
    if not isinstance(adapters, Mapping):
        raise SweeperCandidateError("candidate.config.adapters must be a mapping")

    seen: set[str] = set()
    for provider_name, provider_config in adapters.items():
        if not isinstance(provider_name, str) or not provider_name:
            raise SweeperCandidateError(
                "candidate adapter names must be non-empty strings"
            )
        if not isinstance(provider_config, Mapping):
            raise SweeperCandidateError(
                f"candidate adapter {provider_name!r} must contain a configuration mapping"
            )
        payload = copy.deepcopy(dict(provider_config))
        feature = provider_name.rsplit(".", 1)[-1].replace("-", "_")
        target = {
            "router": "router_config",
            "planner": "planner_config",
            "kvbm": "kvbm_config",
        }.get(feature)
        if target is None:
            if payload:
                raise SweeperCandidateError(
                    f"candidate adapter {provider_name!r} has no generator mapping"
                )
            continue
        if target in seen:
            raise SweeperCandidateError(
                f"multiple candidate adapters map to DynConfig.{target}"
            )
        seen.add(target)
        if feature == "planner":
            planner_enabled = bool(
                payload.get("enable_throughput_scaling")
                or payload.get("enable_load_scaling")
            )
            if not planner_enabled:
                continue
            # Dynamo's provider adds the selected preset name for result
            # reporting. The runtime hook intentionally excludes that field.
            payload.pop("scaling_policy", None)
        raw[f"DynConfig.{target}"] = payload
        if feature == "router":
            raw["DynConfig.enable_router"] = True
            if payload.get("mode") is not None:
                raw["DynConfig.router_mode"] = payload["mode"]


def _resolve_model_facts(model_path: str, config: Mapping[str, Any]) -> ModelFacts:
    try:
        from aiconfigurator.sdk.models import check_is_moe
        from aiconfigurator.sdk.utils import get_model_config_from_model_path

        model_info = get_model_config_from_model_path(model_path)
        is_moe = check_is_moe(model_path, model_info=model_info)
    except Exception as exc:
        raise SweeperCandidateError(
            f"could not resolve generator model facts for {model_path!r}; pass model_facts explicitly"
        ) from exc
    return ModelFacts(
        is_moe=bool(is_moe),
        nextn=config.get("aic_nextn"),
        architecture=model_info.get("architecture"),
    )


def from_sweeper_candidate(
    candidate: Any,
    *,
    workload: Any | None = None,
    deployment_target: str = "dynamo-j2",
    output_dir: str | None = None,
    environment_profile: str | None = None,
    generator_overrides: Mapping[str, Any] | None = None,
    model_facts: ModelFacts | None = None,
) -> GeneratorRequest:
    """Build a generator request from one ranked Sweeper candidate.

    ``candidate`` may be the public Pydantic ``Candidate`` or its JSON mapping.
    ``workload`` should be the matching Sweeper workload (or a mapping containing
    ``isl``/``osl``). Cluster-specific values remain generator overrides or an
    environment profile; evaluated engine and adapter choices always win.
    """

    candidate_payload = _as_mapping(candidate, label="candidate")
    config_value = candidate_payload.get("config", candidate_payload)
    config = _as_mapping(config_value, label="candidate.config")
    workload_payload = _as_mapping(workload, label="workload")

    mode = _required_text(config, "deployment_mode")
    if mode not in {"agg", "disagg"}:
        raise SweeperCandidateError(
            f"candidate.config.deployment_mode must be 'agg' or 'disagg', got {mode!r}"
        )
    model_path = _required_text(config, "model_name")
    backend = _required_text(config, "backend")
    backend_version = _required_text(config, "backend_version")
    hardware_sku = _required_text(config, "hardware_sku")

    active_roles = ("agg",) if mode == "agg" else ("prefill", "decode")
    roles: dict[str, RoleSizing] = {}
    workers: dict[str, int] = {}
    expected_gpus = 0
    for role in active_roles:
        sizing, count = _role_sizing(config, role)
        roles[role] = sizing
        workers[role] = count
        expected_gpus += count * int(sizing.extra["gpus_per_worker"])

    used_gpus = candidate_payload.get("used_gpus", config.get("used_gpus"))
    if used_gpus is not None:
        normalized_used_gpus = _positive_int(used_gpus, path="candidate.used_gpus")
        if normalized_used_gpus != expected_gpus:
            raise SweeperCandidateError(
                "candidate GPU count cannot be lowered losslessly: "
                f"used_gpus={normalized_used_gpus}, topology={expected_gpus}"
            )

    raw = _flatten_overrides(generator_overrides)
    isl = workload_payload.get("isl", raw.get("SlaConfig.isl"))
    osl = workload_payload.get("osl", raw.get("SlaConfig.osl"))
    if isl is None or osl is None:
        raise SweeperCandidateError(
            "the matching workload must provide isl and osl (or set SlaConfig.isl/osl overrides)"
        )
    isl = _positive_int(isl, path="workload.isl")
    osl = _positive_int(osl, path="workload.osl")

    # Evaluated facts take precedence over deployment-environment overrides.
    raw["backend"] = backend
    raw["rule"] = "benchmark"
    raw["preserve_engine_limits"] = True
    raw["NodeConfig.system_name"] = hardware_sku
    raw["SlaConfig.isl"] = isl
    raw["SlaConfig.osl"] = osl
    concurrency = config.get("concurrency")
    if concurrency is not None:
        raw["BenchConfig.estimated_concurrency"] = _positive_int(
            concurrency, path="candidate.config.concurrency"
        )
    _map_adapters(config, raw)

    resolved_model_facts = model_facts or _resolve_model_facts(model_path, config)
    candidate_nextn = config.get("aic_nextn")
    if candidate_nextn is not None and resolved_model_facts.nextn != candidate_nextn:
        resolved_model_facts = ModelFacts(
            is_moe=resolved_model_facts.is_moe,
            nextn=_positive_int(candidate_nextn, path="candidate.config.aic_nextn"),
            prefix=resolved_model_facts.prefix,
            architecture=resolved_model_facts.architecture,
            extra=copy.deepcopy(resolved_model_facts.extra),
        )
    for key, value in resolved_model_facts.extra.items():
        raw[f"ModelConfig.{key}"] = copy.deepcopy(value)
    for key in ("is_moe", "nextn", "prefix", "architecture"):
        value = getattr(resolved_model_facts, key)
        if value is not None:
            raw[f"ModelConfig.{key}"] = value

    request = GeneratorRequest(
        model=ModelSpec(model_path=model_path),
        backend=BackendSpec(name=backend, generated_config_version=backend_version),
        topology=Topology(mode=mode, roles=roles, workers=workers),
        sla=SlaSpec(isl=isl, osl=osl),
        platform=Platform(
            hardware_profile=hardware_sku,
            environment_profile=environment_profile,
        ),
        emit=EmitTargets(
            output_dir=output_dir,
            deployment_target=deployment_target,
        ),
        model_facts=resolved_model_facts,
        overrides=Overrides(raw=raw),
    )
    errors = request.validate()
    if errors:
        raise SweeperCandidateError("invalid generator request: " + "; ".join(errors))
    return request


__all__ = ["SweeperCandidateError", "from_sweeper_candidate"]
