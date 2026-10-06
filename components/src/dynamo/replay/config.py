# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration lowering shared by Dynamo replay SDK integrations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from aisimulate.capacity import materialize_aic_num_gpu_blocks
from aisimulate.config.engine import SpeculationConfig
from pydantic import TypeAdapter

from dynamo.mocker import MockEngineArgs
from dynamo.mocker.args import (
    resolve_planner_profile_data as _resolve_mocker_planner_profile_data,
)


class PlannerProfileDataResult(Protocol):
    npz_path: Path | None


def canonical_upstream_config(
    config: Mapping[str, Any], *, worker_type: str
) -> dict[str, Any]:
    """Translate the upstream Replay metadata protocol to canonical AIS identity."""
    from aisimulate_core.sdk import ForwardPassPerfModelConfig

    if "model" in config:
        if config.get("worker_type") != worker_type:
            raise ValueError(
                f"AIS metadata worker_type must match the {worker_type} engine role"
            )
        return dict(config)
    fields = {
        "model_path": "model",
        "system": "system",
        "backend": "backend",
        "backend_version": "backend_version",
        "tp_size": "tp",
        "pp_size": "pp",
        "attention_dp_size": "attention_dp",
        "moe_tp_size": "moe_tp_size",
        "moe_ep_size": "moe_ep_size",
        "nextn": "nextn",
        "speculation": "speculation",
        "gemm_dtype": "gemm_quant_mode",
        "moe_dtype": "moe_quant_mode",
        "fmha_dtype": "fmha_quant_mode",
        "kv_cache_dtype": "kvcache_quant_mode",
        "comm_dtype": "comm_quant_mode",
    }
    payload = {
        target: config[source]
        for source, target in fields.items()
        if config.get(source) is not None
    }
    payload["worker_type"] = worker_type
    if config.get("forward_model") is not None:
        payload["estimation_mode"] = {
            "op_level": "op_level",
            "fpm": "fpm_interpolation",
        }[config["forward_model"]]
    return ForwardPassPerfModelConfig(**payload).to_dict()


def _materialize_capacity(raw: dict[str, Any]) -> dict[str, Any]:
    lowered = materialize_aic_num_gpu_blocks(raw)
    timing = lowered.get("timing_model")
    if isinstance(timing, dict) and timing.get("type") == "external":
        if timing.get("provider") != "aic":
            raise ValueError("unsupported upstream timing provider")
        capacity_fields = {
            "gpu_memory_utilization",
            "mem_fraction_static",
            "free_gpu_memory_fraction",
            "cuda_graph_reserved_bytes",
        }
        lowered["ais_perf_config"] = {
            key: value
            for key, value in timing["config"].items()
            if key not in capacity_fields
        }
        del lowered["timing_model"]
    return lowered


def resolve_ais_num_gpu_blocks(raw: dict[str, Any]) -> None:
    """Resolve a canonical Dynamo config through the upstream capacity adapter."""
    if any(name.startswith("aic_") for name in raw):
        raise ValueError("AIC config fields were removed; use ais_perf_config")
    canonical = raw.get("ais_perf_config")
    if canonical is None:
        return
    if "timing_model" in raw:
        raise ValueError("ais_perf_config cannot be combined with timing_model")
    if not isinstance(canonical, Mapping):
        raise TypeError("ais_perf_config must be a mapping")
    upstream = dict(raw)
    upstream.pop("ais_perf_config")
    upstream["timing_model"] = {
        "type": "external",
        "provider": "aic",
        "config": {"estimation_mode": "auto", "fallback_policy": "deny", **canonical},
    }
    lowered = _materialize_capacity(upstream)
    raw.clear()
    raw.update(lowered)


def lower_upstream_engine_args(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Consume AISimulate's runner wire protocol at the Dynamo boundary."""
    raw = dict(payload)
    timing = raw.get("timing_model")
    has_custom_timing = isinstance(timing, dict) and timing.get("type") in {
        "fixed",
        "polynomial",
    }
    identity = {
        name[4:]: value for name, value in raw.items() if name.startswith("aic_")
    }
    if has_custom_timing:
        for name in (
            "aic_backend",
            "aic_backend_version",
            "aic_system",
            "aic_model_path",
        ):
            raw.pop(name, None)
    if not has_custom_timing and timing is None and identity.get("backend") is not None:
        raw["timing_model"] = {
            "type": "external",
            "provider": "aic",
            "config": canonical_upstream_config(
                identity, worker_type=raw.get("worker_type", "aggregated")
            ),
        }
    chosen = raw.pop("speculation", None)
    if chosen is not None:
        if raw.get("ais_perf_config") is not None:
            raise ValueError("speculation cannot be combined with ais_perf_config")
        if any(
            raw.get(name) not in (None, 0)
            for name in (
                "aic_nextn",
                "ais_nextn",
                "aic_verify_width",
                "ais_verify_width",
            )
        ) or any(
            raw.get(name) is not None
            for name in (
                "aic_nextn_accepted",
                "aic_nextn_accept_rates",
                "ais_nextn_accept_rates",
                "aic_mtp_seed",
                "ais_mtp_seed",
            )
        ):
            raise ValueError(
                "speculation cannot be combined with legacy speculative fields"
            )
        config = TypeAdapter(SpeculationConfig).validate_python(chosen)
        timing = raw.get("timing_model") or {}
        if timing.get("type") != "external" or timing.get("provider") != "aic":
            raise ValueError("speculation requires an AIS timing configuration")
        cost = dict(timing.get("config", {}))
        if cost.get("nextn") not in (None, 0) or cost.get("speculation") not in (
            None,
            config.cost_config(),
        ):
            raise ValueError("speculation conflicts with timing_model configuration")
        raw["timing_model"] = {
            **timing,
            "config": {**cost, "speculation": config.cost_config()},
        }
        raw.update(
            aic_nextn=config.max_accepted_draft_tokens,
            aic_verify_width=config.verify_width,
            aic_nextn_accept_rates=",".join(
                format(rate, ".17g") for rate in config.acceptance_rates
            ),
            aic_mtp_seed=config.seed,
        )
    expected = raw.pop("aic_nextn_accepted", None)
    if expected is not None:
        from aisimulate.runner import _accept_rates_for_expected

        depth = raw.get("aic_nextn")
        if type(depth) is not int or depth <= 0:
            raise ValueError("aic_nextn_accepted requires a positive aic_nextn")
        if (
            raw.get("aic_nextn_accept_rates") is not None
            or raw.get("ais_nextn_accept_rates") is not None
        ):
            raise ValueError("cannot set both aic_nextn_accepted and acceptance rates")
        raw["aic_nextn_accept_rates"] = _accept_rates_for_expected(
            depth, expected, role=raw.get("worker_type", "aggregated")
        )
    raw = _materialize_capacity(raw)
    raw.pop("cuda_graph_reserved_bytes", None)
    if identity.get("attention_dp_size") is not None:
        raw.setdefault("dp_size", identity["attention_dp_size"])
    if identity.get("tp_size") is not None:
        raw.setdefault("tensor_parallel_size", identity["tp_size"])
    for name in tuple(raw):
        if name.startswith("aic_"):
            value = raw.pop(name)
            if name in {"aic_nextn_accept_rates", "aic_mtp_seed", "aic_verify_width"}:
                raw["ais_" + name[4:]] = value
            elif name == "aic_nextn" and (
                chosen is not None or "ais_perf_config" not in raw
            ):
                raw["ais_nextn"] = value
    return raw


def native_engine_args_payload(
    lowered: Mapping[str, Any], *, authored: Mapping[str, Any]
) -> dict[str, Any]:
    """Preserve capacity provenance through the native constructor boundary."""
    native = dict(lowered)
    native.pop("num_gpu_blocks_is_explicit", None)
    cost = lowered.get("ais_perf_config") or {}
    speculative = (
        lowered.get("ais_nextn") or cost.get("nextn") or cost.get("speculation")
    )
    if speculative and not authored.get(
        "num_gpu_blocks_is_explicit", bool(authored.get("num_gpu_blocks"))
    ):
        # The native loader may discover an Agentic trace after Python's
        # preflight. Leave inferred capacity unauthored for that late gate;
        # ordinary replay materializes the same AIC capacity natively.
        native.pop("num_gpu_blocks", None)
    return native


def resolve_planner_profile_data(
    planner_profile_data: Path | None,
) -> PlannerProfileDataResult:
    if planner_profile_data is None:
        return SimpleNamespace(npz_path=None)
    if planner_profile_data.suffix == ".npz":
        return SimpleNamespace(npz_path=planner_profile_data)
    return _resolve_mocker_planner_profile_data(planner_profile_data)


def load_engine_args(
    raw_args: str | Mapping[str, Any] | None,
) -> MockEngineArgs | None:
    """Lower JSON or mapping engine arguments to ``MockEngineArgs``."""

    if raw_args is None:
        return None
    raw = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
    if not isinstance(raw, dict):
        raise TypeError("engine arguments must contain a JSON object")
    authored = dict(raw)
    worker_type = raw.pop("worker_type", None)
    if worker_type is not None:
        if "is_prefill" in raw or "is_decode" in raw:
            raise ValueError(
                "worker_type cannot be combined with is_prefill or is_decode"
            )
        if worker_type == "prefill":
            raw["is_prefill"] = True
        elif worker_type == "decode":
            raw["is_decode"] = True
        elif worker_type != "aggregated":
            raise ValueError("worker_type must be aggregated, prefill, or decode")
    if "planner_profile_data" in raw:
        profile = raw["planner_profile_data"]
        if profile is None:
            del raw["planner_profile_data"]
        else:
            result = resolve_planner_profile_data(Path(profile))
            if result.npz_path is not None:
                raw["planner_profile_data"] = str(result.npz_path)
            else:
                del raw["planner_profile_data"]
    resolve_ais_num_gpu_blocks(raw)
    return MockEngineArgs.from_json(
        json.dumps(native_engine_args_payload(raw, authored=authored))
    )
