# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration lowering shared by Dynamo replay SDK integrations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Protocol

from aisimulate.capacity import materialize_aic_num_gpu_blocks

from dynamo.mocker import MockEngineArgs
from dynamo.mocker.args import (
    resolve_planner_profile_data as _resolve_mocker_planner_profile_data,
)


class PlannerProfileDataResult(Protocol):
    npz_path: Path | None


def speculation_api(*, required: bool = True) -> ModuleType | None:
    """Load the paired SD contract without breaking older non-SD installs."""
    try:
        api = import_module("aisimulate.speculation")
        if not all(
            callable(getattr(api, name, None))
            for name in (
                "normalize_speculation_engine_args",
                "speculation_report_metadata",
            )
        ):
            raise ImportError("incomplete AISimulate speculation API")
        return api
    except ImportError as error:
        if not required:
            return None
        raise RuntimeError(
            "Dynamo speculative replay requires the matching AISimulate source "
            "and native bindings. Install container/deps/requirements.aisimulate.txt "
            "and rebuild ai-dynamo-runtime with --features ais-forward-pass; "
            "the version label aisimulate==0.13.0 alone does not identify this API."
        ) from error


def _speculation_configs(rank: Mapping[str, Any]) -> tuple[Any, ...]:
    timing = rank.get("timing_model")
    timing_config = timing.get("config") if isinstance(timing, Mapping) else None
    canonical = rank.get("ais_perf_config")
    return (
        rank.get("speculation"),
        timing_config.get("speculation")
        if isinstance(timing_config, Mapping)
        else None,
        canonical.get("speculation") if isinstance(canonical, Mapping) else None,
    )


def has_speculative_decoding(payload: Mapping[str, Any]) -> bool:
    rank = payload.get("rank", payload)
    if not isinstance(rank, Mapping):
        return False
    if any(chosen is not None for chosen in _speculation_configs(rank)):
        return True
    timing = rank.get("timing_model")
    config = timing.get("config") if isinstance(timing, Mapping) else None
    return any(
        item.get(name) not in (None, 0)
        for item in (rank, config, rank.get("ais_perf_config"))
        if isinstance(item, Mapping)
        for name in ("nextn", "aic_nextn", "ais_nextn")
    )


def validate_speculation_payload(payload: Mapping[str, Any]) -> None:
    """Apply Dynamo's method boundary before either replay path executes."""
    rank = payload.get("rank", payload)
    if not isinstance(rank, Mapping):
        return
    for field in ("nextn", "nextn_accept_rates", "mtp_seed"):
        native, upstream = "ais_" + field, "aic_" + field
        if native in rank and upstream in rank and rank[native] != rank[upstream]:
            raise ValueError(f"{native} conflicts with {upstream}")
    if rank.get("speculation") is not None and rank.get("ais_perf_config") is not None:
        raise ValueError(
            "speculation cannot be combined with ais_perf_config; use one canonical "
            "timing_model configuration with the public speculation controls"
        )
    choices = _speculation_configs(rank)
    for chosen in choices:
        if isinstance(chosen, Mapping) and chosen.get("kind") != "mtp":
            raise ValueError(
                "Dynamo replay supports only MTP; it does not support ngram draft scheduling"
            )
    if rank.get("speculation") is None and any(
        isinstance(chosen, Mapping) and chosen.get("kind") == "mtp"
        for chosen in choices[1:]
    ):
        has_mean = any(
            rank.get(name) is not None
            for name in ("aic_nextn_accepted", "nextn_accepted")
        )
        has_rates = any(
            isinstance(rank.get(name), str) and rank[name].strip()
            for name in (
                "aic_nextn_accept_rates",
                "nextn_accept_rates",
                "ais_nextn_accept_rates",
            )
        )
        if not (has_mean or has_rates):
            raise ValueError(
                "canonical MTP requires explicit acceptance; configure an expected "
                "accepted-token count or conditional acceptance rates"
            )


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


def reconcile_mtp_timing(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Align legacy scheduler depth with AIC cost identity before construction."""
    raw = dict(payload)
    timing = raw.get("timing_model")
    depth = raw.get("aic_nextn", raw.get("nextn"))
    if (
        raw.get("speculation") is None
        and (depth is None or type(depth) is int and depth == 0)
        and isinstance(timing, Mapping)
        and timing.get("provider") == "aic"
    ):
        config = timing.get("config", {})
        chosen = config.get("speculation")
        configured_depth = (
            chosen.get("params", {}).get("num_speculative_tokens")
            if isinstance(chosen, Mapping)
            else config.get("nextn")
        )
        if type(configured_depth) is int and configured_depth > 0:
            depth = configured_depth
            raw.pop("nextn", None)
            raw["aic_nextn"] = depth
    if (
        type(depth) is int
        and depth > 0
        and isinstance(timing, Mapping)
        and timing.get("provider") == "aic"
    ):
        config = dict(timing.get("config", {}))
        chosen = config.get("speculation")
        if isinstance(chosen, Mapping):
            if chosen.get("params", {}).get("num_speculative_tokens") != depth:
                raise ValueError(
                    "speculative depth conflicts with timing_model.config.speculation"
                )
        else:
            if config.get("nextn") not in (None, 0, depth):
                raise ValueError(
                    "speculative depth conflicts with timing_model.config.nextn"
                )
            config["nextn"] = depth
        raw["timing_model"] = {**timing, "config": config}
    return raw


def lower_upstream_engine_args(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Consume AISimulate's runner wire protocol at the Dynamo boundary."""
    validate_speculation_payload(payload)
    role = payload.get("worker_type") or (
        "prefill"
        if payload.get("is_prefill")
        else "decode"
        if payload.get("is_decode")
        else "aggregated"
    )
    raw = dict(payload)
    canonical = raw.pop("ais_perf_config", None)
    if canonical is not None:
        if raw.get("timing_model") is not None:
            raise ValueError("ais_perf_config cannot be combined with timing_model")
        raw["timing_model"] = {
            "type": "external",
            "provider": "aic",
            "config": canonical,
        }
    for field in ("nextn", "nextn_accept_rates", "mtp_seed"):
        native, upstream = "ais_" + field, "aic_" + field
        if native in raw:
            raw.setdefault(upstream, raw.pop(native))
    for name in ("aic_nextn", "nextn"):
        if name in raw and raw[name] is not None and type(raw[name]) is not int:
            raise ValueError(f"{name} must be an integer in 0..=5")
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
            "config": canonical_upstream_config(identity, worker_type=role),
        }
    raw = reconcile_mtp_timing(raw)
    if has_speculative_decoding(raw) or any(
        raw.get(name) is not None
        for name in (
            "aic_nextn_accepted",
            "nextn_accepted",
            "aic_nextn_accept_rates",
            "nextn_accept_rates",
        )
    ):
        raw = speculation_api().normalize_speculation_engine_args(raw, role=role)
    else:
        # Explicit zero is the supported SD-off spelling, including when an
        # older AISimulate installation has no public speculation helper.
        for name in ("speculation", "aic_nextn", "nextn"):
            if raw.get(name) in (None, 0):
                raw.pop(name, None)
    raw = reconcile_mtp_timing(raw)
    raw = _materialize_capacity(raw)
    raw.pop("cuda_graph_reserved_bytes", None)
    if identity.get("attention_dp_size") is not None:
        raw.setdefault("dp_size", identity["attention_dp_size"])
    if identity.get("tp_size") is not None:
        raw.setdefault("tensor_parallel_size", identity["tp_size"])
    for name in tuple(raw):
        if name.startswith("aic_"):
            value = raw.pop(name)
            if name in {"aic_nextn_accept_rates", "aic_mtp_seed"}:
                raw["ais_" + name[4:]] = value
            elif name == "aic_nextn":
                raw["ais_nextn"] = value
    return raw


def native_engine_args_payload(
    lowered: Mapping[str, Any], *, authored: Mapping[str, Any]
) -> dict[str, Any]:
    """Preserve capacity provenance through the native constructor boundary."""
    native = dict(lowered)
    native.pop("num_gpu_blocks_is_explicit", None)
    if has_speculative_decoding(lowered) and not authored.get(
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
