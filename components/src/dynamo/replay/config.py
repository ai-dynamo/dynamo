# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration lowering shared by Dynamo replay SDK integrations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

from aisimulate.aic import materialize_aic_num_gpu_blocks

from dynamo._internal.ais import resolve_backend_version
from dynamo.mocker import MockEngineArgs
from dynamo.mocker.args import (
    resolve_planner_profile_data as _resolve_mocker_planner_profile_data,
)


class PlannerProfileDataResult(Protocol):
    npz_path: Path | None


def resolve_ais_num_gpu_blocks(raw: dict[str, Any]) -> None:
    """Materialize capacity using AISimulate's upstream wire adapter."""

    # The upstream compatibility helper still names its flat inputs aic_*.
    # Translate only at this boundary; Dynamo emits ais_* names.
    for name in list(raw):
        if name.startswith("ais_") and name != "ais_perf_config":
            legacy = "aic_" + name[4:]
            if legacy in raw:
                raise ValueError(f"cannot combine {name} with {legacy}")
            raw[legacy] = raw.pop(name)
    canonical = raw.get("ais_perf_config")
    if canonical is not None:
        if "timing_model" in raw:
            raise ValueError("ais_perf_config cannot be combined with timing_model")
        if not isinstance(canonical, Mapping):
            raise TypeError("ais_perf_config must be a mapping")
        canonical = dict(canonical)
        canonical.setdefault("estimation_mode", "auto")
        canonical.setdefault("fallback_policy", "deny")
        raw["timing_model"] = {
            "type": "external",
            "provider": "aic",
            "config": canonical,
        }
        del raw["ais_perf_config"]

    if (
        raw.get("aic_backend") is not None
        or raw.get("aic_attention_dp_size") is not None
    ):
        raw["aic_backend_version"] = resolve_backend_version(
            raw.get("aic_backend") or "vllm", raw.get("aic_backend_version")
        )
    timing = raw.get("timing_model")
    if (
        isinstance(timing, dict)
        and timing.get("type") == "external"
        and timing.get("provider") == "ais"
    ):
        raw["timing_model"] = {**timing, "provider": "aic"}
    timing = raw.get("timing_model")
    if (
        isinstance(timing, dict)
        and timing.get("type") == "external"
        and timing.get("provider") == "aic"
    ):
        config = timing.get("config")
        if isinstance(config, dict) and "model" in config:
            config = {"estimation_mode": "auto", "fallback_policy": "deny", **config}
            raw["timing_model"] = {**timing, "config": config}
    lowered = materialize_aic_num_gpu_blocks(raw)
    timing = lowered.get("timing_model")
    if (
        isinstance(timing, dict)
        and timing.get("type") == "external"
        and timing.get("provider") == "aic"
    ):
        capacity_fields = {
            "gpu_memory_utilization",
            "mem_fraction_static",
            "free_gpu_memory_fraction",
            "cuda_graph_reserved_bytes",
        }
        lowered["timing_model"] = {
            **timing,
            "config": {
                name: value
                for name, value in timing["config"].items()
                if name not in capacity_fields
            },
        }
    raw.clear()
    raw.update(
        {
            ("ais_" + name[4:] if name.startswith("aic_") else name): value
            for name, value in lowered.items()
        }
    )


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
    return MockEngineArgs.from_json(json.dumps(raw))


# Deprecated Python SDK name.
resolve_aic_num_gpu_blocks = resolve_ais_num_gpu_blocks
