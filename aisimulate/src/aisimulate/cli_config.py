# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration lowering for the public ``aisimulate`` command line."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .sweeper.config import SmartSearchConfig, SweepConfig

_BACKENDS = ["vllm", "sglang", "trtllm"]
_PREDICT_LIST_FIELDS = {
    "agg": (
        "agg_max_num_batched_tokens",
        "agg_max_num_seqs",
    ),
    "disagg": (
        "prefill_max_num_batched_tokens",
        "prefill_max_num_seqs",
        "decode_max_num_batched_tokens",
        "decode_max_num_seqs",
    ),
}


class CLIConfigError(ValueError):
    """CLI inputs cannot be lowered to one valid simulation request."""


def _load_yaml_mapping(path: str, *, label: str) -> dict[str, Any]:
    try:
        data = yaml.safe_load(Path(path).read_text())
    except OSError as exc:
        raise CLIConfigError(f"could not read {label} {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise CLIConfigError(f"malformed YAML in {label} {path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise CLIConfigError(f"{label} {path} must contain a YAML mapping")
    return dict(data)


def _workload_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.traffic:
        data = _load_yaml_mapping(args.traffic, label="traffic file")
        workload = data.get("workload", data)
        if not isinstance(workload, Mapping):
            raise CLIConfigError(
                f"traffic file {args.traffic} field 'workload' must be a mapping"
            )
        return dict(workload)

    inline_fields = {
        "isl": args.isl,
        "osl": args.osl,
        "concurrency": args.concurrency,
        "request_rate": args.request_rate,
        "kv_load_ratio": args.kv_load_ratio,
        "num_request_ratio": args.num_request_ratio,
    }
    if not any(value is not None for value in inline_fields.values()):
        return None
    if args.isl is None or args.osl is None:
        raise CLIConfigError("inline synthetic traffic requires both --isl and --osl")
    selected_loads = [
        name
        for name in ("concurrency", "request_rate", "kv_load_ratio")
        if inline_fields[name] is not None
    ]
    if len(selected_loads) > 1:
        raise CLIConfigError(
            "inline traffic accepts exactly one of --concurrency, --request-rate, "
            "or --kv-load-ratio"
        )
    if not selected_loads:
        inline_fields["concurrency"] = 1
    if inline_fields["num_request_ratio"] is None:
        inline_fields["num_request_ratio"] = 10.0
    return {name: value for name, value in inline_fields.items() if value is not None}


def _goal_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    sla = {
        "ttft_ms": args.sla_ttft_ms,
        "itl_ms": args.sla_itl_ms,
        "e2e_ms": args.sla_e2e_ms,
    }
    concrete_sla = {name: value for name, value in sla.items() if value is not None}
    if args.objective is None and not concrete_sla:
        return None
    goal: dict[str, Any] = {"target": args.objective or "goodput"}
    if concrete_sla:
        if args.strict_sla:
            concrete_sla["strict"] = True
        goal["sla"] = concrete_sla
    return goal


def _apply_strict_sla_to_config(raw: dict[str, Any], args: argparse.Namespace) -> None:
    if not args.strict_sla:
        return
    goal = raw.get("goal")
    if not isinstance(goal, dict):
        raise CLIConfigError(
            "--strict-sla requires SLA flags or a goal.sla mapping in --config"
        )
    sla = goal.get("sla")
    if not isinstance(sla, dict):
        raise CLIConfigError(
            "--strict-sla requires SLA flags or a goal.sla mapping in --config"
        )
    sla["strict"] = True


def _set_if_not_none(target: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        target[name] = value


def _merge_shared_overrides(
    raw: dict[str, Any], args: argparse.Namespace, *, recommend: bool
) -> None:
    search_space = raw.setdefault("search_space", {})
    if not isinstance(search_space, dict):
        raise CLIConfigError("config field 'search_space' must be a mapping")

    _set_if_not_none(search_space, "model_name", args.model)
    _set_if_not_none(search_space, "hardware_sku", args.system)
    if args.backend is not None:
        backends = _BACKENDS if args.backend == "auto" else [args.backend]
        search_space["backend"] = backends
    if args.deployment_mode:
        modes = (
            args.deployment_mode
            if isinstance(args.deployment_mode, list)
            else [args.deployment_mode]
        )
        search_space["deployment_mode"] = modes
    if recommend:
        _set_if_not_none(search_space, "gpu_budget", args.total_gpus)

    workload = _workload_from_args(args)
    if workload is not None:
        if "workload" in raw:
            raise CLIConfigError(
                "traffic is specified in both --traffic/inline flags and --config"
            )
        raw["workload"] = workload

    goal = _goal_from_args(args)
    if goal is not None:
        if "goal" in raw:
            raise CLIConfigError(
                "the optimization goal is specified in both flags and --config"
            )
        raw["goal"] = goal
    elif args.strict_sla:
        _apply_strict_sla_to_config(raw, args)


def _validate(raw: dict[str, Any]) -> SmartSearchConfig:
    try:
        return SmartSearchConfig.model_validate(raw)
    except ValidationError as exc:
        raise CLIConfigError(f"invalid simulation config: {exc}") from exc


def _require_fields(
    values: Mapping[str, Any], names: tuple[str, ...], *, command: str
) -> None:
    missing = [name for name in names if values.get(name) is None]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise CLIConfigError(f"{command} without --config requires {flags}")


def _require_singleton(value: Any, *, path: str) -> None:
    if not isinstance(value, list) or len(value) != 1:
        raise CLIConfigError(
            f"predict requires exactly one resolved value at {path}; got {value!r}"
        )


def _require_adapter_singletons(value: Any, *, path: str) -> None:
    if isinstance(value, dict):
        for name, item in value.items():
            _require_adapter_singletons(item, path=f"{path}.{name}")
    elif isinstance(value, list) and len(value) != 1:
        raise CLIConfigError(
            f"predict requires singleton adapter choices at {path}; got {value!r}"
        )


def _pin_predict(config: SmartSearchConfig) -> SmartSearchConfig:
    search_space = config.search_space
    _require_singleton(
        search_space.deployment_mode, path="search_space.deployment_mode"
    )
    _require_singleton(search_space.backend, path="search_space.backend")
    _require_singleton(
        search_space.parallel_configs, path="search_space.parallel_configs"
    )
    mode = search_space.deployment_mode[0]
    for field_name in _PREDICT_LIST_FIELDS[mode]:
        _require_singleton(
            getattr(search_space, field_name), path=f"search_space.{field_name}"
        )
    for name, adapter in config.adapters.items():
        _require_adapter_singletons(
            adapter.search_space, path=f"adapters.{name}.search_space"
        )
    if config.goal.is_pareto:
        raise CLIConfigError("predict does not accept a Pareto optimization goal")
    return config.model_copy(
        update={
            "sweep": SweepConfig(
                max_rounds=1,
                parallel_evals=1,
                candidates_per_round=1,
                max_eval_seconds=config.sweep.max_eval_seconds,
            )
        }
    )


def build_predict_config(args: argparse.Namespace) -> SmartSearchConfig:
    """Build a one-candidate SmartSearchConfig from predict flags or YAML."""

    raw = _load_yaml_mapping(args.config, label="config") if args.config else {}
    _merge_shared_overrides(raw, args, recommend=False)
    search_space = raw.setdefault("search_space", {})

    if not args.config:
        _require_fields(
            vars(args), ("model", "system", "backend", "tp_size"), command="predict"
        )
    mode = search_space.get("deployment_mode", ["agg"])
    if isinstance(mode, list):
        selected_mode = mode[0] if len(mode) == 1 else None
    else:
        selected_mode = mode
    if args.tp_size is not None:
        if selected_mode not in (None, "agg"):
            raise CLIConfigError(
                "--tp-size/--replicas currently describe an aggregated deployment; "
                "pin disaggregated prefill/decode shapes in --config"
            )
        replicas = args.replicas or 1
        search_space["deployment_mode"] = ["agg"]
        search_space["parallel_configs"] = [{"tp": args.tp_size, "replicas": replicas}]
        search_space["gpu_budget"] = args.tp_size * replicas
    elif not args.config:
        raise CLIConfigError("predict without --config requires --tp-size")

    if not args.config:
        search_space.setdefault(
            "agg_max_num_batched_tokens", [args.max_num_batched_tokens or 8192]
        )
        search_space.setdefault("agg_max_num_seqs", [args.max_num_seqs or 256])
    else:
        if args.max_num_batched_tokens is not None:
            search_space["agg_max_num_batched_tokens"] = [args.max_num_batched_tokens]
        if args.max_num_seqs is not None:
            search_space["agg_max_num_seqs"] = [args.max_num_seqs]

    if "workload" not in raw:
        raise CLIConfigError(
            "predict requires traffic in --config, --traffic, or inline --isl/--osl flags"
        )
    return _pin_predict(_validate(raw))


def build_recommend_config(args: argparse.Namespace) -> SmartSearchConfig:
    """Build a SmartSearchConfig from a search YAML or product-level flags."""

    raw = _load_yaml_mapping(args.config, label="config") if args.config else {}
    _merge_shared_overrides(raw, args, recommend=True)
    search_space = raw.setdefault("search_space", {})

    if not args.config:
        _require_fields(
            vars(args), ("model", "system", "total_gpus"), command="recommend"
        )
        search_space.setdefault("backend", _BACKENDS)
        search_space.setdefault("deployment_mode", ["agg", "disagg"])
        if args.stack == "dynamo":
            raw.setdefault(
                "adapters",
                {
                    "dynamo.planner": {"search_space": {}},
                    "dynamo.router": {"search_space": {}},
                },
            )
    if "workload" not in raw:
        raise CLIConfigError(
            "recommend requires traffic in --config, --traffic, or inline --isl/--osl flags"
        )

    if any(
        value is not None
        for value in (
            args.max_rounds,
            args.parallel_evals,
            args.candidates_per_round,
        )
    ):
        sweep = deepcopy(raw.get("sweep", {}))
        if not isinstance(sweep, dict):
            raise CLIConfigError("config field 'sweep' must be a mapping")
        _set_if_not_none(sweep, "max_rounds", args.max_rounds)
        _set_if_not_none(sweep, "parallel_evals", args.parallel_evals)
        _set_if_not_none(sweep, "candidates_per_round", args.candidates_per_round)
        raw["sweep"] = sweep
    return _validate(raw)
