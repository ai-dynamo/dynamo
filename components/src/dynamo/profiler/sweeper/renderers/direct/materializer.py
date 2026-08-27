# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Map a Sweeper result directly onto Dynamo's v1 DGD templates.

The direct renderer consumes the plain ``Candidate.config`` mapping and reuses
the existing ``CONFIG_MODIFIERS`` implementation without invoking AIC's
generator. Keeping this adapter independent of Sweeper orchestration makes it
usable by the standalone CLI and straightforward to compare with the AIC
renderer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dynamo.planner.config.defaults import SubComponentType
from dynamo.profiler.utils.config import update_image
from dynamo.profiler.utils.config_modifiers import CONFIG_MODIFIERS

_STRATEGY_SETTERS = {
    "tp": "set_config_tp_size",
    "tep": "set_config_tep_size",
    "dep": "set_config_dep_size",
}

# Candidate.config's key names for one worker's shape/kv-cache/scheduler/
# replica/attention-dp fields, by deployment mode. Confirmed against
# aisimulate.sweeper.sample.unroll_sample/_unroll_parallel: agg mode uses a
# mix of bare keys (tp, replicas, strategy, attention_dp, moe_tp, moe_ep) and
# "agg_"-prefixed keys (agg_block_size, agg_max_num_seqs, ...) -- NOT a
# uniform prefix. Disagg mode uses a uniform "{role}_" prefix for every one
# of these fields instead (prefill_tp, decode_replicas, ...), with role in
# ("prefill", "decode"). model_name is shared across roles either way.
_AGG_KEYS = {
    "strategy": "strategy",
    "tp": "tp",
    "moe_tp": "moe_tp",
    "moe_ep": "moe_ep",
    "block_size": "agg_block_size",
    "gpu_memory_utilization": "agg_gpu_memory_utilization",
    "enable_prefix_caching": "agg_enable_prefix_caching",
    "max_num_seqs": "agg_max_num_seqs",
    "max_num_batched_tokens": "agg_max_num_batched_tokens",
    "replicas": "replicas",
    "attention_dp": "attention_dp",
}


def _role_keys(role: str) -> dict[str, str]:
    return {
        "strategy": f"{role}_strategy",
        "tp": f"{role}_tp",
        "moe_tp": f"{role}_moe_tp",
        "moe_ep": f"{role}_moe_ep",
        "block_size": f"{role}_block_size",
        "gpu_memory_utilization": f"{role}_gpu_memory_utilization",
        "enable_prefix_caching": f"{role}_enable_prefix_caching",
        "max_num_seqs": f"{role}_max_num_seqs",
        "max_num_batched_tokens": f"{role}_max_num_batched_tokens",
        "replicas": f"{role}_replicas",
        "attention_dp": f"{role}_attention_dp",
    }


# Fields on Candidate.config that describe *what the candidate was evaluated
# against*, not the deployment itself. Confirmed via a real Pareto search
# fixture that kv_load_ratio/concurrency can vary across candidates with an
# otherwise byte-identical deployment shape -- these must never be folded
# into the DGD spec, and must not be silently dropped either.
EVALUATION_CONTEXT_FIELDS = frozenset(
    {
        "concurrency",
        "kv_load_ratio",
        "hardware_sku",
        "gpu_budget",
        "min_gpu_budget",
        "context_length",
        "startup_time",
        "aic_nextn",
    }
)


class MaterializationError(Exception):
    """The direct renderer could not faithfully materialize a DGD."""


@dataclass
class MaterializationResult:
    """A rendered DGD plus evaluation inputs that do not belong in its spec."""

    dgd: dict[str, Any]
    experimental: dict[str, Any] = field(default_factory=dict)


def _materialize_worker(
    config: dict,
    modifier: Any,
    candidate_config: dict[str, Any],
    keys: dict[str, str],
    *,
    backend: str,
    component_type: SubComponentType,
    num_gpus_per_node: int,
) -> dict:
    """Run the full per-worker setter chain for one role (agg's single
    worker, or one side of a disagg pair), using ``keys`` to look up that
    role's field names on ``candidate_config``. Shared by both deployment
    modes so the chain -- and any future fix to it -- only needs to be
    correct in one place.
    """
    strategy = candidate_config[keys["strategy"]]
    setter_name = _STRATEGY_SETTERS.get(strategy)
    if setter_name is None:
        raise MaterializationError(f"unknown parallelism strategy {strategy!r}")
    setter = getattr(modifier, setter_name, None)
    if setter is None:
        raise MaterializationError(
            f"{backend} modifier has no {setter_name} implementation"
        )

    if strategy == "tp":
        config = setter(config, candidate_config[keys["tp"]], component_type)
    else:
        # TEP/DEP setters need the physical node boundary in addition to
        # the parallel shape carried by the Candidate.
        tp_or_ep_key = keys["moe_tp"] if strategy == "tep" else keys["moe_ep"]
        config = setter(
            config,
            candidate_config[tp_or_ep_key],
            num_gpus_per_node=num_gpus_per_node,
            component_type=component_type,
        )

    config = modifier.set_config_kv_cache(
        config,
        block_size=candidate_config[keys["block_size"]],
        memory_fraction=candidate_config[keys["gpu_memory_utilization"]],
        prefix_caching=candidate_config[keys["enable_prefix_caching"]],
        component_type=component_type,
    )
    config = modifier.set_prefill_config(
        config,
        max_batch_size=candidate_config[keys["max_num_seqs"]],
        max_num_tokens=candidate_config[keys["max_num_batched_tokens"]],
        component_type=component_type,
    )
    config = modifier.set_config_model(
        config,
        model_name=candidate_config["model_name"],  # shared across roles
        component_type=component_type,
    )
    config = modifier.set_config_replicas(
        config,
        replicas=candidate_config[keys["replicas"]],
        component_type=component_type,
    )
    # TRT-LLM-only: neither vLLM nor SGLang's base templates reference the
    # extra-engine-args file this derives from, so this is not part of the
    # shared ConfigModifierProtocol -- guard with hasattr rather than a
    # backend name check, matching how other backend-specific capability
    # gaps are already handled in this codebase (e.g. set_config_tep_size
    # raising NotImplementedError for TRT-LLM).
    if hasattr(modifier, "set_config_attention_dp"):
        config = modifier.set_config_attention_dp(
            config,
            attention_dp=candidate_config.get(keys["attention_dp"], 1),
            component_type=component_type,
        )
    return config


def materialize_dgd_from_candidate(
    candidate_config: dict[str, Any],
    *,
    image: str,
    num_gpus_per_node: int = 8,
    component_type: SubComponentType = SubComponentType.DECODE,
) -> MaterializationResult:
    """Materialize one Candidate.config dict into a real DGD.

    ``image`` must already be resolved by the caller (backend + backend
    version -> container image). That lookup is a separate, still-open
    concern -- confirmed absent from build_dgd_config and everywhere else in
    this package -- and deliberately not this function's job.

    ``component_type`` only applies to agg mode (it picks which generic
    worker service load_default_config's template exposes -- matching
    BaseConfigModifier._resolve_component_name's own agg fallback: "try the
    standard DECODE lookup, then fall back to any non-Frontend/Planner
    service"). Disagg mode ignores it: both PREFILL and DECODE are always
    materialized, using their own role-prefixed fields from
    candidate_config (confirmed via aisimulate.sweeper.sample.py: a disagg
    Candidate.config has no bare tp/strategy/replicas/agg_* keys at all --
    everything is prefill_*/decode_*-prefixed).

    Raises MaterializationError for a known, explicit gap (e.g. an
    unsupported strategy/backend combination) rather than letting the
    underlying NotImplementedError/KeyError surface directly, so callers get
    one exception type to catch regardless of which CONFIG_MODIFIERS call
    failed and why.
    """
    backend = candidate_config.get("backend")
    mode = candidate_config.get("deployment_mode")

    modifier = CONFIG_MODIFIERS.get(backend)
    if modifier is None:
        raise MaterializationError(f"no CONFIG_MODIFIERS entry for backend {backend!r}")
    if mode not in ("agg", "disagg"):
        raise MaterializationError(f"unsupported deployment_mode {mode!r}")

    try:
        config = modifier.load_default_config(mode=mode)
        config = update_image(config, image)

        if mode == "agg":
            config = _materialize_worker(
                config,
                modifier,
                candidate_config,
                _AGG_KEYS,
                backend=backend,
                component_type=component_type,
                num_gpus_per_node=num_gpus_per_node,
            )
        else:  # disagg
            for role, role_component_type in (
                ("prefill", SubComponentType.PREFILL),
                ("decode", SubComponentType.DECODE),
            ):
                config = _materialize_worker(
                    config,
                    modifier,
                    candidate_config,
                    _role_keys(role),
                    backend=backend,
                    component_type=role_component_type,
                    num_gpus_per_node=num_gpus_per_node,
                )
    except NotImplementedError as exc:
        # e.g. TRT-LLM's set_config_tep_size: confirmed real and reachable --
        # a real Sweeper-produced Candidate (strategy="tep", backend="trtllm")
        # hits exactly this path. This is the DEP's "materialization failure"
        # case, not a bug in this function.
        raise MaterializationError(
            f"{backend}/{mode} materialization not supported: {exc}"
        ) from exc
    except KeyError as exc:
        raise MaterializationError(
            f"candidate_config missing required field: {exc}"
        ) from exc

    experimental = {
        key: candidate_config[key]
        for key in EVALUATION_CONTEXT_FIELDS
        if key in candidate_config
    }
    return MaterializationResult(dgd=config, experimental=experimental)