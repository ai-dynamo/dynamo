# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unroll a selected sample (one value per searchable knob) into a flat config.

A *selection* assigns one concrete value to each searchable knob (a future Vizier
suggestion), plus the one chosen parallel config (an element of the generated
``parallel_configs``) and the independent load-predictor sweep result.
:func:`unroll_sample` expands the composite knobs and folds in the pinned scalars
to produce a flat, self-contained deployment config (the *selected sample*,
i.e. :class:`dynamo.profiler.spica.config.Candidate`'s ``config``) so downstream code never has
to re-decode preset strings or structured objects.

Composite knobs that get unrolled:

- the chosen **parallel config** -> ``tp``/``pp``/``attention_dp``/``moe_tp``/
  ``moe_ep`` (+ ``replicas``, ``strategy``, ``used_gpus``); ``prefill_*`` +
  ``decode_*`` for disagg (see :mod:`dynamo.profiler.spica.parallel_enum`);
- ``planner_scaling_policy`` -> the four scaling fields (:data:`SCALING_POLICIES`);
- ``planner_fpm_sampling`` / ``planner_load_sensitivity`` -> numeric planner
  fields (:data:`FPM_SAMPLING` / :data:`LOAD_SENSITIVITY`);
- the load predictor -> family + knobs, resolved from the sweep winner for the
  selected policy's throughput interval (:func:`predictor_fields`).

Only the knobs relevant to the chosen ``deployment_mode`` / ``router_mode`` /
planner state are emitted, so the flattened sample is ready for DGD generation.
"""

from __future__ import annotations

from typing import Any

from .config import SearchSpace
from .load_predictor_sweep import LoadPredictorResult, predictor_fields
from .parallel_enum import DisaggParallelConfig, ParallelShape, ReplicaParallelConfig
from .planner import fpm_fields, load_sensitivity_fields, scaling_fields

# Pinned deployment/runtime scalars folded in so the selected sample stands alone.
# The GPU bounds and endpoint floor are search constraints for a static candidate,
# but become live planner limits for a scaling candidate and therefore must survive
# into replay and generated deployment artifacts.
_DEPLOYMENT_PINNED = (
    "model_name",
    "hardware_sku",
    "gpu_budget",
    "min_gpu_budget",
    "min_endpoint",
    "context_length",
    "startup_time",
    "aic_nextn",
)
_KV_MANAGER = (
    "num_g2_blocks",
    "kv_bytes_per_token",
    "bandwidth_g1_to_g2_gbps",
    "bandwidth_g2_to_g1_gbps",
    "offload_batch_size",
)

# engine knobs per branch: searched batching + pinned scalars.
_AGG_SEARCHED = ("agg_max_num_batched_tokens", "agg_max_num_seqs")
_AGG_PINNED = (
    "agg_block_size",
    "agg_gpu_memory_utilization",
    "agg_enable_prefix_caching",
)
_PREFILL_SEARCHED = ("prefill_max_num_batched_tokens", "prefill_max_num_seqs")
_PREFILL_PINNED = (
    "prefill_block_size",
    "prefill_gpu_memory_utilization",
    "prefill_enable_prefix_caching",
)
_DECODE_SEARCHED = ("decode_max_num_batched_tokens", "decode_max_num_seqs")
_DECODE_PINNED = (
    "decode_block_size",
    "decode_gpu_memory_utilization",
    "decode_enable_prefix_caching",
)

# router knobs that only matter under kv_router.
_KV_ROUTER_KNOBS = (
    "overlap_score_credit",
    "prefill_load_scale",
    "host_cache_hit_weight",
    "disk_cache_hit_weight",
    "router_temperature",
)
_ROUTER_ADMISSION = (
    "active_decode_blocks_threshold",
    "active_prefill_tokens_threshold",
    "active_prefill_tokens_threshold_frac",
    "no_admission_control",
)


def _shape_fields(shape: ParallelShape) -> dict[str, Any]:
    return {
        "tp": shape.tp,
        "pp": shape.pp,
        "attention_dp": shape.dp,
        "moe_tp": shape.moe_tp,
        "moe_ep": shape.moe_ep,
        "strategy": shape.strategy,
    }


def _unroll_parallel(
    deployment_mode: str, parallel_config: ReplicaParallelConfig | DisaggParallelConfig
) -> dict[str, Any]:
    if deployment_mode == "agg":
        if not isinstance(parallel_config, ReplicaParallelConfig):
            raise TypeError("agg deployment_mode needs a ReplicaParallelConfig")
        out = _shape_fields(parallel_config.shape)
        out["replicas"] = parallel_config.replicas
        out["used_gpus"] = parallel_config.total_gpus
        return out
    if not isinstance(parallel_config, DisaggParallelConfig):
        raise TypeError("disagg deployment_mode needs a DisaggParallelConfig")
    out = {}
    for role, rc in (
        ("prefill", parallel_config.prefill),
        ("decode", parallel_config.decode),
    ):
        for key, value in _shape_fields(rc.shape).items():
            out[f"{role}_{key}"] = value
        out[f"{role}_replicas"] = rc.replicas
    out["used_gpus"] = parallel_config.total_gpus
    return out


def _unroll_planner(
    selection: dict[str, Any], load_predictor: LoadPredictorResult | None
) -> dict[str, Any]:
    # Each planner composite entry is a preset id (str) or a dict pinning the
    # unrolled fields; the planner decoders accept both.
    scaling_entry = selection["planner_scaling_policy"]
    scaling = scaling_fields(scaling_entry)
    enable_throughput = scaling["enable_throughput_scaling"]
    enable_load = scaling["enable_load_scaling"]
    out: dict[str, Any] = {
        "planner_scaling_policy": scaling_entry,  # raw entry (str or dict) kept for traceability
        "enable_throughput_scaling": enable_throughput,
        "enable_load_scaling": enable_load,
    }
    if not (enable_throughput or enable_load):
        return (
            out  # planner off (disabled): no intervals / fpm / sensitivity / predictor
        )

    out["throughput_adjustment_interval_seconds"] = scaling[
        "throughput_adjustment_interval_seconds"
    ]
    out["load_adjustment_interval_seconds"] = scaling[
        "load_adjustment_interval_seconds"
    ]
    out.update(fpm_fields(selection["planner_fpm_sampling"]))
    out.update(load_sensitivity_fields(selection["planner_load_sensitivity"]))

    # The load predictor is the forecaster for predictive throughput scaling; it
    # comes from the independent sweep's winner for this policy's interval.
    if enable_throughput and load_predictor is not None:
        entry = load_predictor.best_by_interval.get(
            scaling["throughput_adjustment_interval_seconds"]
        )
        if entry is not None:
            out.update(predictor_fields(entry))
    return out


def unroll_sample(
    *,
    search_space: SearchSpace,
    selection: dict[str, Any],
    parallel_config: ReplicaParallelConfig | DisaggParallelConfig,
    load_predictor: LoadPredictorResult | None = None,
) -> dict[str, Any]:
    """Expand a per-knob ``selection`` into a flat deployment config dict.

    ``selection`` holds one value per searchable knob (atomic knobs + the planner
    preset ids); ``parallel_config`` is the chosen ``parallel_configs`` element;
    ``load_predictor`` is the result of :func:`dynamo.profiler.spica.load_predictor_sweep.sweep_load_predictor`
    (its per-interval winner is consulted only under throughput scaling). Composite
    knobs are unrolled and only mode/router/planner-relevant knobs are emitted.
    """
    mode = selection["deployment_mode"]
    sample: dict[str, Any] = {"deployment_mode": mode, "backend": selection["backend"]}

    for key in _DEPLOYMENT_PINNED + _KV_MANAGER:
        sample[key] = getattr(search_space, key)

    sample.update(_unroll_parallel(mode, parallel_config))

    # engine knobs for the active branch only
    if mode == "agg":
        searched, pinned = _AGG_SEARCHED, _AGG_PINNED
    else:
        searched = _PREFILL_SEARCHED + _DECODE_SEARCHED
        pinned = _PREFILL_PINNED + _DECODE_PINNED
    for key in searched:
        sample[key] = selection[key]
    for key in pinned:
        sample[key] = getattr(search_space, key)

    # router: kv-router knobs + admission control only apply under kv_router
    router_mode = selection["router_mode"]
    sample["router_mode"] = router_mode
    if router_mode == "kv_router":
        for key in _KV_ROUTER_KNOBS:
            if (
                key in selection
            ):  # host/disk cache weights are gated out when offload is off
                sample[key] = selection[key]
        for key in _ROUTER_ADMISSION:
            sample[key] = getattr(search_space, key)

    sample.update(_unroll_planner(selection, load_predictor))
    return sample
