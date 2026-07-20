# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate an unrolled sample (flat config dict) into replay deployment inputs.

Produces the JSON payloads + worker counts the :class:`dynamo.profiler.spica.evaluator.ReplayEvaluator`
feeds to Dynamo replay: per-role ``MockEngineArgs`` dicts (built from the AIC
parallelism), an optional ``PlannerConfig`` dict, worker counts, and router config.

Two replay paths, keyed on the planner policy (see investigation: no planner
config == static/"disabled"):

- ``enable_*_scaling`` both off  -> ``planner_config = None`` -> the plain
  ``run_trace_replay`` (static worker counts; emits gpu_hours, not goodput).
- otherwise -> a ``PlannerConfig`` -> the planner-bridge replay (scaling; goodput
  + gpu_hours). ``optimization_target`` is the planner's scaling objective derived
  from the sweep goal (see ``OptimizationTarget.planner_optimization_target``):
  ``goodput``/``goodput_per_gpu`` -> ``"sla"``, ``throughput`` ->
  ``"throughput"``, ``e2e_latency`` -> ``"latency"``.

Pure dict-building (no dynamo import), so it is unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SLATarget

# Per-engine GPU count and the goodput SLA are threaded so the report carries
# gpu_hours / goodput; the planner's own scaling SLA is set independently.

# Planner fields copied straight from the unrolled sample into the PlannerConfig
# payload when present (unroll_sample already decoded the presets to these).
_PLANNER_PASSTHROUGH = (
    "enable_throughput_scaling",
    "enable_load_scaling",
    "throughput_adjustment_interval_seconds",
    "load_adjustment_interval_seconds",
    "max_num_fpm_samples",
    "fpm_sample_bucket_size",
    "load_scaling_down_sensitivity",
    "load_min_observations",
    "load_predictor",
    "load_predictor_log1p",
    "prophet_window_size",
    "kalman_q_level",
    "kalman_q_trend",
    "kalman_r",
    "kalman_min_points",
)


@dataclass(frozen=True)
class DeploymentPlan:
    """Everything the evaluator needs to run one replay for a candidate."""

    deployment_mode: str  # "agg" | "disagg"
    is_static: bool  # True -> plain replay (no planner); False -> planner bridge
    # MockEngineArgs JSON payloads (omit num_gpu_blocks -> replay estimates it).
    agg_engine_args: dict[str, Any] | None
    prefill_engine_args: dict[str, Any] | None
    decode_engine_args: dict[str, Any] | None
    num_workers: int  # agg replica count
    num_prefill_workers: int
    num_decode_workers: int
    router_mode: str
    router_config: dict[str, Any] | None  # kv-router knobs, or None for round_robin
    planner_config: dict[str, Any] | None  # PlannerConfig payload, or None when static


def _role_prefix(role: str) -> str:
    """Field prefix in the unrolled sample for a role ('' for agg shape fields)."""
    return "" if role == "agg" else f"{role}_"


def _engine_args_payload(
    sample: dict, role: str, *, backend_version: str
) -> dict[str, Any]:
    """MockEngineArgs JSON for one role, from the sample's AIC parallelism."""
    p = _role_prefix(role)
    tp = int(sample[f"{p}tp"])
    attention_dp = int(sample[f"{p}attention_dp"])
    moe_tp = int(sample[f"{p}moe_tp"])
    moe_ep = int(sample[f"{p}moe_ep"])
    payload: dict[str, Any] = {
        "worker_type": "aggregated" if role == "agg" else role,
        # the simulated scheduler backend; mirrors the swept aic_backend so the
        # mocker picks the right engine (vllm/sglang/trtllm), not its vLLM default.
        "engine_type": sample["backend"],
        "aic_backend": sample["backend"],
        "aic_backend_version": backend_version,
        "aic_system": sample["hardware_sku"],
        "aic_model_path": sample["model_name"],
        "aic_tp_size": tp,
        "aic_attention_dp_size": attention_dp,
        # batching + memory knobs for the role
        "max_num_batched_tokens": int(sample[f"{role}_max_num_batched_tokens"]),
        "max_num_seqs": int(sample[f"{role}_max_num_seqs"]),
        "block_size": int(sample[f"{role}_block_size"]),
        "gpu_memory_utilization": float(sample[f"{role}_gpu_memory_utilization"]),
        "enable_prefix_caching": bool(sample[f"{role}_enable_prefix_caching"]),
    }
    # MoE expert sharding only for MoE shapes (tp*attention_dp == moe_tp*moe_ep);
    # dense (moe_tp==moe_ep==1) leaves them unset.
    if moe_tp * moe_ep > 1:
        payload["aic_moe_tp_size"] = moe_tp
        payload["aic_moe_ep_size"] = moe_ep
    # TODO(dynamo.profiler.spica): fuller speculative-decoding (nextn / MTP) support is future work — expose
    #   nextn + accept-rates as searchable knobs and validate them; the related low-priority
    #   test/polish items from the PR review are deferred until then.
    if sample.get("aic_nextn") is not None:
        payload["aic_nextn"] = int(sample["aic_nextn"])
    if sample.get("startup_time") is not None:
        payload["startup_time"] = float(sample["startup_time"])
    # KVBM is disabled when G2 has no blocks. When enabled it runs on aggregate or
    # prefill workers only; disaggregated decode workers use the transfer connector
    # and must not be scored with a second local offload tier.
    if role != "decode" and int(sample.get("num_g2_blocks") or 0) > 0:
        for key, value in sample.items():
            if value is None:
                continue
            if (
                key in {"kv_bytes_per_token", "offload_batch_size"}
                or key.startswith("num_g")
                or key.startswith("bandwidth_g")
            ):
                payload[key] = value
    return payload


def _planner_config_payload(
    sample: dict, *, optimization_target: str, planner_sla: SLATarget | None
) -> dict[str, Any] | None:
    """PlannerConfig JSON for a scaling candidate, or None when the planner is
    disabled (static -> plain replay).

    ``optimization_target`` is the planner's scaling objective, derived from the
    sweep's goal (``goal.target.planner_optimization_target``) — NOT from the scaling
    policy. The ``planner_scaling_policy`` only decides which scaling mechanisms run
    (enable_throughput / enable_load + intervals); the policy filter upstream already
    ensures throughput scaling (which needs ``"sla"``) is only present for goodput
    sweeps. ttft/itl are seeded only under ``"sla"`` (other targets ignore them)."""
    enable_throughput = bool(sample.get("enable_throughput_scaling", False))
    enable_load = bool(sample.get("enable_load_scaling", False))
    if not (enable_throughput or enable_load):
        return None  # "disabled" -> static plain replay

    # SLA-based scaling drives the planner off ttft+itl; an e2e-only SLA can't seed
    # them (the planner has no e2e scaling target), so reject it up front rather than
    # silently scaling with no SLA. (The goodput SLA the evaluator uses is separate.)
    if (
        optimization_target == "sla"
        and planner_sla is not None
        and (planner_sla.ttft_ms is None or planner_sla.itl_ms is None)
    ):
        raise ValueError(
            "SLA-based planner scaling (optimization_target='sla') requires both ttft_ms "
            "and itl_ms; an e2e-only SLA cannot drive the planner's scaling target"
        )

    payload: dict[str, Any] = {
        "mode": sample["deployment_mode"],
        "optimization_target": optimization_target,
    }
    # Spica consumes the trace_report directly and sweeps many candidates, so turn
    # off the planner's per-run diagnostics (a ~5 MB HTML in ./planner_reports/ per
    # candidate) and the live dashboard, both of which default on.
    payload["report_interval_hours"] = None
    payload["live_dashboard_port"] = 0
    for key in _PLANNER_PASSTHROUGH:
        if key in sample:
            payload[key] = sample[key]
    # Search-space limits become runtime policy once the planner is active. Keep
    # them explicit rather than broadly passing arbitrary sample fields through.
    if sample.get("gpu_budget") is not None:
        payload["max_gpu_budget"] = int(sample["gpu_budget"])
    if sample.get("min_gpu_budget") is not None:
        payload["min_gpu_budget"] = int(sample["min_gpu_budget"])
    if sample.get("min_endpoint") is not None:
        payload["min_endpoint"] = int(sample["min_endpoint"])
    # Per-engine GPU counts for the planner's cost logic (gpu_hours itself is
    # derived by the mocker from aic_tp x aic_attention_dp, not from these).
    if sample["deployment_mode"] == "disagg":
        payload["prefill_engine_num_gpu"] = int(sample["prefill_tp"]) * int(
            sample["prefill_attention_dp"]
        )
        payload["decode_engine_num_gpu"] = int(sample["decode_tp"]) * int(
            sample["decode_attention_dp"]
        )
    else:
        payload["decode_engine_num_gpu"] = int(sample["tp"]) * int(
            sample["attention_dp"]
        )
    # SLA-based scaling uses ttft/itl; seed them from the goal's SLA. Other targets
    # ignore ttft/itl (planner warns), so only set them under "sla".
    if optimization_target == "sla" and planner_sla is not None:
        if planner_sla.ttft_ms is not None:
            payload["ttft_ms"] = planner_sla.ttft_ms
        if planner_sla.itl_ms is not None:
            payload["itl_ms"] = planner_sla.itl_ms
    return payload


def _router_config_payload(sample: dict) -> dict[str, Any] | None:
    """kv-router knobs, or None under round_robin."""
    if sample.get("router_mode") != "kv_router":
        return None
    keys = (
        "overlap_score_credit",
        "prefill_load_scale",
        "host_cache_hit_weight",
        "disk_cache_hit_weight",
        "router_temperature",
    )
    return {k: sample[k] for k in keys if k in sample}


def build_deployment(
    sample: dict,
    *,
    backend_version: str,
    optimization_target: str = "sla",
    planner_sla: SLATarget | None = None,
) -> DeploymentPlan:
    """Translate one unrolled sample into a :class:`DeploymentPlan`.

    ``optimization_target`` is the planner's scaling objective (from the sweep goal,
    via ``OptimizationTarget.planner_optimization_target``); it is only used when the
    candidate has a planner (a non-disabled scaling policy)."""
    mode = sample["deployment_mode"]
    planner_config = _planner_config_payload(
        sample, optimization_target=optimization_target, planner_sla=planner_sla
    )
    router_mode = sample.get("router_mode", "round_robin")
    common = dict(
        deployment_mode=mode,
        is_static=planner_config is None,
        router_mode=router_mode,
        router_config=_router_config_payload(sample),
        planner_config=planner_config,
    )
    if mode == "agg":
        return DeploymentPlan(
            agg_engine_args=_engine_args_payload(
                sample, "agg", backend_version=backend_version
            ),
            prefill_engine_args=None,
            decode_engine_args=None,
            num_workers=int(sample["replicas"]),
            num_prefill_workers=0,
            num_decode_workers=0,
            **common,
        )
    return DeploymentPlan(
        agg_engine_args=None,
        prefill_engine_args=_engine_args_payload(
            sample, "prefill", backend_version=backend_version
        ),
        decode_engine_args=_engine_args_payload(
            sample, "decode", backend_version=backend_version
        ),
        num_workers=0,
        num_prefill_workers=int(sample["prefill_replicas"]),
        num_decode_workers=int(sample["decode_replicas"]),
        **common,
    )
