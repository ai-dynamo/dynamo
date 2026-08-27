# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner preparation for Dynamo offline replay SDK callers."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import msgspec

from dynamo._internal.aic import create_session
from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.mocker import MockEngineArgs

if TYPE_CHECKING:
    from dynamo.planner.core.types import EngineCapabilities

logger = logging.getLogger(__name__)


def _engine_caps(args: MockEngineArgs) -> EngineCapabilities:
    """Derive Planner engine capabilities from mock-engine arguments."""

    from dynamo.planner.core.types import EngineCapabilities

    dp_size = max(args.dp_size, 1)
    max_kv_tokens = args.num_gpu_blocks * args.block_size * dp_size
    return EngineCapabilities(
        num_gpu=(args.aic_tp_size or 1) * dp_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        context_length=args.max_model_len,
        max_kv_tokens=max_kv_tokens if max_kv_tokens > 0 else None,
        speculative_nextn=args.aic_nextn,
    )


def _generate_aic_prefill_fpms(
    aic_session,
    engine_args: MockEngineArgs,
    granularity: int = 8,
) -> list[ForwardPassMetrics]:
    prefill_max = engine_args.max_num_batched_tokens or 8192
    prefill_step = max(1, (prefill_max - 100) // granularity)
    prefill_fpms: list[ForwardPassMetrics] = []
    for isl in range(100, prefill_max + 1, prefill_step):
        ttft_ms = aic_session.predict_prefill(1, isl, 0)
        if ttft_ms > 0:
            prefill_fpms.append(
                ForwardPassMetrics(
                    wall_time=ttft_ms / 1000.0,
                    scheduled_requests=ScheduledRequestMetrics(
                        num_prefill_requests=1,
                        sum_prefill_tokens=isl,
                    ),
                )
            )
    return prefill_fpms


def _generate_aic_decode_fpms(
    aic_session,
    engine_args: MockEngineArgs,
    granularity: int = 8,
) -> list[ForwardPassMetrics]:
    max_kv_tokens = engine_args.num_gpu_blocks * engine_args.block_size
    if max_kv_tokens <= 0:
        max_kv_tokens = 16384 * 16

    decode_fpms: list[ForwardPassMetrics] = []
    ctx_lengths = [500, 2000, 4000, 8000]
    bs_max = engine_args.max_num_seqs or 256
    bs_step = max(1, bs_max // granularity)
    for ctx_len in ctx_lengths:
        for bs in range(1, bs_max + 1, bs_step):
            sum_kv = bs * ctx_len
            if sum_kv > max_kv_tokens:
                break
            itl_ms = aic_session.predict_decode(bs, ctx_len, 2)
            if itl_ms > 0:
                decode_fpms.append(
                    ForwardPassMetrics(
                        wall_time=itl_ms / 1000.0,
                        scheduled_requests=ScheduledRequestMetrics(
                            num_decode_requests=bs,
                            sum_decode_kv_tokens=sum_kv,
                        ),
                    )
                )
    return decode_fpms


def _aic_fpm_digest(
    prefill_fpms: list[ForwardPassMetrics],
    decode_fpms: list[ForwardPassMetrics],
) -> str:
    payload = {
        "prefill": [msgspec.to_builtins(fpm) for fpm in prefill_fpms],
        "decode": [msgspec.to_builtins(fpm) for fpm in decode_fpms],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _aic_performance_model_config(
    metadata: dict[str, Any], roles: tuple[str, ...]
) -> dict[str, Any] | None:
    """Select one runner-neutral AIC identity from equivalent role aliases."""

    for role in roles:
        raw = metadata.get(role)
        if raw is None:
            continue
        if not isinstance(raw, dict):
            raise TypeError(f"performance_model_metadata.{role} must be a mapping")
        if raw.get("provider") != "aic":
            continue
        config = raw.get("config")
        if not isinstance(config, dict):
            raise TypeError(
                f"performance_model_metadata.{role}.config must be a mapping"
            )
        return dict(config)
    return None


def _aic_performance_model_configs(
    metadata: dict[str, Any] | None, mode: str
) -> dict[str, dict[str, Any]]:
    """Return the AIC identity used for each generated Planner FPM role."""

    if not metadata:
        return {}
    if mode == "agg":
        config = _aic_performance_model_config(metadata, ("aggregated", "agg"))
        return {} if config is None else {"prefill": config, "decode": config}

    configs = {
        role: config
        for role in ("prefill", "decode")
        if (config := _aic_performance_model_config(metadata, (role,))) is not None
    }
    if configs and set(configs) != {"prefill", "decode"}:
        missing = sorted({"prefill", "decode"} - set(configs))
        raise ValueError(
            "disagg performance_model_metadata must provide AIC identities for "
            f"both prefill and decode; missing {missing}"
        )
    return configs


def _aic_session_kwargs(
    perf_config: dict[str, Any] | None,
    engine_args: MockEngineArgs,
) -> dict[str, Any] | None:
    """Build one role-specific AIC session request."""

    backend = (
        perf_config.get("backend")
        if perf_config is not None
        else engine_args.aic_backend
    )
    system = (
        perf_config.get("system") if perf_config is not None else engine_args.aic_system
    )
    model_path = (
        perf_config.get("model_path")
        if perf_config is not None
        else engine_args.aic_model_path
    )
    if backend is None or system is None or model_path is None:
        return None
    nextn = (
        perf_config.get("nextn") if perf_config is not None else engine_args.aic_nextn
    )
    nextn_accept_rates = ",".join(["0"] * int(nextn)) if nextn is not None else None
    return {
        "backend_name": backend,
        "system": system,
        "model_path": model_path,
        "tp_size": (
            perf_config.get("tp_size", 1)
            if perf_config is not None
            else engine_args.aic_tp_size or 1
        ),
        "backend_version": (
            perf_config.get("backend_version")
            if perf_config is not None
            else engine_args.aic_backend_version
        ),
        "moe_tp_size": (
            perf_config.get("moe_tp_size")
            if perf_config is not None
            else engine_args.aic_moe_tp_size
        ),
        "moe_ep_size": (
            perf_config.get("moe_ep_size")
            if perf_config is not None
            else engine_args.aic_moe_ep_size
        ),
        "attention_dp_size": (
            perf_config.get("attention_dp_size")
            if perf_config is not None
            else engine_args.aic_attention_dp_size
        ),
        "gemm_dtype": engine_args.aic_gemm_dtype,
        "moe_dtype": engine_args.aic_moe_dtype,
        "fmha_dtype": engine_args.aic_fmha_dtype,
        "kv_cache_dtype": engine_args.aic_kv_cache_dtype,
        "comm_dtype": engine_args.aic_comm_dtype,
        "nextn": nextn,
        "nextn_accept_rates": nextn_accept_rates,
    }


def prepare_planner_replay(
    extra_engine_args: MockEngineArgs | None,
    prefill_engine_args: MockEngineArgs | None,
    decode_engine_args: MockEngineArgs | None,
    planner_config_arg: str,
    benchmark_granularity: int = 8,
    capture_details: bool = True,
    performance_model_metadata: dict[str, Any] | None = None,
):
    """Create and bootstrap the scaling component for an offline replay."""

    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.core.types import WorkerCapabilities
    from dynamo.planner.offline.replay_adapter import create_replay_planner_adapter
    from dynamo.planner.offline.trace_data import (
        extract_traffic_observations_from_trace,
    )

    planner_config = PlannerConfig.from_config_arg(planner_config_arg)
    planner_config.advisory = True

    if planner_config.mode == "agg":
        extra_engine_args = extra_engine_args or MockEngineArgs()
        capabilities = WorkerCapabilities(decode=_engine_caps(extra_engine_args))
    elif planner_config.mode == "disagg":
        if prefill_engine_args is None or decode_engine_args is None:
            raise ValueError(
                "disagg planner replay requires prefill and decode engine arguments"
            )
        capabilities = WorkerCapabilities(
            prefill=_engine_caps(prefill_engine_args),
            decode=_engine_caps(decode_engine_args),
        )
    else:
        raise ValueError(
            "planner-in-the-loop replay supports mode='agg' or 'disagg', "
            f"got {planner_config.mode!r}"
        )

    warmup_observations = None
    if planner_config.load_predictor_warmup_trace is not None:
        warmup_observations = extract_traffic_observations_from_trace(
            planner_config.load_predictor_warmup_trace,
            planner_config.throughput_adjustment_interval_seconds,
        )

    adapter = create_replay_planner_adapter(
        planner_config=planner_config,
        capabilities=capabilities,
        benchmark_granularity=benchmark_granularity,
        warmup_observations=warmup_observations,
        capture_details=capture_details,
    )
    adapter.set_bootstrap_metadata({"status": "not_required"})
    if adapter._is_easy_mode():
        return adapter

    perf_configs = _aic_performance_model_configs(
        performance_model_metadata, planner_config.mode
    )
    if perf_configs:
        ref_args = (
            extra_engine_args
            or decode_engine_args
            or prefill_engine_args
            or MockEngineArgs()
        )
    else:
        ref_args = (
            extra_engine_args
            or (
                decode_engine_args
                if decode_engine_args is not None
                and decode_engine_args.aic_backend is not None
                else None
            )
            or prefill_engine_args
            or decode_engine_args
            or MockEngineArgs()
        )
    p_args = (
        extra_engine_args if planner_config.mode == "agg" else prefill_engine_args
    ) or ref_args
    d_args = (
        extra_engine_args if planner_config.mode == "agg" else decode_engine_args
    ) or ref_args
    if perf_configs:
        prefill_session_kwargs = _aic_session_kwargs(perf_configs["prefill"], p_args)
        decode_session_kwargs = _aic_session_kwargs(perf_configs["decode"], d_args)
    else:
        shared_session_kwargs = _aic_session_kwargs(None, ref_args)
        prefill_session_kwargs = shared_session_kwargs
        decode_session_kwargs = shared_session_kwargs
    if prefill_session_kwargs is None or decode_session_kwargs is None:
        adapter.set_bootstrap_metadata(
            {
                "status": "not_configured_load_only",
                "benchmark_granularity": benchmark_granularity,
            }
        )
        logger.warning(
            "throughput-based scaling regression requires AIC perf model; "
            "falling back to load-based scaling only"
        )
        return adapter

    try:
        prefill_session = create_session(**prefill_session_kwargs)
        decode_session = (
            prefill_session
            if decode_session_kwargs == prefill_session_kwargs
            else create_session(**decode_session_kwargs)
        )
    except (
        ImportError,
        RuntimeError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        logger.warning(
            "AIC session creation failed (%s); throughput regression will not "
            "be bootstrapped",
            exc,
        )
        adapter.set_bootstrap_metadata(
            {
                "status": "session_failed_load_only",
                "benchmark_granularity": benchmark_granularity,
            }
        )
        return adapter

    try:
        prefill_fpms = _generate_aic_prefill_fpms(
            prefill_session, p_args, benchmark_granularity
        )
        decode_fpms = _generate_aic_decode_fpms(
            decode_session, d_args, benchmark_granularity
        )
    except (RuntimeError, ValueError, KeyError, ArithmeticError) as exc:
        logger.warning(
            "AIC benchmark generation failed (%s); throughput regression will "
            "not be bootstrapped",
            exc,
        )
        prefill_fpms, decode_fpms = [], []

    bootstrap_metadata = {
        "status": "installed",
        "benchmark_granularity": benchmark_granularity,
        "prefill_fpm_count": len(prefill_fpms),
        "decode_fpm_count": len(decode_fpms),
        "fpm_sha256": _aic_fpm_digest(prefill_fpms, decode_fpms),
    }
    if planner_config.mode == "agg":
        agg_fpms = prefill_fpms + decode_fpms
        if agg_fpms:
            adapter.install_benchmark_fpms(agg_fpms=agg_fpms)
        else:
            bootstrap_metadata["status"] = "empty"
            logger.warning("AIC produced no agg benchmark FPMs")
    elif prefill_fpms and decode_fpms:
        adapter.install_benchmark_fpms(
            prefill_fpms=prefill_fpms,
            decode_fpms=decode_fpms,
        )
    else:
        bootstrap_metadata["status"] = "empty"
        logger.warning(
            "AIC produced empty benchmark FPMs (prefill=%d, decode=%d)",
            len(prefill_fpms),
            len(decode_fpms),
        )
    adapter.set_bootstrap_metadata(bootstrap_metadata)
    return adapter


@contextmanager
def planner_replay_adapter(
    extra_engine_args: MockEngineArgs | None,
    prefill_engine_args: MockEngineArgs | None,
    decode_engine_args: MockEngineArgs | None,
    planner_config_arg: str,
    benchmark_granularity: int = 8,
    capture_details: bool = True,
    performance_model_metadata: dict[str, Any] | None = None,
) -> Iterator:
    """Own Planner preparation, replay execution, and cleanup as one scope."""

    adapter = prepare_planner_replay(
        extra_engine_args=extra_engine_args,
        prefill_engine_args=prefill_engine_args,
        decode_engine_args=decode_engine_args,
        planner_config_arg=planner_config_arg,
        performance_model_metadata=performance_model_metadata,
        benchmark_granularity=benchmark_granularity,
        capture_details=capture_details,
    )
    with adapter:
        yield adapter
