# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner preparation for Dynamo offline replay SDK callers."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

import msgspec
from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.mocker import MockEngineArgs

if TYPE_CHECKING:
    from dynamo.planner.core.types import EngineCapabilities


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


def prepare_planner_replay(
    extra_engine_args: MockEngineArgs | None,
    prefill_engine_args: MockEngineArgs | None,
    decode_engine_args: MockEngineArgs | None,
    planner_config_arg: str,
    benchmark_granularity: int = 8,
    capture_details: bool = True,
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
    aic_backend = ref_args.aic_backend
    if (
        aic_backend is None
        or ref_args.aic_system is None
        or ref_args.aic_model_path is None
    ):
        adapter.set_bootstrap_metadata(
            {
                "status": "not_configured_load_only",
                "benchmark_granularity": benchmark_granularity,
            }
        )
        sys.stderr.write(
            "Note: throughput-based scaling regression requires AIC perf model; "
            "falling back to load-based scaling only.\n"
        )
        return adapter

    try:
        from dynamo._internal.aic import create_session

        aic_session = create_session(
            backend_name=aic_backend,
            system=ref_args.aic_system,
            model_path=ref_args.aic_model_path,
            tp_size=ref_args.aic_tp_size or 1,
            backend_version=ref_args.aic_backend_version,
            moe_tp_size=ref_args.aic_moe_tp_size,
            moe_ep_size=ref_args.aic_moe_ep_size,
            attention_dp_size=ref_args.aic_attention_dp_size,
            gemm_dtype=ref_args.aic_gemm_dtype,
            moe_dtype=ref_args.aic_moe_dtype,
            fmha_dtype=ref_args.aic_fmha_dtype,
            kv_cache_dtype=ref_args.aic_kv_cache_dtype,
            comm_dtype=ref_args.aic_comm_dtype,
            nextn=d_args.aic_nextn,
            nextn_accept_rates=(
                ",".join(["0"] * d_args.aic_nextn)
                if d_args.aic_nextn is not None
                else None
            ),
        )
    except (
        ImportError,
        RuntimeError,
        ValueError,
        KeyError,
        FileNotFoundError,
    ) as exc:
        sys.stderr.write(
            f"Warning: AIC session creation failed ({exc}); throughput regression "
            "will not be bootstrapped.\n"
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
            aic_session, p_args, benchmark_granularity
        )
        decode_fpms = _generate_aic_decode_fpms(
            aic_session, d_args, benchmark_granularity
        )
    except (RuntimeError, ValueError, KeyError, ArithmeticError) as exc:
        sys.stderr.write(
            f"Warning: AIC benchmark generation failed ({exc}); throughput "
            "regression will not be bootstrapped.\n"
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
            sys.stderr.write("Warning: AIC produced no agg benchmark FPMs\n")
    elif prefill_fpms and decode_fpms:
        adapter.install_benchmark_fpms(
            prefill_fpms=prefill_fpms,
            decode_fpms=decode_fpms,
        )
    else:
        bootstrap_metadata["status"] = "empty"
        sys.stderr.write(
            "Warning: AIC produced empty benchmark FPMs "
            f"(prefill={len(prefill_fpms)}, decode={len(decode_fpms)})\n"
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
) -> Iterator:
    """Own Planner preparation, replay execution, and cleanup as one scope."""

    adapter = prepare_planner_replay(
        extra_engine_args=extra_engine_args,
        prefill_engine_args=prefill_engine_args,
        decode_engine_args=decode_engine_args,
        planner_config_arg=planner_config_arg,
        benchmark_granularity=benchmark_granularity,
        capture_details=capture_details,
    )
    with adapter:
        yield adapter
