# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``python -m dynamo.replay`` CLI for single-run simulation.

This module owns Dynamo-specific Planner preparation and the online replay
surface in addition to the shared offline replay integration.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol

import msgspec

from aisimulate.aic import materialize_aic_num_gpu_blocks
from aisimulate.replay.cli import add_base_replay_arguments
from aisimulate.replay.config import parse_base_replay_config

if TYPE_CHECKING:
    from dynamo.planner.core.types import EngineCapabilities

from dynamo._internal.aic import _normalize_aic_quant_mode
from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.llm import AicPerfConfig, KvRouterConfig
from dynamo.mocker import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay, run_trace_replay
from dynamo.replay.reporting import format_report_table, write_report_json


class PlannerProfileDataResult(Protocol):
    npz_path: Path | None


def _load_router_config(
    router_config_json: str | None,
    router_policy_config: str | None,
):
    if router_policy_config is None:
        return (
            KvRouterConfig.from_json(router_config_json)
            if router_config_json is not None
            else None
        )

    values = json.loads(router_config_json) if router_config_json is not None else {}
    if not isinstance(values, dict):
        raise ValueError("--router-config must contain a JSON object")
    values["router_policy_config"] = router_policy_config
    return KvRouterConfig.from_json(json.dumps(values))


def resolve_planner_profile_data(
    planner_profile_data: Path | None,
) -> PlannerProfileDataResult:
    if planner_profile_data is None:
        return SimpleNamespace(npz_path=None)

    if planner_profile_data.suffix == ".npz":
        return SimpleNamespace(npz_path=planner_profile_data)

    try:
        module = importlib.import_module("dynamo.mocker.args")
    except ImportError:
        return SimpleNamespace(
            npz_path=None,
        )
    return module.resolve_planner_profile_data(planner_profile_data)


def _resolve_aic_num_gpu_blocks(raw: dict) -> None:
    """Compatibility wrapper around the AISimulate-owned capacity lowerer."""

    lowered = materialize_aic_num_gpu_blocks(raw)
    raw.clear()
    raw.update(lowered)


def _with_aic_speculation(raw: dict | None, args) -> dict | None:
    """Default the speculative-decoding engine args from the ``--aic-*`` flags.

    ``--aic-nextn`` and ``--aic-nextn-accept-rates`` are routed into
    ``AicPerfConfig``, which only feeds the router's prefill-load estimator. The
    engine timing and the speculative burst sampler read them off the engine
    args instead, which is why the same values passed through
    ``--extra-engine-args`` take effect while the dedicated flags did not.

    An explicit value in the engine args wins, so a caller can still override
    per worker type. Returns ``raw`` untouched when neither flag was given, so
    a run with no engine args and no speculation keeps passing ``None``.
    """
    nextn = getattr(args, "aic_nextn", None)
    accept_rates = getattr(args, "aic_nextn_accept_rates", None)
    if nextn is None and accept_rates is None:
        return raw

    merged = dict(raw or {})
    if nextn is not None:
        merged.setdefault("aic_nextn", nextn)
    if accept_rates is not None:
        merged.setdefault("aic_nextn_accept_rates", accept_rates)
    return merged


def _load_engine_args(raw_args: str | None):
    if raw_args is None:
        return None

    raw = json.loads(raw_args)
    if not isinstance(raw, dict):
        raise ValueError("engine-args must be a JSON object")
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
            raise ValueError(
                "worker_type must be one of 'aggregated', 'prefill', or 'decode'"
            )
    if "planner_profile_data" in raw:
        if raw["planner_profile_data"] is None:
            del raw["planner_profile_data"]
        else:
            profile_data_result = resolve_planner_profile_data(
                Path(raw["planner_profile_data"])
            )
            if profile_data_result.npz_path is not None:
                raw["planner_profile_data"] = str(profile_data_result.npz_path)
            else:
                del raw["planner_profile_data"]
    _resolve_aic_num_gpu_blocks(raw)
    return MockEngineArgs.from_json(json.dumps(raw))


def _load_aic_perf_config(args: argparse.Namespace):
    values = {
        "aic_backend": args.aic_backend,
        "aic_system": args.aic_system,
        "aic_model_path": args.aic_model_path,
        "aic_backend_version": args.aic_backend_version,
        "aic_tp_size": args.aic_tp_size,
        "aic_moe_tp_size": args.aic_moe_tp_size,
        "aic_moe_ep_size": args.aic_moe_ep_size,
        "aic_attention_dp_size": args.aic_attention_dp_size,
        # Normalize here so `--aic-*-dtype auto` (== model default) doesn't make
        # the "any AIC value set?" check below think an AIC config was requested
        # and then demand --aic-backend/--aic-system/--aic-model-path.
        "aic_gemm_dtype": _normalize_aic_quant_mode(args.aic_gemm_dtype),
        "aic_moe_dtype": _normalize_aic_quant_mode(args.aic_moe_dtype),
        "aic_fmha_dtype": _normalize_aic_quant_mode(args.aic_fmha_dtype),
        "aic_kv_cache_dtype": _normalize_aic_quant_mode(args.aic_kv_cache_dtype),
        "aic_comm_dtype": _normalize_aic_quant_mode(args.aic_comm_dtype),
        "aic_nextn": args.aic_nextn,
        "aic_nextn_accept_rates": args.aic_nextn_accept_rates,
    }
    if not any(value is not None for value in values.values()):
        return None

    missing = [
        name
        for name in ("aic_backend", "aic_system", "aic_model_path")
        if values[name] is None
    ]
    if missing:
        missing_flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"AIC replay modeling requires {missing_flags}")

    return AicPerfConfig(
        aic_backend=values["aic_backend"],
        aic_system=values["aic_system"],
        aic_model_path=values["aic_model_path"],
        aic_tp_size=values["aic_tp_size"] or 1,
        aic_backend_version=values["aic_backend_version"],
        aic_moe_tp_size=values["aic_moe_tp_size"],
        aic_moe_ep_size=values["aic_moe_ep_size"],
        aic_attention_dp_size=values["aic_attention_dp_size"],
        aic_gemm_dtype=values["aic_gemm_dtype"],
        aic_moe_dtype=values["aic_moe_dtype"],
        aic_fmha_dtype=values["aic_fmha_dtype"],
        aic_kv_cache_dtype=values["aic_kv_cache_dtype"],
        aic_comm_dtype=values["aic_comm_dtype"],
        aic_nextn=values["aic_nextn"],
        aic_nextn_accept_rates=values["aic_nextn_accept_rates"],
    )


def _engine_caps(args: MockEngineArgs) -> EngineCapabilities:
    """Derive EngineCapabilities from MockEngineArgs."""
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
    """Generate prefill benchmark FPMs using AIC predictions.

    Sweeps ISL at batch_size=1 within the engine's per-pass token budget
    (``max_num_batched_tokens``); a single forward pass physically cannot
    process more than that, so the regression shouldn't see larger sums.
    For longer ISL, callers use chunked TTFT estimation.
    """
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
    """Generate decode benchmark FPMs using AIC predictions.

    Sweeps (batch_size x context_length). ``granularity`` controls the
    number of sample points per axis; the batch-size ceiling comes from
    the engine's ``max_num_seqs`` so the regression sees realistic
    concurrency, not an artificial cap at the sweep density.
    """
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


def _prepare_planner_replay(
    extra_engine_args: MockEngineArgs | None,
    prefill_engine_args: MockEngineArgs | None,
    decode_engine_args: MockEngineArgs | None,
    planner_config_arg: str,
    benchmark_granularity: int = 8,
    capture_details: bool = True,
):
    """Create and bootstrap the scaling component for an offline replay.

    # TODO(jthomson04): SLA-based scaling (optimization_target="sla") with
    # disagg mode requires planner_profile_data (NPZ) or AIC-backed engine
    # args.  The default polynomial perf model does not account for batch
    # size in its decode timing, causing the DecodeRegressionModel's
    # num_decode_requests coefficient to go negative and reject the fit.
    # Fix the polynomial model to incorporate batch_size, or gate disagg
    # SLA mode on having a non-polynomial perf model.
    """
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
                "disagg planner replay requires --prefill-engine-args and --decode-engine-args"
            )
        capabilities = WorkerCapabilities(
            prefill=_engine_caps(prefill_engine_args),
            decode=_engine_caps(decode_engine_args),
        )
    else:
        raise ValueError(
            f"planner-in-the-loop replay supports mode='agg' or 'disagg', got '{planner_config.mode}'"
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

    # Bootstrap regression models from mocker's perf model.
    # AIC provides accurate batch-size-aware timing that works with the
    # planner's linear regression. The default polynomial model cannot
    # feed the throughput regression (its decode formula is quadratic in
    # utilization ratio, causing negative regression coefficients).
    if not adapter._is_easy_mode():
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
                "Note: throughput-based scaling regression requires AIC perf model "
                "(set aic_backend/aic_system/aic_model_path in --extra-engine-args). "
                "Falling back to load-based scaling only.\n"
            )
        else:
            # Create AIC session -- narrow to the concrete exception types
            # AIC/PyO3 can raise so we degrade gracefully on missing
            # dependencies or bad config, but don't swallow unrelated bugs
            # (AttributeError, KeyboardInterrupt, etc.) introduced by
            # refactors.
            try:
                from dynamo._internal.aic import create_session

                # The bootstrap uses one shared AIC identity (model / backend /
                # quantization, from ref_args) for both prefill and decode FPM
                # generation. That matches the planner's current model, where
                # prefill and decode differ only in parallelism — not in model
                # or quant — so a single AIC session is sufficient here.
                # Pre-existing behavior.
                aic_session = create_session(
                    backend_name=aic_backend,
                    system=ref_args.aic_system,
                    model_path=ref_args.aic_model_path,
                    tp_size=ref_args.aic_tp_size or 1,
                    backend_version=ref_args.aic_backend_version,
                    moe_tp_size=ref_args.aic_moe_tp_size,
                    moe_ep_size=ref_args.aic_moe_ep_size,
                    attention_dp_size=ref_args.aic_attention_dp_size,
                    # Same quant overrides the mocker's latency engine and the
                    # KV-block estimator use, so the planner's throughput
                    # regression is bootstrapped at the quantized precision
                    # instead of the model's default dtype. (AicSession
                    # normalizes the strings via _resolve_quant_mode.)
                    gemm_dtype=ref_args.aic_gemm_dtype,
                    moe_dtype=ref_args.aic_moe_dtype,
                    fmha_dtype=ref_args.aic_fmha_dtype,
                    kv_cache_dtype=ref_args.aic_kv_cache_dtype,
                    comm_dtype=ref_args.aic_comm_dtype,
                    nextn=d_args.aic_nextn,
                    # Model the full verification round; real conditional rates
                    # are consumed only by the mocker's burst sampler.
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
            ) as e:
                sys.stderr.write(
                    f"Warning: AIC session creation failed ({e}); "
                    "throughput regression will not be bootstrapped.\n"
                )
                adapter.set_bootstrap_metadata(
                    {
                        "status": "session_failed_load_only",
                        "benchmark_granularity": benchmark_granularity,
                    }
                )
                aic_session = None

            # Generate benchmark FPMs and load into regression.  Disagg
            # prefill and decode engines typically have different
            # max_num_seqs and KV cache sizes, so each sweep uses its own
            # engine args.  Agg has a single engine so uses one set.  AIC's
            # predict_* can raise on unsupported model/system combos or
            # numerical edge cases; log and fall back in those cases.
            if aic_session is not None:
                try:
                    prefill_fpms = _generate_aic_prefill_fpms(
                        aic_session, p_args, benchmark_granularity
                    )
                    decode_fpms = _generate_aic_decode_fpms(
                        aic_session, d_args, benchmark_granularity
                    )
                except (RuntimeError, ValueError, KeyError, ArithmeticError) as e:
                    sys.stderr.write(
                        f"Warning: AIC benchmark generation failed ({e}); "
                        "throughput regression will not be bootstrapped.\n"
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
                    # Agg regression fits on (sum_prefill_tokens, sum_decode_kv_tokens);
                    # combine prefill-only and decode-only points so both features
                    # have variance.
                    agg_fpms = prefill_fpms + decode_fpms
                    if agg_fpms:
                        adapter.install_benchmark_fpms(agg_fpms=agg_fpms)
                    else:
                        bootstrap_metadata["status"] = "empty"
                        sys.stderr.write(
                            "Warning: AIC produced no agg benchmark FPMs\n"
                        )
                else:
                    if prefill_fpms and decode_fpms:
                        adapter.install_benchmark_fpms(
                            prefill_fpms=prefill_fpms, decode_fpms=decode_fpms
                        )
                    else:
                        bootstrap_metadata["status"] = "empty"
                        sys.stderr.write(
                            f"Warning: AIC produced empty benchmark FPMs "
                            f"(prefill={len(prefill_fpms)}, decode={len(decode_fpms)})\n"
                        )
                adapter.set_bootstrap_metadata(bootstrap_metadata)

    return adapter


@contextmanager
def _planner_replay_adapter(
    extra_engine_args: MockEngineArgs | None,
    prefill_engine_args: MockEngineArgs | None,
    decode_engine_args: MockEngineArgs | None,
    planner_config_arg: str,
    benchmark_granularity: int = 8,
    capture_details: bool = True,
):
    """Own planner preparation, replay execution, and cleanup as one scope."""
    adapter = _prepare_planner_replay(
        extra_engine_args=extra_engine_args,
        prefill_engine_args=prefill_engine_args,
        decode_engine_args=decode_engine_args,
        planner_config_arg=planner_config_arg,
        benchmark_granularity=benchmark_granularity,
        capture_details=capture_details,
    )
    with adapter:
        yield adapter


def _write_per_request_jsonl(
    output_path: str | Path,
    records: list[dict] | None,
) -> None:
    if records is None:
        raise ValueError("per-request capture was not enabled")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            output.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """Compose the AISimulate base parser with Dynamo-only extensions."""

    parser = argparse.ArgumentParser(
        prog="python -m dynamo.replay",
        description="Run one Dynamo-integrated replay.",
    )
    add_base_replay_arguments(parser)
    parser.add_argument("--router-config")
    parser.add_argument(
        "--router-policy-config",
        default=os.environ.get("DYN_ROUTER_POLICY_CONFIG"),
        help=(
            "startup-only policy-family and cache-bucket queue YAML path; "
            "overrides router_policy_config inside --router-config"
        ),
    )
    parser.add_argument(
        "--model-name",
        help="model profile to select from --router-policy-config",
    )
    parser.add_argument("--aic-backend")
    parser.add_argument("--aic-system")
    parser.add_argument("--aic-backend-version")
    parser.add_argument("--aic-tp-size", type=int)
    parser.add_argument("--aic-model-path")
    parser.add_argument("--aic-moe-tp-size", type=int)
    parser.add_argument("--aic-moe-ep-size", type=int)
    parser.add_argument("--aic-attention-dp-size", type=int)
    parser.add_argument(
        "--aic-gemm-dtype",
        help="dense-GEMM quant mode for the AIC KV-router prefill-load "
        "estimator, e.g. fp8, fp8_block, int4, nvfp4; 'auto'/omitted = model "
        "default. Worker-engine quant is set via --extra-engine-args.",
    )
    parser.add_argument(
        "--aic-moe-dtype",
        help="MoE quant mode for the AIC prefill-load estimator, e.g. fp8, "
        "nvfp4, w4a16_mxfp4; 'auto'/omitted = model default",
    )
    parser.add_argument(
        "--aic-fmha-dtype",
        help="attention (FMHA) quant mode for the AIC prefill-load estimator, "
        "e.g. fp8; 'auto'/omitted = model default",
    )
    parser.add_argument(
        "--aic-kv-cache-dtype",
        help="KV-cache quant mode for the AIC prefill-load estimator, e.g. fp8, "
        "int8; 'auto'/omitted = model default",
    )
    parser.add_argument(
        "--aic-comm-dtype",
        help="communication (collective) quant mode for the AIC prefill-load "
        "estimator, e.g. fp8, int8; 'auto'/omitted = model default",
    )
    parser.add_argument("--aic-nextn", type=int)
    parser.add_argument("--aic-nextn-accept-rates")
    parser.add_argument(
        "--router-mode",
        choices=("round_robin", "kv_router"),
        default="round_robin",
    )
    parser.add_argument(
        "--planner-config",
        help="path to planner config YAML/JSON or inline JSON; enables planner-in-the-loop replay (offline agg/disagg)",
    )
    parser.add_argument(
        "--benchmark-granularity",
        type=int,
        default=8,
        help="number of sweep points for synthetic perf model benchmark (default: 8, matching profiler)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    try:
        base_config = parse_base_replay_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    using_trace_file = base_config.uses_trace
    workload = base_config.workload
    per_request_jsonl = base_config.output.per_request_jsonl

    if (
        per_request_jsonl is not None
        and base_config.replay_mode == "online"
        and not using_trace_file
    ):
        parser.error(
            "--per-request-jsonl with online replay currently only supports trace files"
        )
    if args.max_sim_time_seconds is not None:
        if args.planner_config is not None:
            parser.error(
                "--max-sim-time-seconds is not supported with --planner-config"
            )
    # These two checks exist because defaulting the flags into the engine args
    # moves their validation earlier. Both conditions are already rejected, by
    # normalize_conditional_accept_rates, but that now raises from
    # MockEngineArgs.from_json below, which is outside the error handling that
    # turns a bad value into a usage message.
    if args.aic_nextn_accept_rates is not None and args.aic_nextn is None:
        parser.error("--aic-nextn-accept-rates requires --aic-nextn")
    if args.aic_nextn is not None and not 1 <= args.aic_nextn <= 5:
        parser.error(f"--aic-nextn must be in 1..=5, got {args.aic_nextn}")

    # Speculation is a decode-side concept, so the flags are defaulted into the
    # aggregated or the decode engine args, never prefill: prefill does not
    # speculate and modelling it there would change prefill timing.
    #
    # The aggregated slot has to stay empty for a disaggregated run.
    # validate_replay_args_mode rejects extra_engine_args combined with
    # prefill/decode args, so inventing aggregated args here would abort the run
    # rather than simulate it.
    disagg_engine_args = (
        base_config.prefill_engine_args is not None
        or base_config.decode_engine_args is not None
    )
    extra_raw = (
        base_config.extra_engine_args
        if disagg_engine_args
        else _with_aic_speculation(base_config.extra_engine_args, args)
    )
    extra_engine_args = _load_engine_args(
        json.dumps(extra_raw) if extra_raw is not None else None
    )
    prefill_engine_args = _load_engine_args(
        json.dumps(base_config.prefill_engine_args)
        if base_config.prefill_engine_args is not None
        else None
    )
    # Only when decode args already exist. Creating them from the flags alone
    # would turn "prefill and decode must be provided together" into a
    # different failure.
    decode_raw = (
        _with_aic_speculation(base_config.decode_engine_args, args)
        if base_config.decode_engine_args is not None
        else None
    )
    decode_engine_args = _load_engine_args(
        json.dumps(decode_raw) if decode_raw is not None else None
    )
    router_config = _load_router_config(
        args.router_config,
        args.router_policy_config,
    )
    try:
        aic_perf_config = _load_aic_perf_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    if args.planner_config is not None:
        if args.replay_mode != "offline":
            parser.error("--planner-config only supports --replay-mode=offline")
        if using_trace_file and args.trace_format not in ("mooncake", "dynamo"):
            parser.error(
                "--planner-config only supports --trace-format=mooncake or dynamo"
            )

    capture_per_request = (
        base_config.replay_mode == "offline" and per_request_jsonl is not None
    )
    replay_options = {
        "extra_engine_args": extra_engine_args,
        "prefill_engine_args": prefill_engine_args,
        "decode_engine_args": decode_engine_args,
        "router_config": router_config,
        "aic_perf_config": aic_perf_config,
        "num_workers": base_config.num_workers,
        "num_prefill_workers": base_config.num_prefill_workers,
        "num_decode_workers": base_config.num_decode_workers,
        "replay_concurrency": workload.get("replay_concurrency"),
        "replay_mode": base_config.replay_mode,
        "router_mode": args.router_mode,
        "arrival_speedup_ratio": workload.get("arrival_speedup_ratio", 1.0),
        "model_name": args.model_name,
        "sla_ttft_ms": args.sla_ttft_ms,
        "sla_itl_ms": args.sla_itl_ms,
        "sla_e2e_ms": args.sla_e2e_ms,
        "planner_config": args.planner_config,
        "benchmark_granularity": args.benchmark_granularity,
        "capture_per_request": capture_per_request,
    }

    if using_trace_file:
        report = run_trace_replay(
            list(base_config.trace_files),
            trace_block_size=workload.get("trace_block_size"),
            trace_format=workload.get("trace_format", "mooncake"),
            trace_shared_prefix_ratio=workload.get("trace_shared_prefix_ratio", 0.0),
            trace_num_prefix_groups=workload.get("trace_num_prefix_groups", 0),
            report_jsonl_path=per_request_jsonl,
            max_sim_time_ms=workload.get("max_sim_time_ms"),
            **replay_options,
        )
    else:
        report = run_synthetic_trace_replay(
            workload["isl"],
            workload["osl"],
            workload["request_count"],
            request_rate=workload.get("request_rate"),
            arrival_interval_ms=workload.get("arrival_interval_ms"),
            arrival_seed=workload.get("arrival_seed", 42),
            turns_per_session=workload.get("turns_per_session", 1),
            shared_prefix_ratio=workload.get("shared_prefix_ratio", 0.0),
            num_prefix_groups=workload.get("num_prefix_groups", 0),
            inter_turn_delay_ms=workload.get("inter_turn_delay_ms", 0.0),
            **replay_options,
        )

    if base_config.replay_mode == "online":
        summary = report
        report_payload = report
    else:
        summary = report.summary
        report_payload = report.to_dict()
        if per_request_jsonl is not None and not using_trace_file:
            _write_per_request_jsonl(per_request_jsonl, report.per_request)

    report_path = write_report_json(report_payload, base_config.output.report_json)
    sys.stdout.write(format_report_table(summary))
    sys.stdout.write("\n")
    sys.stdout.write(f"Saved full report to: {report_path}\n")
    if base_config.replay_mode == "offline" and report.planner is not None:
        planner = report.planner
        if planner.scaling_events:
            sys.stdout.write("\nScaling events:\n")
            for event in planner.scaling_events:
                sys.stdout.write(
                    f"  t={event.at_s:.1f}s [{event.component}]: "
                    f"{event.from_count} -> {event.to_count} workers"
                    f" ({event.reason})\n"
                )
        sys.stdout.write(f"Planner ticks: {planner.total_ticks}\n")
        if planner.html_report_path:
            sys.stdout.write(
                f"Planner diagnostics report: {planner.html_report_path}\n"
            )
    return 0
