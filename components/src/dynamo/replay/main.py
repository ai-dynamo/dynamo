# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from dynamo.planner.core.types import EngineCapabilities

from dynamo._internal.aic import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_MEM_FRACTION_STATIC,
    estimate_num_gpu_blocks,
)
from dynamo.common.forward_pass_metrics import (
    ForwardPassMetrics,
    ScheduledRequestMetrics,
)
from dynamo.llm import AicPerfConfig, KvRouterConfig
from dynamo.mocker import MockEngineArgs
from dynamo.mocker.utils.kv_cache import compute_kv_bytes_per_token
from dynamo.replay import run_synthetic_trace_replay, run_trace_replay
from dynamo.replay.reporting import format_report_table, write_report_json


class PlannerProfileDataResult(Protocol):
    npz_path: Path | None


_DEFAULT_AIC_SYSTEM = "h200_sxm"
_DEFAULT_MAX_NUM_BATCHED_TOKENS = 8192
_DEFAULT_VLLM_BLOCK_SIZE = 64


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


_DEFAULT_SGLANG_BLOCK_SIZE = 1
_DEFAULT_TRTLLM_BLOCK_SIZE = 32


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


def _resolve_block_size_for_capacity(raw: dict) -> int:
    block_size = raw.get("block_size")
    if block_size is not None:
        return cast(int, block_size)
    if raw.get("engine_type") == "sglang":
        sglang = raw.get("sglang")
        if isinstance(sglang, dict) and sglang.get("page_size") is not None:
            return cast(int, sglang["page_size"])
        return _DEFAULT_SGLANG_BLOCK_SIZE
    if raw.get("engine_type") == "trtllm":
        return _DEFAULT_TRTLLM_BLOCK_SIZE
    return _DEFAULT_VLLM_BLOCK_SIZE


def _resolve_aic_num_gpu_blocks(raw: dict) -> None:
    if raw.get("num_gpu_blocks") is not None:
        return

    aic_backend = raw.get("aic_backend")
    if aic_backend is None:
        return

    aic_model_path = raw.get("aic_model_path")
    if not aic_model_path:
        raise ValueError(
            "AIC KV cache capacity estimation requires aic_model_path in engine args"
        )

    tp_size = raw.get("aic_tp_size")
    max_num_batched_tokens = raw.get("max_num_batched_tokens")
    gpu_memory_utilization = raw.get("gpu_memory_utilization")
    mem_fraction_static = raw.get("mem_fraction_static")
    free_gpu_memory_fraction = raw.get("free_gpu_memory_fraction")

    raw["num_gpu_blocks"] = estimate_num_gpu_blocks(
        backend_name=aic_backend,
        system=raw.get("aic_system") or _DEFAULT_AIC_SYSTEM,
        model_path=aic_model_path,
        tp_size=cast(int, tp_size if tp_size is not None else 1),
        block_size=_resolve_block_size_for_capacity(raw),
        max_num_batched_tokens=cast(
            int,
            max_num_batched_tokens
            if max_num_batched_tokens is not None
            else _DEFAULT_MAX_NUM_BATCHED_TOKENS,
        ),
        gpu_memory_utilization=cast(
            float,
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else DEFAULT_GPU_MEMORY_UTILIZATION,
        ),
        mem_fraction_static=cast(
            float,
            mem_fraction_static
            if mem_fraction_static is not None
            else DEFAULT_MEM_FRACTION_STATIC,
        ),
        # None -> aic.py applies the TRT-LLM default (0.9).
        free_gpu_memory_fraction=free_gpu_memory_fraction,
        backend_version=raw.get("aic_backend_version"),
        moe_tp_size=raw.get("aic_moe_tp_size"),
        moe_ep_size=raw.get("aic_moe_ep_size"),
        attention_dp_size=raw.get("aic_attention_dp_size"),
    )


def _resolve_kv_bytes_per_token(raw: dict) -> None:
    if raw.get("kv_bytes_per_token") is not None:
        return

    offload_requested = (
        any(
            isinstance(raw.get(name), int) and raw[name] > 0
            for name in ("num_g2_blocks", "num_g3_blocks")
        )
        or raw.get("enable_g4_storage") is True
    )
    if not offload_requested:
        return

    model_path = raw.get("aic_model_path")
    if not model_path:
        return

    kv_bytes_per_token = compute_kv_bytes_per_token(model_path, "auto")
    if kv_bytes_per_token is not None:
        raw["kv_bytes_per_token"] = kv_bytes_per_token


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
    _resolve_kv_bytes_per_token(raw)
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
        aic_nextn=values["aic_nextn"],
        aic_nextn_accept_rates=values["aic_nextn_accept_rates"],
    )


def _engine_caps(args: MockEngineArgs) -> EngineCapabilities:
    """Derive EngineCapabilities from MockEngineArgs."""
    from dynamo.planner.core.types import EngineCapabilities

    max_kv_tokens = args.num_gpu_blocks * args.block_size
    return EngineCapabilities(
        num_gpu=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        context_length=max_kv_tokens if max_kv_tokens > 0 else None,
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


@dataclass
class SyntheticWorkload:
    """A synthetic workload for planner replay: ``request_count`` sessions of fixed
    ``input_tokens``/``output_tokens``. ``turns_per_session`` > 1 makes each session
    multi-turn (total requests = ``request_count * turns_per_session``).
    ``shared_prefix_ratio`` / ``num_prefix_groups`` control prefix-cache sharing."""

    input_tokens: int
    output_tokens: int
    request_count: int
    arrival_interval_ms: float = 1.0
    turns_per_session: int = 1
    shared_prefix_ratio: float = 0.0
    num_prefix_groups: int = 0
    inter_turn_delay_ms: float = 0.0


def _run_planner_replay(
    trace_file: str | None,
    extra_engine_args: MockEngineArgs | None,
    prefill_engine_args: MockEngineArgs | None,
    decode_engine_args: MockEngineArgs | None,
    router_config: KvRouterConfig | None,
    num_workers: int,
    num_prefill_workers: int,
    num_decode_workers: int,
    router_mode: str,
    arrival_speedup_ratio: float,
    trace_block_size: int,
    planner_config_arg: str,
    model_name: str | None = None,
    benchmark_granularity: int = 8,
    sla_ttft_ms: float | None = None,
    sla_itl_ms: float | None = None,
    sla_e2e_ms: float | None = None,
    replay_concurrency: int | None = None,
    synthetic: SyntheticWorkload | None = None,
):
    """Run an offline replay with planner-in-the-loop (agg or disagg).

    ``sla_ttft_ms`` / ``sla_itl_ms`` / ``sla_e2e_ms`` are the **goodput** SLA used
    to classify SLA-satisfying requests in the report. They are independent of
    the planner's own scaling SLA in ``planner_config`` and are never read from
    it — pass whatever goodput threshold the caller wants measured.

    # TODO(jthomson04): SLA-based scaling (optimization_target="sla") with
    # disagg mode requires planner_profile_data (NPZ) or AIC-backed engine
    # args.  The default polynomial perf model does not account for batch
    # size in its decode timing, causing the DecodeRegressionModel's
    # num_decode_requests coefficient to go negative and reject the fit.
    # Fix the polynomial model to incorporate batch_size, or gate disagg
    # SLA mode on having a non-polynomial perf model.
    """
    from dynamo.mocker import PlannerReplayBridge
    from dynamo.planner.config.planner_config import PlannerConfig
    from dynamo.planner.core.types import WorkerCapabilities
    from dynamo.planner.offline.replay_adapter import ReplayPlannerAdapter

    planner_config = PlannerConfig.from_config_arg(planner_config_arg)
    planner_config.advisory = True

    if (trace_file is None) == (synthetic is None):
        raise ValueError(
            "planner replay requires exactly one of trace_file or synthetic"
        )

    if planner_config.mode == "agg":
        if extra_engine_args is None:
            extra_engine_args = MockEngineArgs()
        if synthetic is not None:
            bridge = PlannerReplayBridge.from_synthetic(
                input_tokens=synthetic.input_tokens,
                output_tokens=synthetic.output_tokens,
                request_count=synthetic.request_count,
                extra_engine_args=extra_engine_args,
                num_workers=num_workers,
                router_mode=router_mode,
                router_config=router_config,
                model_name=model_name,
                replay_concurrency=replay_concurrency,
                arrival_speedup_ratio=arrival_speedup_ratio,
                arrival_interval_ms=synthetic.arrival_interval_ms,
                turns_per_session=synthetic.turns_per_session,
                shared_prefix_ratio=synthetic.shared_prefix_ratio,
                num_prefix_groups=synthetic.num_prefix_groups,
                inter_turn_delay_ms=synthetic.inter_turn_delay_ms,
                sla_ttft_ms=sla_ttft_ms,
                sla_itl_ms=sla_itl_ms,
                sla_e2e_ms=sla_e2e_ms,
            )
        else:
            if trace_file is None:  # guaranteed by the trace/synthetic check above
                raise ValueError("planner replay needs trace_file in trace mode")
            bridge = PlannerReplayBridge(
                trace_file=trace_file,
                extra_engine_args=extra_engine_args,
                num_workers=num_workers,
                router_mode=router_mode,
                router_config=router_config,
                model_name=model_name,
                arrival_speedup_ratio=arrival_speedup_ratio,
                trace_block_size=trace_block_size,
                replay_concurrency=replay_concurrency,
                sla_ttft_ms=sla_ttft_ms,
                sla_itl_ms=sla_itl_ms,
                sla_e2e_ms=sla_e2e_ms,
            )
        capabilities = WorkerCapabilities(decode=_engine_caps(extra_engine_args))

    elif planner_config.mode == "disagg":
        if prefill_engine_args is None or decode_engine_args is None:
            raise ValueError(
                "disagg planner replay requires --prefill-engine-args and --decode-engine-args"
            )
        if synthetic is not None:
            bridge = PlannerReplayBridge.from_synthetic_disagg(
                input_tokens=synthetic.input_tokens,
                output_tokens=synthetic.output_tokens,
                request_count=synthetic.request_count,
                prefill_engine_args=prefill_engine_args,
                decode_engine_args=decode_engine_args,
                num_prefill_workers=num_prefill_workers,
                num_decode_workers=num_decode_workers,
                router_mode=router_mode,
                router_config=router_config,
                model_name=model_name,
                replay_concurrency=replay_concurrency,
                arrival_speedup_ratio=arrival_speedup_ratio,
                arrival_interval_ms=synthetic.arrival_interval_ms,
                turns_per_session=synthetic.turns_per_session,
                shared_prefix_ratio=synthetic.shared_prefix_ratio,
                num_prefix_groups=synthetic.num_prefix_groups,
                inter_turn_delay_ms=synthetic.inter_turn_delay_ms,
                sla_ttft_ms=sla_ttft_ms,
                sla_itl_ms=sla_itl_ms,
                sla_e2e_ms=sla_e2e_ms,
            )
        else:
            if trace_file is None:  # guaranteed by the trace/synthetic check above
                raise ValueError("planner replay needs trace_file in trace mode")
            bridge = PlannerReplayBridge.create_disagg(
                trace_file=trace_file,
                prefill_engine_args=prefill_engine_args,
                decode_engine_args=decode_engine_args,
                num_prefill_workers=num_prefill_workers,
                num_decode_workers=num_decode_workers,
                router_mode=router_mode,
                router_config=router_config,
                model_name=model_name,
                arrival_speedup_ratio=arrival_speedup_ratio,
                trace_block_size=trace_block_size,
                replay_concurrency=replay_concurrency,
                sla_ttft_ms=sla_ttft_ms,
                sla_itl_ms=sla_itl_ms,
                sla_e2e_ms=sla_e2e_ms,
            )
        capabilities = WorkerCapabilities(
            prefill=_engine_caps(prefill_engine_args),
            decode=_engine_caps(decode_engine_args),
        )

    else:
        raise ValueError(
            f"planner-in-the-loop replay supports mode='agg' or 'disagg', got '{planner_config.mode}'"
        )

    adapter = ReplayPlannerAdapter(
        planner_config=planner_config,
        bridge=bridge,
        capabilities=capabilities,
    )

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

                aic_session = create_session(
                    backend_name=aic_backend,
                    system=ref_args.aic_system,
                    model_path=ref_args.aic_model_path,
                    tp_size=ref_args.aic_tp_size or 1,
                    backend_version=ref_args.aic_backend_version,
                    moe_tp_size=ref_args.aic_moe_tp_size,
                    moe_ep_size=ref_args.aic_moe_ep_size,
                    attention_dp_size=ref_args.aic_attention_dp_size,
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

                if planner_config.mode == "agg":
                    # Agg regression fits on (sum_prefill_tokens, sum_decode_kv_tokens);
                    # combine prefill-only and decode-only points so both features
                    # have variance.
                    agg_fpms = prefill_fpms + decode_fpms
                    if agg_fpms:
                        adapter.install_benchmark_fpms(agg_fpms=agg_fpms)
                    else:
                        sys.stderr.write(
                            "Warning: AIC produced no agg benchmark FPMs\n"
                        )
                else:
                    if prefill_fpms and decode_fpms:
                        adapter.install_benchmark_fpms(
                            prefill_fpms=prefill_fpms, decode_fpms=decode_fpms
                        )
                    else:
                        sys.stderr.write(
                            f"Warning: AIC produced empty benchmark FPMs "
                            f"(prefill={len(prefill_fpms)}, decode={len(decode_fpms)})\n"
                        )

    # gpu_hours (and prefill/decode_gpus_per_worker) are computed in the mocker
    # from its own worker parallelism (aic_tp x aic_attention_dp) and ride the
    # report dict — no per-engine GPU count from the planner config is needed.
    return adapter.run()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dynamo.replay")
    parser.add_argument("trace_file", nargs="?")
    parser.add_argument("--extra-engine-args")
    parser.add_argument("--prefill-engine-args")
    parser.add_argument("--decode-engine-args")
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
    parser.add_argument("--aic-nextn", type=int)
    parser.add_argument("--aic-nextn-accept-rates")
    parser.add_argument("--input-tokens", type=int)
    parser.add_argument("--output-tokens", type=int)
    parser.add_argument(
        "--request-count",
        type=int,
        help="number of synthetic requests; when --turns-per-session > 1, this is the number of sessions",
    )
    parser.add_argument("--arrival-interval-ms", type=float, default=1.0)
    parser.add_argument("--turns-per-session", type=int, default=1)
    parser.add_argument("--shared-prefix-ratio", type=float, default=0.0)
    parser.add_argument("--num-prefix-groups", type=int, default=0)
    parser.add_argument("--inter-turn-delay-ms", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-prefill-workers", type=int, default=1)
    parser.add_argument("--num-decode-workers", type=int, default=1)
    parser.add_argument("--replay-concurrency", type=int)
    parser.add_argument(
        "--replay-mode",
        choices=("offline", "online"),
        default="offline",
    )
    parser.add_argument(
        "--router-mode",
        choices=("round_robin", "kv_router"),
        default="round_robin",
    )
    parser.add_argument("--arrival-speedup-ratio", type=float, default=1.0)
    parser.add_argument(
        "--trace-format",
        choices=(
            "mooncake",
            "mooncake-delta",
            "agentic_mooncake",
            "applied_compute_agentic",
        ),
        default="mooncake",
        help=(
            "format of trace_file when replaying from a file; mooncake-delta "
            "accumulates per-session input deltas into cumulative prompts and "
            "can use substantially more memory than mooncake; agentic_mooncake "
            "replays request-level workflow dependencies"
        ),
    )
    parser.add_argument(
        "--trace-block-size",
        type=int,
        default=512,
        help="tokens represented by each hash_id in the trace file; only used for file replay",
    )
    parser.add_argument(
        "--trace-shared-prefix-ratio",
        type=float,
        default=0.0,
        help="fraction of the initial prompt blocks to share across sessions for applied_compute_agentic trace replay",
    )
    parser.add_argument(
        "--trace-num-prefix-groups",
        type=int,
        default=0,
        help="number of cross-session shared-prefix groups for applied_compute_agentic trace replay",
    )
    parser.add_argument(
        "--report-json",
        help="path to save the full replay report JSON; defaults to a timestamped file in the current directory",
    )
    parser.add_argument(
        "--report-jsonl",
        default=None,
        help="optional path to emit one JSON object per request (offline disagg replay only). "
        "Useful for per-request analysis (TTFT vs ISL scatter, ITL trace per request, "
        "worker-residency analysis). Each line carries arrival/admit/token timestamps, "
        "input/output lengths, full ITL series, and prefill/decode worker indices "
        "(prefill_worker_idx=None indicates a conditional-prefill bypass).",
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
    parser.add_argument(
        "--max-sim-time-seconds",
        type=float,
        default=None,
        help="optional cap on simulated wall-clock duration for offline replay (disagg and agg); when set, replay stops once the simulated clock would exceed this many seconds, leaving in-flight requests as incomplete in the report",
    )
    # Goodput SLA — the threshold a request must meet to count toward goodput in
    # the report. This is INDEPENDENT of the planner's own scaling SLA (in
    # --planner-config); the two can hold different values and are never read
    # from each other. Set --sla-ttft-ms + --sla-itl-ms, or --sla-e2e-ms alone.
    parser.add_argument(
        "--sla-ttft-ms",
        type=float,
        default=None,
        help="goodput SLA: max time-to-first-token (ms). Independent of the planner's scaling SLA. "
        "When an SLA is set, the report adds goodput_* (SLA-satisfying throughput).",
    )
    parser.add_argument(
        "--sla-itl-ms",
        type=float,
        default=None,
        help="goodput SLA: max average inter-token latency (ms), per request (e2e - ttft)/(osl - 1) "
        "(aiperf definition). Independent of the planner's scaling SLA.",
    )
    parser.add_argument(
        "--sla-e2e-ms",
        type=float,
        default=None,
        help="goodput SLA: max end-to-end latency (ms); use instead of --sla-ttft-ms/--sla-itl-ms. "
        "Independent of the planner's scaling SLA.",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    using_trace_file = args.trace_file is not None
    synthetic_args = (args.input_tokens, args.output_tokens, args.request_count)
    using_synthetic = any(value is not None for value in synthetic_args) or any(
        (
            args.turns_per_session != 1,
            args.shared_prefix_ratio != 0.0,
            args.num_prefix_groups != 0,
            args.inter_turn_delay_ms != 0.0,
        )
    )

    if using_trace_file == using_synthetic:
        parser.error(
            "provide either trace_file or all of --input-tokens/--output-tokens/--request-count"
        )
    if using_synthetic and not all(value is not None for value in synthetic_args):
        parser.error(
            "synthetic replay requires --input-tokens, --output-tokens, and --request-count"
        )
    if (
        using_trace_file
        and args.trace_format == "applied_compute_agentic"
        and args.replay_concurrency is None
    ):
        parser.error(
            "--trace-format=applied_compute_agentic requires --replay-concurrency because the source traces do not include first-turn timestamps"
        )

    if args.report_jsonl is not None:
        if args.replay_mode != "offline":
            parser.error("--report-jsonl only supports --replay-mode=offline")
        if args.planner_config is not None:
            parser.error("--report-jsonl is not supported with --planner-config")
        if not using_trace_file:
            parser.error("--report-jsonl currently only supports trace-file replay")
    if args.max_sim_time_seconds is not None:
        if args.planner_config is not None:
            parser.error(
                "--max-sim-time-seconds is not supported with --planner-config"
            )
        if not using_trace_file:
            parser.error(
                "--max-sim-time-seconds currently only supports trace-file replay"
            )
    if (
        any(v is not None for v in (args.sla_ttft_ms, args.sla_itl_ms, args.sla_e2e_ms))
        and args.planner_config is None
    ):
        parser.error(
            "goodput SLA (--sla-ttft-ms/--sla-itl-ms/--sla-e2e-ms) currently requires --planner-config"
        )

    extra_engine_args = _load_engine_args(args.extra_engine_args)
    prefill_engine_args = _load_engine_args(args.prefill_engine_args)
    decode_engine_args = _load_engine_args(args.decode_engine_args)
    router_config = _load_router_config(
        args.router_config,
        args.router_policy_config,
    )
    try:
        aic_perf_config = _load_aic_perf_config(args)
    except ValueError as exc:
        parser.error(str(exc))

    # Planner-in-the-loop mode
    if args.planner_config is not None:
        if args.replay_mode != "offline":
            parser.error("--planner-config only supports --replay-mode=offline")
        if using_trace_file and args.trace_format != "mooncake":
            parser.error("--planner-config only supports --trace-format=mooncake")

        synthetic = None
        if not using_trace_file:
            synthetic = SyntheticWorkload(
                input_tokens=args.input_tokens,
                output_tokens=args.output_tokens,
                request_count=args.request_count,
                arrival_interval_ms=args.arrival_interval_ms,
                turns_per_session=args.turns_per_session,
                shared_prefix_ratio=args.shared_prefix_ratio,
                num_prefix_groups=args.num_prefix_groups,
                inter_turn_delay_ms=args.inter_turn_delay_ms,
            )

        planner_report = _run_planner_replay(
            trace_file=args.trace_file if using_trace_file else None,
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            router_config=router_config,
            num_workers=args.num_workers,
            num_prefill_workers=args.num_prefill_workers,
            num_decode_workers=args.num_decode_workers,
            router_mode=args.router_mode,
            arrival_speedup_ratio=args.arrival_speedup_ratio,
            trace_block_size=args.trace_block_size,
            planner_config_arg=args.planner_config,
            model_name=args.model_name,
            benchmark_granularity=args.benchmark_granularity,
            sla_ttft_ms=args.sla_ttft_ms,
            sla_itl_ms=args.sla_itl_ms,
            sla_e2e_ms=args.sla_e2e_ms,
            replay_concurrency=args.replay_concurrency,
            synthetic=synthetic,
        )
        report = planner_report.trace_report
        if planner_report.scaling_events:
            sys.stdout.write("\nScaling events:\n")
            for event in planner_report.scaling_events:
                sys.stdout.write(
                    f"  t={event.at_s:.1f}s [{event.component}]: "
                    f"{event.from_count} -> {event.to_count} workers"
                    f" ({event.reason})\n"
                )
        report_path = write_report_json(report, args.report_json)
        sys.stdout.write(format_report_table(report))
        sys.stdout.write("\n")
        sys.stdout.write(f"Saved full report to: {report_path}\n")
        sys.stdout.write(f"Planner ticks: {planner_report.total_ticks}\n")
        if planner_report.html_report_path:
            sys.stdout.write(
                f"Planner diagnostics report: {planner_report.html_report_path}\n"
            )
        return 0

    if using_trace_file:
        max_sim_time_ms = (
            args.max_sim_time_seconds * 1_000.0
            if args.max_sim_time_seconds is not None
            else None
        )
        report = run_trace_replay(
            args.trace_file,
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            router_config=router_config,
            aic_perf_config=aic_perf_config,
            num_workers=args.num_workers,
            num_prefill_workers=args.num_prefill_workers,
            num_decode_workers=args.num_decode_workers,
            replay_concurrency=args.replay_concurrency,
            replay_mode=args.replay_mode,
            router_mode=args.router_mode,
            arrival_speedup_ratio=args.arrival_speedup_ratio,
            trace_block_size=args.trace_block_size,
            trace_format=args.trace_format,
            trace_shared_prefix_ratio=args.trace_shared_prefix_ratio,
            trace_num_prefix_groups=args.trace_num_prefix_groups,
            report_jsonl_path=args.report_jsonl,
            max_sim_time_ms=max_sim_time_ms,
            model_name=args.model_name,
        )
    else:
        report = run_synthetic_trace_replay(
            args.input_tokens,
            args.output_tokens,
            args.request_count,
            extra_engine_args=extra_engine_args,
            prefill_engine_args=prefill_engine_args,
            decode_engine_args=decode_engine_args,
            router_config=router_config,
            aic_perf_config=aic_perf_config,
            num_workers=args.num_workers,
            num_prefill_workers=args.num_prefill_workers,
            num_decode_workers=args.num_decode_workers,
            replay_concurrency=args.replay_concurrency,
            replay_mode=args.replay_mode,
            router_mode=args.router_mode,
            arrival_speedup_ratio=args.arrival_speedup_ratio,
            arrival_interval_ms=args.arrival_interval_ms,
            turns_per_session=args.turns_per_session,
            shared_prefix_ratio=args.shared_prefix_ratio,
            num_prefix_groups=args.num_prefix_groups,
            inter_turn_delay_ms=args.inter_turn_delay_ms,
            model_name=args.model_name,
        )

    report_path = write_report_json(report, args.report_json)
    sys.stdout.write(format_report_table(report))
    sys.stdout.write("\n")
    sys.stdout.write(f"Saved full report to: {report_path}\n")
    return 0
