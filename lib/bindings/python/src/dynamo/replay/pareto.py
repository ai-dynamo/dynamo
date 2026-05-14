# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Mocker-based parallelism + batch-size sweep producing a Pareto front.

For each (parallelism config, batch_size) point that AIC's enumeration would
produce for a given (model, system, backend, total_gpus), this script runs
dynamo's mock engine (`run_synthetic_trace_replay`) with AIC-driven timing,
collects throughput / latency, and writes the non-dominated frontier on
(tokens/s/user, tokens/s/gpu) axes.

The CLI surface mirrors `aiconfigurator cli default` to make side-by-side
comparison with AIC straightforward; the companion script
`benchmarks/mocker/pareto_comparison.py` runs both AIC and mocker and plots
the two Pareto curves together.

Usage::

    python -m dynamo.replay.pareto \
        --model moonshotai/Kimi-K2.5 \
        --system b200_sxm \
        --backend vllm \
        --backend-version 0.19.0 \
        --total-gpus 8 \
        --isl 8192 --osl 1024 \
        --ttft 1000 --tpot 50 \
        --save-dir results/kimi_b200

Outputs in ``--save-dir``:
    raw_mocker.csv      every evaluated (parallelism, bs) point
    pareto_mocker.csv   Pareto-front subset on (tokens/s/user, tokens/s/gpu)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from aiconfigurator.sdk import common
from aiconfigurator.sdk.pareto_analysis import get_pareto_front
from aiconfigurator.sdk.utils import enumerate_parallel_config

from dynamo.llm import MockEngineArgs
from dynamo.replay import run_synthetic_trace_replay

# v1 agg batch-size sweep: powers of 2 up to 256.
# Coarse subset of AIC's internal `b_list_default` (~45 values; see
# vllm_backend.py:474, trtllm_backend.py:548, sglang_backend.py:474).
# We stop at 256 because mocker's per-config wall time scales with
# bs × num_requests (where num_requests = max(50, 5*bs)); at bs=512 a single
# config can take many minutes to simulate. With --workers > 1 the high-bs
# tier parallelizes naturally. Tradeoffs documented in
# benchmarks/mocker/README.md.
BATCH_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Disagg sweep grids. Kept tight in v1 to bound wall time — the disagg cross
# product (prefill_parallel × decode_parallel × prefill_bs × decode_bs ×
# prefill_workers × decode_workers) explodes quickly. Prefill is typically
# small-batch / latency-bound; decode benefits from larger batches.
# Empirically on Kimi-K2.5 / b200_sxm / 8 GPUs these tight caps yield
# ~150-300 viable points after the total_gpus filter, ~10-15 min wall time
# with --workers 6. Loosen if you want denser coverage.
DISAGG_PREFILL_BS: tuple[int, ...] = (1, 2, 4, 8)
DISAGG_DECODE_BS: tuple[int, ...] = (32, 64, 128)
DISAGG_PREFILL_WORKERS: tuple[int, ...] = (1,)
DISAGG_DECODE_WORKERS: tuple[int, ...] = (1, 2, 4)


@dataclass(frozen=True)
class ParallelConfig:
    tp: int
    pp: int
    dp: int
    moe_tp: int
    moe_ep: int

    @property
    def num_gpus(self) -> int:
        return self.tp * self.pp * self.dp


@dataclass(frozen=True)
class DisaggPoint:
    prefill: ParallelConfig
    decode: ParallelConfig
    prefill_bs: int
    decode_bs: int
    prefill_workers: int
    decode_workers: int

    @property
    def total_gpus(self) -> int:
        return (
            self.prefill.num_gpus * self.prefill_workers
            + self.decode.num_gpus * self.decode_workers
        )


_MOE_KEYS = (
    "num_experts",
    "num_local_experts",
    "moe_num_experts",
    "num_routed_experts",
    "n_routed_experts",
    "moe_intermediate_size",
    "moe_layer_freq",
    "num_experts_per_tok",
)


def _detect_is_moe(model_path: str) -> bool:
    """Detect MoE from a model's HF config.json (local dir or HF id).

    Walks nested dicts because multimodal models (e.g., Kimi-K2.5) put the
    MoE-relevant keys under a `text_config` sub-block, not at the top level.
    """
    cfg_path = Path(model_path) / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
    else:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(repo_id=model_path, filename="config.json")
        cfg = json.loads(Path(local).read_text())

    def _walk(node: object) -> bool:
        if isinstance(node, dict):
            if any(k in node for k in _MOE_KEYS):
                return True
            return any(_walk(v) for v in node.values())
        return False

    return _walk(cfg)


def _derive_parallel_grid(
    total_gpus: int,
    system: str,
    backend: str,
    is_moe: bool,
    enable_wideep: bool,
) -> list[ParallelConfig]:
    """Build the parallelism grid via AIC's enumerator with backend-aware lists."""
    if system in ("gb200", "gb300") and not is_moe:
        tp_list = [1, 2, 4, 8, 16]
        num_gpu_per_worker = [1, 2, 4, 8, 16]
    else:
        tp_list = [1, 2, 4, 8]
        num_gpu_per_worker = [1, 2, 4, 8]

    pp_list = [1]
    dp_list = [1, 2, 4, 8] if is_moe else [1]
    moe_tp_list = [1, 2, 4, 8] if is_moe else [1]
    moe_ep_list = [1, 2, 4, 8] if is_moe else [1]

    if is_moe and enable_wideep and backend in ("trtllm", "sglang"):
        num_gpu_per_worker = [2, 4, 8, 16, 32, 64]
        dp_list = [2, 4, 8, 16, 32, 64]
        moe_tp_list = [1]
        moe_ep_list = [2, 4, 8, 16, 32, 64]

    configs = enumerate_parallel_config(
        num_gpu_list=num_gpu_per_worker,
        tp_list=tp_list,
        pp_list=pp_list,
        dp_list=dp_list,
        moe_tp_list=moe_tp_list,
        moe_ep_list=moe_ep_list,
        is_moe=is_moe,
        backend=common.BackendName(backend),
        enable_wideep=enable_wideep,
        real_silicon_sweep=True,
        max_num_gpus=total_gpus,
    )
    return [
        ParallelConfig(tp=c[0], pp=c[1], dp=c[2], moe_tp=c[3], moe_ep=c[4])
        for c in configs
    ]


def _build_engine_args(
    parallel_cfg: ParallelConfig,
    bs: int,
    *,
    backend: str,
    backend_version: str | None,
    system: str,
    model_path: str,
    worker_type: str | None = None,
) -> dict:
    """Build the MockEngineArgs JSON for one (parallelism, bs) point.

    ``worker_type`` is set for disagg points only (``"prefill"`` or
    ``"decode"``); leave None for agg.
    """
    # MockEngineArgs.engine_type accepts only vllm/sglang; trtllm sources use vllm
    # as a structural placeholder while aic_backend="trtllm" drives the timing model.
    engine_type = "sglang" if backend == "sglang" else "vllm"

    args: dict = {
        "engine_type": engine_type,
        "max_num_seqs": bs,
        "max_num_batched_tokens": max(bs * 2, 8192),
        "enable_prefix_caching": False,
        "block_size": 64 if engine_type == "sglang" else 16,
        # Mocker requires dp_size=1 in MockEngineArgs even when the deployment
        # has DP>1; deployment DP is carried via num-workers and aic_attention_dp_size.
        "dp_size": 1,
        "aic_backend": backend,
        "aic_system": system,
        "aic_model_path": model_path,
        "aic_tp_size": parallel_cfg.tp,
    }
    if worker_type is not None:
        args["worker_type"] = worker_type
    if backend_version is not None:
        args["aic_backend_version"] = backend_version
    # Always populate the parallelism fields explicitly (even when ==1).
    # Mocker's AIC callback multiplies these internally; leaving any as None
    # crashes with `TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'`.
    args["aic_attention_dp_size"] = parallel_cfg.dp
    args["aic_moe_tp_size"] = parallel_cfg.moe_tp
    args["aic_moe_ep_size"] = parallel_cfg.moe_ep
    return args


def _evaluate_one(
    parallel_cfg: ParallelConfig,
    bs: int,
    *,
    isl: int,
    osl: int,
    backend: str,
    backend_version: str | None,
    system: str,
    model_path: str,
) -> dict:
    """Run mocker for one (parallelism, bs) point. Returns a flat metrics dict."""
    engine_args = _build_engine_args(
        parallel_cfg,
        bs,
        backend=backend,
        backend_version=backend_version,
        system=system,
        model_path=model_path,
    )
    num_requests = max(50, 5 * bs)

    base_row = {
        "tp": parallel_cfg.tp,
        "pp": parallel_cfg.pp,
        "dp": parallel_cfg.dp,
        "moe_tp": parallel_cfg.moe_tp,
        "moe_ep": parallel_cfg.moe_ep,
        "bs": bs,
        "num_gpus": parallel_cfg.num_gpus,
    }

    try:
        report = run_synthetic_trace_replay(
            isl,
            osl,
            num_requests,
            extra_engine_args=MockEngineArgs.from_json(json.dumps(engine_args)),
            num_workers=max(parallel_cfg.dp, 1),
            replay_concurrency=bs,
            replay_mode="offline",
            router_mode="kv_router" if parallel_cfg.dp > 1 else "round_robin",
        )
    except Exception as exc:
        return {
            **base_row,
            "ttft_ms": None,
            "tpot_ms": None,
            "request_latency_ms": None,
            "tokens/s/user": None,
            "tokens/s/gpu": None,
            "output_throughput_tok_s": None,
            "completed_requests": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    num_gpus = parallel_cfg.num_gpus
    output_throughput = report.get("output_throughput_tok_s") or 0.0
    return {
        **base_row,
        "ttft_ms": report.get("mean_ttft_ms"),
        "tpot_ms": report.get("mean_itl_ms"),
        "request_latency_ms": report.get("mean_e2e_latency_ms"),
        "tokens/s/user": report.get("mean_output_token_throughput_per_user"),
        "tokens/s/gpu": (output_throughput / num_gpus) if num_gpus > 0 else None,
        "output_throughput_tok_s": output_throughput,
        "completed_requests": report.get("completed_requests"),
        "error": None,
    }


def _describe_task(parallel_cfg: ParallelConfig, bs: int) -> str:
    return (
        f"tp={parallel_cfg.tp} pp={parallel_cfg.pp} "
        f"dp={parallel_cfg.dp} moe_tp={parallel_cfg.moe_tp} "
        f"moe_ep={parallel_cfg.moe_ep} bs={bs}"
    )


def _describe_disagg_point(point: DisaggPoint) -> str:
    return (
        f"P[tp={point.prefill.tp} dp={point.prefill.dp} "
        f"moe_tp={point.prefill.moe_tp} moe_ep={point.prefill.moe_ep} "
        f"bs={point.prefill_bs} w={point.prefill_workers}] "
        f"D[tp={point.decode.tp} dp={point.decode.dp} "
        f"moe_tp={point.decode.moe_tp} moe_ep={point.decode.moe_ep} "
        f"bs={point.decode_bs} w={point.decode_workers}] "
        f"gpus={point.total_gpus}"
    )


def _derive_disagg_grid(
    grid: list[ParallelConfig], total_gpus: int
) -> list[DisaggPoint]:
    """Enumerate viable disagg (prefill, decode, workers, bs) combinations.

    Invariants enforced (v1, keeps the cross-product manageable):
      * total GPUs used ≤ ``total_gpus``
      * ``decode.num_gpus * decode_workers >= prefill.num_gpus * prefill_workers``
        — decode is the throughput bottleneck in disagg; configurations
        where decode has fewer GPUs than prefill are rarely interesting
        and explode the search space without adding insight.
    """
    points: list[DisaggPoint] = []
    for p in grid:
        for pw in DISAGG_PREFILL_WORKERS:
            p_gpus = p.num_gpus * pw
            if p_gpus > total_gpus:
                continue
            remaining = total_gpus - p_gpus
            for d in grid:
                for dw in DISAGG_DECODE_WORKERS:
                    d_gpus = d.num_gpus * dw
                    if d_gpus > remaining:
                        continue
                    if d_gpus < p_gpus:
                        continue
                    for pbs in DISAGG_PREFILL_BS:
                        for dbs in DISAGG_DECODE_BS:
                            points.append(
                                DisaggPoint(
                                    prefill=p,
                                    decode=d,
                                    prefill_bs=pbs,
                                    decode_bs=dbs,
                                    prefill_workers=pw,
                                    decode_workers=dw,
                                )
                            )
    return points


def _disagg_base_row(point: DisaggPoint) -> dict:
    return {
        "p_tp": point.prefill.tp,
        "p_pp": point.prefill.pp,
        "p_dp": point.prefill.dp,
        "p_moe_tp": point.prefill.moe_tp,
        "p_moe_ep": point.prefill.moe_ep,
        "p_bs": point.prefill_bs,
        "p_workers": point.prefill_workers,
        "d_tp": point.decode.tp,
        "d_pp": point.decode.pp,
        "d_dp": point.decode.dp,
        "d_moe_tp": point.decode.moe_tp,
        "d_moe_ep": point.decode.moe_ep,
        "d_bs": point.decode_bs,
        "d_workers": point.decode_workers,
        "total_gpus": point.total_gpus,
    }


def _evaluate_one_disagg(
    point: DisaggPoint,
    *,
    isl: int,
    osl: int,
    backend: str,
    backend_version: str | None,
    system: str,
    model_path: str,
) -> dict:
    """Run mocker for one disagg point. Returns a flat metrics dict."""
    p_args = _build_engine_args(
        point.prefill,
        point.prefill_bs,
        backend=backend,
        backend_version=backend_version,
        system=system,
        model_path=model_path,
        worker_type="prefill",
    )
    d_args = _build_engine_args(
        point.decode,
        point.decode_bs,
        backend=backend,
        backend_version=backend_version,
        system=system,
        model_path=model_path,
        worker_type="decode",
    )
    # Drive enough in-flight load to saturate the decode pool. num_requests
    # scales with the heavier (decode) side per the same heuristic as agg.
    target_in_flight = point.decode_bs * point.decode_workers
    num_requests = max(50, 5 * target_in_flight)
    base_row = _disagg_base_row(point)

    try:
        report = run_synthetic_trace_replay(
            isl,
            osl,
            num_requests,
            prefill_engine_args=MockEngineArgs.from_json(json.dumps(p_args)),
            decode_engine_args=MockEngineArgs.from_json(json.dumps(d_args)),
            num_prefill_workers=point.prefill_workers,
            num_decode_workers=point.decode_workers,
            replay_concurrency=target_in_flight,
            replay_mode="offline",
            router_mode="kv_router",
        )
    except Exception as exc:
        return {
            **base_row,
            "ttft_ms": None,
            "tpot_ms": None,
            "request_latency_ms": None,
            "tokens/s/user": None,
            "tokens/s/gpu": None,
            "output_throughput_tok_s": None,
            "completed_requests": None,
            "error": f"{type(exc).__name__}: {exc}",
        }

    output_throughput = report.get("output_throughput_tok_s") or 0.0
    return {
        **base_row,
        "ttft_ms": report.get("mean_ttft_ms"),
        "tpot_ms": report.get("mean_itl_ms"),
        "request_latency_ms": report.get("mean_e2e_latency_ms"),
        "tokens/s/user": report.get("mean_output_token_throughput_per_user"),
        "tokens/s/gpu": (
            output_throughput / point.total_gpus if point.total_gpus > 0 else None
        ),
        "output_throughput_tok_s": output_throughput,
        "completed_requests": report.get("completed_requests"),
        "error": None,
    }


def _run_parallel_sweep(
    tasks: list,
    evaluator,
    common_kwargs: dict,
    *,
    workers: int,
    describe,
    fallback_row,
) -> list[dict]:
    """Generic parallel/serial sweep runner used by both agg and disagg paths."""
    total = len(tasks)
    rows: list[dict] = []
    if workers <= 1:
        for done, task in enumerate(tasks, start=1):
            print(f"[{done}/{total}] {describe(task)}", flush=True)
            rows.append(
                evaluator(*task, **common_kwargs)
                if isinstance(task, tuple)
                else evaluator(task, **common_kwargs)
            )
        return rows

    print(f"Parallel sweep with {workers} workers ({total} tasks).", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        future_to_task = {}
        for task in tasks:
            if isinstance(task, tuple):
                fut = pool.submit(evaluator, *task, **common_kwargs)
            else:
                fut = pool.submit(evaluator, task, **common_kwargs)
            future_to_task[fut] = task
        done = 0
        for fut in as_completed(future_to_task):
            done += 1
            task = future_to_task[fut]
            try:
                rows.append(fut.result())
            except Exception as exc:
                rows.append(fallback_row(task, f"worker: {type(exc).__name__}: {exc}"))
            print(f"[{done}/{total}] {describe(task)}", flush=True)
    return rows


def _agg_fallback_row(task, error: str) -> dict:
    parallel_cfg, bs = task
    return {
        "tp": parallel_cfg.tp,
        "pp": parallel_cfg.pp,
        "dp": parallel_cfg.dp,
        "moe_tp": parallel_cfg.moe_tp,
        "moe_ep": parallel_cfg.moe_ep,
        "bs": bs,
        "num_gpus": parallel_cfg.num_gpus,
        "ttft_ms": None,
        "tpot_ms": None,
        "request_latency_ms": None,
        "tokens/s/user": None,
        "tokens/s/gpu": None,
        "output_throughput_tok_s": None,
        "completed_requests": None,
        "error": error,
    }


def _disagg_fallback_row(point: DisaggPoint, error: str) -> dict:
    return {
        **_disagg_base_row(point),
        "ttft_ms": None,
        "tpot_ms": None,
        "request_latency_ms": None,
        "tokens/s/user": None,
        "tokens/s/gpu": None,
        "output_throughput_tok_s": None,
        "completed_requests": None,
        "error": error,
    }


def _compute_pareto_and_save(
    rows: list[dict],
    *,
    save_dir: Path,
    raw_name: str,
    pareto_name: str,
    ttft: float,
    tpot: float,
    strict_sla: bool,
) -> tuple[Path, Path, int, int]:
    raw_df = pd.DataFrame(rows)
    raw_path = save_dir / raw_name
    raw_df.to_csv(raw_path, index=False)

    candidates = raw_df.dropna(subset=["tokens/s/user", "tokens/s/gpu"])
    candidates = candidates[candidates["ttft_ms"] <= ttft]
    if strict_sla:
        candidates = candidates[candidates["tpot_ms"] <= tpot]

    pareto_df = (
        get_pareto_front(
            candidates,
            x_col="tokens/s/user",
            y_col="tokens/s/gpu",
            maximize_x=True,
            maximize_y=True,
        )
        if not candidates.empty
        else candidates
    )
    pareto_path = save_dir / pareto_name
    pareto_df.to_csv(pareto_path, index=False)
    return raw_path, pareto_path, len(raw_df), len(pareto_df)


def sweep_pareto(args: argparse.Namespace) -> dict[str, Path]:
    """Run the mocker sweep(s) per ``args.mode`` and write CSVs."""
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    is_moe = _detect_is_moe(args.model_path)
    grid = _derive_parallel_grid(
        args.total_gpus, args.system, args.backend, is_moe, args.enable_wideep
    )
    print(
        f"Enumerated {len(grid)} parallelism configs "
        f"(model={args.model_path}, is_moe={is_moe}, total_gpus={args.total_gpus})."
    )

    workers = max(1, args.workers)
    common_kwargs = {
        "isl": args.isl,
        "osl": args.osl,
        "backend": args.backend,
        "backend_version": args.backend_version,
        "system": args.system,
        "model_path": args.model_path,
    }
    outputs: dict[str, Path] = {}

    if args.mode in ("agg", "both"):
        print("\n=== AGG SWEEP ===")
        agg_tasks = [(parallel_cfg, bs) for parallel_cfg in grid for bs in BATCH_SIZES]
        agg_rows = _run_parallel_sweep(
            agg_tasks,
            _evaluate_one,
            common_kwargs,
            workers=workers,
            describe=lambda t: _describe_task(t[0], t[1]),
            fallback_row=_agg_fallback_row,
        )
        raw_p, pareto_p, n_raw, n_pareto = _compute_pareto_and_save(
            agg_rows,
            save_dir=save_dir,
            raw_name="raw_mocker.csv",
            pareto_name="pareto_mocker.csv",
            ttft=args.ttft,
            tpot=args.tpot,
            strict_sla=args.strict_sla,
        )
        outputs["raw"] = raw_p
        outputs["pareto"] = pareto_p
        print(f"\nWrote {n_raw} agg raw points to {raw_p}")
        print(f"Wrote {n_pareto} agg pareto points to {pareto_p}")

    if args.mode in ("disagg", "both"):
        print("\n=== DISAGG SWEEP ===")
        disagg_points = _derive_disagg_grid(grid, args.total_gpus)
        print(f"Enumerated {len(disagg_points)} disagg points.")
        disagg_rows = _run_parallel_sweep(
            disagg_points,
            _evaluate_one_disagg,
            common_kwargs,
            workers=workers,
            describe=_describe_disagg_point,
            fallback_row=_disagg_fallback_row,
        )
        raw_p, pareto_p, n_raw, n_pareto = _compute_pareto_and_save(
            disagg_rows,
            save_dir=save_dir,
            raw_name="raw_mocker_disagg.csv",
            pareto_name="pareto_mocker_disagg.csv",
            ttft=args.ttft,
            tpot=args.tpot,
            strict_sla=args.strict_sla,
        )
        outputs["raw_disagg"] = raw_p
        outputs["pareto_disagg"] = pareto_p
        print(f"\nWrote {n_raw} disagg raw points to {raw_p}")
        print(f"Wrote {n_pareto} disagg pareto points to {pareto_p}")

    return outputs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m dynamo.replay.pareto",
        description=(
            "Mocker-based parallelism + batch-size sweep producing a Pareto "
            "front on (tokens/s/user, tokens/s/gpu)."
        ),
    )
    p.add_argument(
        "--model",
        "--model-path",
        dest="model_path",
        required=True,
        help="HF model id (e.g., 'moonshotai/Kimi-K2.5') or local model directory.",
    )
    p.add_argument("--system", required=True, help="GPU system (e.g., b200_sxm).")
    p.add_argument(
        "--backend",
        required=True,
        choices=("vllm", "trtllm", "sglang"),
        help="Inference backend.",
    )
    p.add_argument(
        "--backend-version",
        default=None,
        help="AIC backend DB version (default: latest available).",
    )
    p.add_argument(
        "--total-gpus",
        type=int,
        required=True,
        help="Total GPUs budget for parallelism enumeration.",
    )
    p.add_argument("--isl", type=int, default=4000, help="Input sequence length.")
    p.add_argument("--osl", type=int, default=1000, help="Output sequence length.")
    p.add_argument("--ttft", type=float, default=2000.0, help="TTFT SLA target (ms).")
    p.add_argument("--tpot", type=float, default=30.0, help="TPOT SLA target (ms).")
    p.add_argument(
        "--strict-sla",
        action="store_true",
        help="Filter the Pareto frontier to TPOT-compliant configs as well "
        "(TTFT is always enforced).",
    )
    p.add_argument(
        "--enable-wideep",
        action="store_true",
        help="Enable Wide Expert Parallelism for MoE models.",
    )
    p.add_argument(
        "--mode",
        default="agg",
        choices=("agg", "disagg", "both"),
        help="Deployment mode. 'agg' (default) sweeps aggregated deployments; "
        "'disagg' sweeps disaggregated prefill/decode pools; 'both' runs each "
        "and writes separate CSVs (raw_mocker[_disagg].csv etc.).",
    )
    p.add_argument("--save-dir", required=True, help="Output directory.")
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of worker processes for the mocker sweep "
        "(default: half of os.cpu_count). Use 1 for serial.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    sweep_pareto(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
