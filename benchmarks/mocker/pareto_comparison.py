# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIC vs Mocker Pareto comparison driver.

This script sweeps the same (parallelism, batch_size) grid through two perf
predictors — AIC's analytical model and dynamo's mocker simulator — and plots
their respective Pareto fronts side by side on a single figure so disagreements
are visible at a glance.

The mocker side is run by importing ``dynamo.replay.pareto.sweep_pareto``; the
AIC side is run by driving ``aiconfigurator.sdk.inference_session.InferenceSession``
in-process. The two outputs use the same Pareto algorithm
(``aiconfigurator.sdk.pareto_analysis.get_pareto_front``) so any divergence is
purely model-disagreement, not algorithmic.

Usage::

    python benchmarks/mocker/pareto_comparison.py \
        --model moonshotai/Kimi-K2.5 \
        --system b200_sxm \
        --backend vllm \
        --backend-version 0.19.0 \
        --total-gpus 8 \
        --isl 8192 --osl 1024 \
        --ttft 1000 --tpot 50 \
        --save-dir results/kimi_b200_compare

Outputs in ``--save-dir`` (in addition to mocker's own raw/pareto CSVs)::

    raw_aic.csv         every AIC-evaluated (parallelism, bs) point
    pareto_aic.csv      AIC's Pareto-front subset
    pareto_plot.png     dual-curve plot (AIC + Mocker)
    sweep_meta.json     invocation args + AIC + mocker version pins
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from aiconfigurator.sdk.backends.factory import get_backend
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk.inference_session import (
    DisaggInferenceSession,
    InferenceSession,
)
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.pareto_analysis import get_pareto_front
from aiconfigurator.sdk.perf_database import get_database

from dynamo.replay.pareto import (
    BATCH_SIZES,
    DisaggPoint,
    ParallelConfig,
    _derive_disagg_grid,
    _derive_parallel_grid,
    _detect_is_moe,
    sweep_pareto,
)

X_COL = "tokens/s/user"
Y_COL = "tokens/s/gpu"


_AIC_METRIC_KEYS = ("ttft_ms", "tpot_ms", "request_latency_ms", X_COL, Y_COL)


def _empty_aic_row(parallel_cfg: ParallelConfig, bs: int, error: str) -> dict:
    return {
        "tp": parallel_cfg.tp,
        "pp": parallel_cfg.pp,
        "dp": parallel_cfg.dp,
        "moe_tp": parallel_cfg.moe_tp,
        "moe_ep": parallel_cfg.moe_ep,
        "bs": bs,
        "num_gpus": parallel_cfg.num_gpus,
        **{k: None for k in _AIC_METRIC_KEYS},
        "error": error,
    }


def _evaluate_aic_point(
    session: InferenceSession,
    parallel_cfg: ParallelConfig,
    bs: int,
    *,
    isl: int,
    osl: int,
    ttft: float,
    tpot: float,
) -> dict:
    """Run AIC's analytical perf model for one (parallelism, bs) point.

    AIC's ``run_agg`` requires ``ctx_tokens`` as a kwarg (the per-step prefill
    token budget for IFB scheduling, analogous to vllm's
    ``--max-num-batched-tokens``). We use ``ctx_tokens=isl`` — one full
    prefill per step at this bs level, which is the natural operating point
    for vllm-style IFB with chunked-prefill disabled. Earlier versions used
    ``max(isl, 8192)`` which inflated TTFT predictions at low bs because AIC
    simulated a larger-than-needed forward pass per step.
    """
    rt = RuntimeConfig(batch_size=bs, isl=isl, osl=osl, ttft=ttft, tpot=tpot)
    ctx_tokens = isl
    try:
        summary = session.run_agg(rt, ctx_tokens=ctx_tokens)
        df = summary.get_summary_df()
    except Exception as exc:
        return _empty_aic_row(parallel_cfg, bs, f"{type(exc).__name__}: {exc}")

    if df is None or df.empty:
        return _empty_aic_row(parallel_cfg, bs, "empty summary")

    row = df.iloc[0].to_dict()
    return {
        "tp": parallel_cfg.tp,
        "pp": parallel_cfg.pp,
        "dp": parallel_cfg.dp,
        "moe_tp": parallel_cfg.moe_tp,
        "moe_ep": parallel_cfg.moe_ep,
        "bs": bs,
        "num_gpus": parallel_cfg.num_gpus,
        "ttft_ms": row.get("ttft"),
        "tpot_ms": row.get("tpot"),
        "request_latency_ms": row.get("request_latency"),
        X_COL: row.get(X_COL),
        Y_COL: row.get(Y_COL),
        "error": None,
    }


def evaluate_aic(args: argparse.Namespace) -> list[dict]:
    """Evaluate AIC's analytical model on the same parallelism × bs grid."""
    is_moe = _detect_is_moe(args.model_path)
    grid = _derive_parallel_grid(
        args.total_gpus, args.system, args.backend, is_moe, args.enable_wideep
    )
    print(f"AIC: evaluating {len(grid)} parallelism configs × {len(BATCH_SIZES)} bs.")

    database = get_database(
        system=args.system, backend=args.backend, version=args.backend_version
    )
    backend_obj = get_backend(args.backend)

    rows: list[dict] = []
    total = len(grid) * len(BATCH_SIZES)
    done = 0
    for parallel_cfg in grid:
        model_cfg = ModelConfig(
            tp_size=parallel_cfg.tp,
            pp_size=parallel_cfg.pp,
            attention_dp_size=parallel_cfg.dp,
            moe_tp_size=parallel_cfg.moe_tp,
            moe_ep_size=parallel_cfg.moe_ep,
            enable_wideep=args.enable_wideep,
        )
        try:
            model = get_model(
                model_path=args.model_path,
                model_config=model_cfg,
                backend_name=args.backend,
            )
            session = InferenceSession(model, database, backend_obj)
        except Exception as exc:
            done += len(BATCH_SIZES)
            err = f"model init: {type(exc).__name__}: {exc}"
            print(
                f"  skip tp={parallel_cfg.tp} dp={parallel_cfg.dp} "
                f"moe_tp={parallel_cfg.moe_tp} moe_ep={parallel_cfg.moe_ep}: {err}",
                flush=True,
            )
            for bs in BATCH_SIZES:
                rows.append(_empty_aic_row(parallel_cfg, bs, err))
            continue

        for bs in BATCH_SIZES:
            done += 1
            print(
                f"[AIC {done}/{total}] tp={parallel_cfg.tp} pp={parallel_cfg.pp} "
                f"dp={parallel_cfg.dp} moe_tp={parallel_cfg.moe_tp} "
                f"moe_ep={parallel_cfg.moe_ep} bs={bs}",
                flush=True,
            )
            rows.append(
                _evaluate_aic_point(
                    session,
                    parallel_cfg,
                    bs,
                    isl=args.isl,
                    osl=args.osl,
                    ttft=args.ttft,
                    tpot=args.tpot,
                )
            )

    return rows


def _filter_sla(
    df: pd.DataFrame, ttft: float, tpot: float, strict: bool
) -> pd.DataFrame:
    candidates = df.dropna(subset=[X_COL, Y_COL])
    candidates = candidates[candidates["ttft_ms"] <= ttft]
    if strict:
        candidates = candidates[candidates["tpot_ms"] <= tpot]
    return candidates


_DISAGG_AIC_METRIC_KEYS = ("ttft_ms", "tpot_ms", "request_latency_ms", X_COL, Y_COL)


def _empty_disagg_row(point: DisaggPoint, error: str) -> dict:
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
        **{k: None for k in _DISAGG_AIC_METRIC_KEYS},
        "error": error,
    }


def _evaluate_aic_disagg_point(
    session: DisaggInferenceSession,
    point: DisaggPoint,
    *,
    model_path: str,
    isl: int,
    osl: int,
    ttft: float,
    tpot: float,
) -> dict:
    """Run AIC's disagg analytical model for one (prefill, decode, workers, bs) point."""
    rt = RuntimeConfig(batch_size=None, isl=isl, osl=osl, ttft=ttft, tpot=tpot)
    p_cfg = ModelConfig(
        tp_size=point.prefill.tp,
        pp_size=point.prefill.pp,
        attention_dp_size=point.prefill.dp,
        moe_tp_size=point.prefill.moe_tp,
        moe_ep_size=point.prefill.moe_ep,
    )
    d_cfg = ModelConfig(
        tp_size=point.decode.tp,
        pp_size=point.decode.pp,
        attention_dp_size=point.decode.dp,
        moe_tp_size=point.decode.moe_tp,
        moe_ep_size=point.decode.moe_ep,
    )
    try:
        summary = session.run_disagg(
            model_path=model_path,
            runtime_config=rt,
            prefill_model_config=p_cfg,
            prefill_batch_size=point.prefill_bs,
            prefill_num_worker=point.prefill_workers,
            decode_model_config=d_cfg,
            decode_batch_size=point.decode_bs,
            decode_num_worker=point.decode_workers,
        )
        df = summary.get_summary_df()
    except Exception as exc:
        return _empty_disagg_row(point, f"{type(exc).__name__}: {exc}")

    if df is None or df.empty:
        return _empty_disagg_row(point, "empty summary")

    row = df.iloc[0].to_dict()
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
        "ttft_ms": row.get("ttft"),
        "tpot_ms": row.get("tpot"),
        "request_latency_ms": row.get("request_latency"),
        X_COL: row.get(X_COL),
        Y_COL: row.get(Y_COL),
        "error": None,
    }


def evaluate_aic_disagg(args: argparse.Namespace) -> list[dict]:
    """Evaluate AIC's analytical model on the disagg search space."""
    is_moe = _detect_is_moe(args.model_path)
    grid = _derive_parallel_grid(
        args.total_gpus, args.system, args.backend, is_moe, args.enable_wideep
    )
    points = _derive_disagg_grid(grid, args.total_gpus)
    print(f"AIC disagg: evaluating {len(points)} points.")

    database = get_database(
        system=args.system, backend=args.backend, version=args.backend_version
    )
    backend_obj = get_backend(args.backend)
    # Same database + backend for prefill and decode (single-system v1).
    session = DisaggInferenceSession(
        prefill_database=database,
        prefill_backend=backend_obj,
        decode_database=database,
        decode_backend=backend_obj,
    )

    rows: list[dict] = []
    for done, point in enumerate(points, start=1):
        if done % 25 == 0 or done == 1:
            print(f"[AIC disagg {done}/{len(points)}]", flush=True)
        rows.append(
            _evaluate_aic_disagg_point(
                session,
                point,
                model_path=args.model_path,
                isl=args.isl,
                osl=args.osl,
                ttft=args.ttft,
                tpot=args.tpot,
            )
        )
    return rows


_CURVE_STYLE = {
    # (raw_color, raw_alpha, pareto_color, pareto_marker, pareto_linestyle)
    "agg_aic": ("C0", 0.2, "C0", "o", "-"),
    "agg_mocker": ("C1", 0.2, "C1", "s", "--"),
    "disagg_aic": ("C2", 0.2, "C2", "^", "-"),
    "disagg_mocker": ("C3", 0.2, "C3", "D", "--"),
}


def _plot_curve(
    ax,
    label: str,
    *,
    raw: pd.DataFrame | None,
    pareto: pd.DataFrame | None,
    style_key: str,
) -> None:
    raw_color, raw_alpha, p_color, p_marker, p_ls = _CURVE_STYLE[style_key]
    if raw is not None and not raw.empty:
        ax.scatter(
            raw[X_COL],
            raw[Y_COL],
            alpha=raw_alpha,
            color=raw_color,
            s=15,
            label=f"{label} raw",
        )
    if pareto is not None and not pareto.empty:
        sorted_p = pareto.sort_values(X_COL)
        ax.plot(
            sorted_p[X_COL],
            sorted_p[Y_COL],
            marker=p_marker,
            linewidth=2,
            color=p_color,
            linestyle=p_ls,
            label=f"{label} pareto",
        )


def _plot_pareto(
    save_path: Path,
    *,
    curves: dict[str, dict[str, pd.DataFrame | None]],
    title: str,
) -> None:
    """Plot up to 4 curves on a single figure.

    ``curves`` maps style_key (e.g. ``"agg_aic"``) to a dict with
    ``"raw"`` and ``"pareto"`` DataFrames (either may be None to skip).
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))

    label_map = {
        "agg_aic": "AIC agg",
        "agg_mocker": "Mocker agg",
        "disagg_aic": "AIC disagg",
        "disagg_mocker": "Mocker disagg",
    }
    for key, dfs in curves.items():
        _plot_curve(
            ax,
            label_map.get(key, key),
            raw=dfs.get("raw"),
            pareto=dfs.get("pareto"),
            style_key=key,
        )

    ax.set_xlabel(f"{X_COL} (tokens/s/user)")
    ax.set_ylabel(f"{Y_COL} (tokens/s/gpu)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pareto_comparison.py",
        description="Run AIC + mocker Pareto sweeps over the same config grid and plot both.",
    )
    p.add_argument(
        "--model",
        "--model-path",
        dest="model_path",
        required=True,
        help="HF model id or local model directory.",
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
    p.add_argument("--total-gpus", type=int, required=True, help="Total GPUs budget.")
    p.add_argument("--isl", type=int, default=4000, help="Input sequence length.")
    p.add_argument("--osl", type=int, default=1000, help="Output sequence length.")
    p.add_argument("--ttft", type=float, default=2000.0, help="TTFT SLA target (ms).")
    p.add_argument("--tpot", type=float, default=30.0, help="TPOT SLA target (ms).")
    p.add_argument(
        "--strict-sla",
        action="store_true",
        help="Also filter the Pareto frontier by TPOT (TTFT is always enforced).",
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
        help="Deployment mode. 'agg' (default), 'disagg', or 'both'.",
    )
    p.add_argument("--save-dir", required=True, help="Output directory.")
    p.add_argument(
        "--skip-mocker",
        action="store_true",
        help="Skip mocker sweep; expects raw_mocker.csv/pareto_mocker.csv already in save-dir.",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Worker processes for the mocker sweep (default: half of cpu_count).",
    )
    return p


def _run_aic_agg(
    args: argparse.Namespace, save_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = evaluate_aic(args)
    raw = pd.DataFrame(rows)
    raw.to_csv(save_dir / "raw_aic.csv", index=False)
    cand = _filter_sla(raw, args.ttft, args.tpot, args.strict_sla)
    pareto = (
        get_pareto_front(
            cand, x_col=X_COL, y_col=Y_COL, maximize_x=True, maximize_y=True
        )
        if not cand.empty
        else cand
    )
    pareto.to_csv(save_dir / "pareto_aic.csv", index=False)
    return raw, pareto


def _run_aic_disagg(
    args: argparse.Namespace, save_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = evaluate_aic_disagg(args)
    raw = pd.DataFrame(rows)
    raw.to_csv(save_dir / "raw_aic_disagg.csv", index=False)
    cand = _filter_sla(raw, args.ttft, args.tpot, args.strict_sla)
    pareto = (
        get_pareto_front(
            cand, x_col=X_COL, y_col=Y_COL, maximize_x=True, maximize_y=True
        )
        if not cand.empty
        else cand
    )
    pareto.to_csv(save_dir / "pareto_aic_disagg.csv", index=False)
    return raw, pareto


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_mocker:
        sweep_pareto(args)

    curves: dict[str, dict] = {}
    counts: dict[str, int] = {}

    if args.mode in ("agg", "both"):
        mocker_raw = pd.read_csv(save_dir / "raw_mocker.csv")
        mocker_pareto = pd.read_csv(save_dir / "pareto_mocker.csv")
        aic_raw, aic_pareto = _run_aic_agg(args, save_dir)
        curves["agg_aic"] = {"raw": aic_raw, "pareto": aic_pareto}
        curves["agg_mocker"] = {"raw": mocker_raw, "pareto": mocker_pareto}
        counts.update(
            mocker_raw_agg=len(mocker_raw),
            mocker_pareto_agg=len(mocker_pareto),
            aic_raw_agg=len(aic_raw),
            aic_pareto_agg=len(aic_pareto),
        )

    if args.mode in ("disagg", "both"):
        mocker_raw_d = pd.read_csv(save_dir / "raw_mocker_disagg.csv")
        mocker_pareto_d = pd.read_csv(save_dir / "pareto_mocker_disagg.csv")
        aic_raw_d, aic_pareto_d = _run_aic_disagg(args, save_dir)
        curves["disagg_aic"] = {"raw": aic_raw_d, "pareto": aic_pareto_d}
        curves["disagg_mocker"] = {"raw": mocker_raw_d, "pareto": mocker_pareto_d}
        counts.update(
            mocker_raw_disagg=len(mocker_raw_d),
            mocker_pareto_disagg=len(mocker_pareto_d),
            aic_raw_disagg=len(aic_raw_d),
            aic_pareto_disagg=len(aic_pareto_d),
        )

    plot_path = save_dir / "pareto_plot.png"
    title = (
        f"{args.model_path} on {args.system} | "
        f"backend={args.backend} v={args.backend_version or 'latest'} | "
        f"ISL={args.isl} OSL={args.osl} | "
        f"TTFT≤{args.ttft}ms TPOT≤{args.tpot}ms | mode={args.mode}"
    )
    _plot_pareto(plot_path, curves=curves, title=title)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "args": vars(args),
        "counts": counts,
    }
    (save_dir / "sweep_meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"\nWrote plot: {plot_path}")
    print(f"Wrote meta: {save_dir / 'sweep_meta.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
