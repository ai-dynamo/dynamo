# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Leaderboard — sweep autoscalers × workloads on one pinned config, then rank.

This is the Arena's top-level use case: "run an autoscaler comparison." It pins a
single substrate config (model × engine × hardware × topology) and a planner
config, then for each (autoscaler, workload) runs one DynoSim match and scores
it. The default ranking axis is **Efficiency = goodput / average GPU** for a chosen
SLO profile (the Arena's headline), but every scorecard field is retained.

All autoscalers share the SAME substrate config — the planner-specific fields are
simply ignored by the rival adapters — so the comparison is apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from autoscaling_arena.adapters import (
    KedaAutoscaler,
    ReactiveAutoscaler,
    StaticAutoscaler,
    planner_engine_factory,
)
from autoscaling_arena.runners import run_arena_replay
from autoscaling_arena.scorecard import (
    DEFAULT_PROFILES,
    SCORECARD_SCHEMA_VERSION,
    SLOProfile,
    scorecard,
)
from autoscaling_arena.workloads import get_workload


@dataclass
class AutoscalerSpec:
    """One entry in the leaderboard: a named autoscaler + its start fleet."""

    name: str
    factory: Callable[
        [Any, Any], Any
    ]  # (PlannerConfig, WorkerCapabilities) -> EngineProtocol
    start_prefill: int
    start_decode: int


def default_autoscalers(
    *, mode: str = "disagg", max_prefill: int = 16, max_decode: int = 8
) -> list[AutoscalerSpec]:
    """The v1 roster: Planner (SUT) vs KEDA / reactive / best-static baselines.

    Dynamic scalers start at the floor (1P1D) and must earn their fleet; the
    static baseline holds a fixed, decent shape (4P4D).
    """
    return [
        AutoscalerSpec("planner", planner_engine_factory, 1, 1),
        AutoscalerSpec(
            "keda",
            lambda c, k: KedaAutoscaler(
                mode=mode,
                capabilities=k,
                max_prefill=max_prefill,
                max_decode=max_decode,
            ),
            1,
            1,
        ),
        AutoscalerSpec(
            "reactive",
            lambda c, k: ReactiveAutoscaler(
                mode=mode,
                capabilities=k,
                max_prefill=max_prefill,
                max_decode=max_decode,
            ),
            1,
            1,
        ),
        AutoscalerSpec(
            "static-4P4D",
            lambda c, k: StaticAutoscaler(num_prefill=4, num_decode=4, mode=mode),
            4,
            4,
        ),
    ]


def run_leaderboard(
    *,
    autoscalers: list[AutoscalerSpec],
    workload_names: list[str],
    prefill_engine_args: str,
    decode_engine_args: str,
    substrate_config: str,
    trace_dir: Path,
    profiles: tuple[SLOProfile, ...] = DEFAULT_PROFILES,
    score_profile: Optional[SLOProfile] = None,
    seed: int = 0,
    arrival_speedup_ratio: float = 1.0,
    max_requests: Optional[int] = None,
    on_progress: Optional[Callable[[str, str], None]] = None,
) -> list[dict[str, Any]]:
    """Run the full sweep; return one result row per (autoscaler, workload).

    Each row: ``{autoscaler, workload, scorecard}``. Workloads are materialized
    once (deterministically) under ``trace_dir``. ``max_requests`` caps each
    workload to its earliest N requests (faster sweeps);
    ``arrival_speedup_ratio`` is applied by the replay runtime (not baked into
    the trace).

    ``score_profile``: when set, its SLA is also supplied to the replay so the
    mocker emits in-Rust ``goodput_*`` for that profile as a capture-disabled
    fallback. Per-request capture scores every requested profile post hoc.
    """
    sla = score_profile
    rows: list[dict[str, Any]] = []
    for wl_name in workload_names:
        wl = get_workload(wl_name)
        trace = str(wl.materialize(trace_dir, seed=seed, max_requests=max_requests))
        for spec in autoscalers:
            if on_progress:
                on_progress(spec.name, wl_name)
            report = run_arena_replay(
                trace_file=trace,
                autoscaler=spec.factory,
                substrate_config=substrate_config,
                prefill_engine_args=prefill_engine_args,
                decode_engine_args=decode_engine_args,
                num_prefill_workers=spec.start_prefill,
                num_decode_workers=spec.start_decode,
                arrival_speedup_ratio=arrival_speedup_ratio,
                trace_block_size=wl.block_size,
                sla_ttft_ms=(sla.ttft_ms if sla else None),
                sla_itl_ms=(sla.itl_ms if sla else None),
                sla_e2e_ms=(sla.e2e_ms if sla else None),
                capture_per_request=True,
            )
            rows.append(
                {
                    "autoscaler": spec.name,
                    "workload": wl_name,
                    "scorecard": scorecard(report, profiles, sla_profile=sla),
                }
            )
    return rows


def _fmt(v: Optional[float], width: int = 8, prec: int = 2) -> str:
    return (
        f"{v:>{width}.{prec}f}" if isinstance(v, (int, float)) else f"{'n/a':>{width}}"
    )


def format_leaderboard(
    rows: list[dict[str, Any]],
    *,
    profile: str = "interactive",
    rank_by: str = "goodput_per_gpu",
) -> str:
    """Render a per-workload ranking table for one SLO profile + an overall mean.

    Ranks by ``goodput_per_gpu`` (Efficiency) for ``profile`` by default.
    """
    for row in rows:
        version = row["scorecard"].get("scorecard_schema_version")
        if version != SCORECARD_SCHEMA_VERSION:
            raise ValueError(
                "leaderboard row uses unsupported scorecard schema "
                f"{version!r}; regenerate it with schema {SCORECARD_SCHEMA_VERSION}"
            )
    workloads = sorted({r["workload"] for r in rows})
    autoscalers = list(dict.fromkeys(r["autoscaler"] for r in rows))  # preserve order
    lines: list[str] = []
    lines.append(
        f"\nSLO profile: {profile}   |   ranked by {rank_by} "
        "(Efficiency = goodput / average allocated GPU)\n"
    )

    overall: dict[str, list[float]] = {a: [] for a in autoscalers}
    for wl in workloads:
        lines.append(f"── workload: {wl} " + "─" * max(0, 48 - len(wl)))
        lines.append(
            f"  {'autoscaler':<13}{'good/GPU':>11}{'goodput_rps':>13}"
            f"{'good_rate':>11}{'avg_GPU':>9}{'GPU-h':>8}{'mean_ttft':>11}{'osc':>6}"
        )
        wl_rows = [r for r in rows if r["workload"] == wl]
        ranked = sorted(
            wl_rows,
            key=lambda r: (r["scorecard"]["profiles"][profile].get(rank_by) or -1.0),
            reverse=True,
        )
        for r in ranked:
            sc = r["scorecard"]
            p = sc["profiles"][profile]
            efficiency = p.get(rank_by)
            if isinstance(efficiency, (int, float)):
                overall[r["autoscaler"]].append(efficiency)
            lines.append(
                f"  {r['autoscaler']:<13}{_fmt(p.get('goodput_per_gpu'), 11)}"
                f"{_fmt(p.get('goodput_rps'), 13)}{_fmt(p.get('good_rate'), 11)}"
                f"{_fmt(sc.get('average_gpus'), 9, 2)}{_fmt(sc.get('gpu_hours'), 8, 3)}"
                f"{_fmt(sc.get('mean_ttft_ms'), 11, 0)}"
                f"{sc.get('oscillation_count', 0):>6}"
            )
        lines.append("")

    lines.append("══ overall (mean good/GPU across workloads) " + "═" * 14)
    means = {a: (sum(v) / len(v) if v else 0.0) for a, v in overall.items()}
    for a, m in sorted(means.items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"  {a:<13}{_fmt(m, 11)}")
    return "\n".join(lines)
