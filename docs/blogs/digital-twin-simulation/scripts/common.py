# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the digital-twin blog planner-experiment scripts.

Each `expN_*/run.py` builds a sweep of `python -m dynamo.replay` invocations,
runs them in a process pool, and parses the AIPerf metrics table from stdout
plus the cumulative GPU-hours line from the generated HTML report.

The expected layout is:

    docs/blogs/digital-twin-simulation/
        scripts/
            common.py                        (this file)
            planner_exp_N/run.py             (uses run_sweep)
            planner_exp_N/plot.py            (consumes results.json)
        images/
            planner_exp_N.png                (rendered figure for the blog)

All replay artifacts (per-run HTML, log, JSON, AND the aggregated
`results.json`) land outside the docs tree, under
`<repo_root>/planner_reports/blog_exp/planner_exp_N/`, so the docs
directory only carries the blog-facing PNGs.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# --- Repo paths --------------------------------------------------------------

REPO = Path(__file__).resolve().parents[4]
PY = REPO / ".venv/bin/python"
TRACE = REPO / "traces/mooncake_fast25/toolagent_trace.jsonl"
BLOG_DIR = REPO / "docs/blogs/digital-twin-simulation"
IMAGES_DIR = BLOG_DIR / "images"
REPORTS_ROOT = REPO / "planner_reports/blog_exp"


def results_json_path(exp_name: str) -> Path:
    """Aggregated per-run metrics JSON for a given experiment."""
    return REPORTS_ROOT / exp_name / "results.json"


def image_path(exp_name: str) -> Path:
    """Rendered figure path for a given experiment (lives in the blog images dir)."""
    return IMAGES_DIR / f"{exp_name}.png"


# --- Model / system defaults (Qwen3-32B, BF16, TP=2, H200-SXM, vLLM 0.12.0) --

DEFAULT_AIC_BACKEND = "vllm"
DEFAULT_AIC_BACKEND_VERSION = "0.12.0"
DEFAULT_AIC_SYSTEM = "h200_sxm"
DEFAULT_AIC_MODEL_PATH = "Qwen/Qwen3-32B"
DEFAULT_AIC_TP_SIZE = 2
DEFAULT_GPU_PER_REPLICA = 2  # = TP_SIZE for both prefill and decode roles
DEFAULT_MAX_GPU_BUDGET = 16  # blog framing — let the planner explore up to 8 replicas

# toolagent_trace.jsonl spans 3537s (0–3,536,999 ms). Used to compute
# analytical GPU-hours for static (no-planner) runs since the replay HTML
# is only generated on the planner path.
TRACE_DURATION_S = 3537.0


def base_engine_args(startup_time: float | None = None) -> dict[str, Any]:
    """Engine args common to every replay invocation in this blog."""
    args: dict[str, Any] = {
        "aic_backend": DEFAULT_AIC_BACKEND,
        "aic_backend_version": DEFAULT_AIC_BACKEND_VERSION,
        "aic_system": DEFAULT_AIC_SYSTEM,
        "aic_model_path": DEFAULT_AIC_MODEL_PATH,
        "aic_tp_size": DEFAULT_AIC_TP_SIZE,
    }
    if startup_time is not None and startup_time > 0:
        args["startup_time"] = startup_time
    return args


def base_planner_config(report_filename: str, **overrides: Any) -> dict[str, Any]:
    """Planner config with the blog's defaults baked in."""
    cfg: dict[str, Any] = {
        "mode": "agg",
        "optimization_target": "sla",
        "ttft": 1500,
        "itl": 50,
        "enable_throughput_scaling": True,
        "enable_load_scaling": True,
        "pre_deployment_sweeping_mode": "rapid",
        "throughput_adjustment_interval": 300,
        "load_adjustment_interval": 10,
        "prefill_engine_num_gpu": DEFAULT_GPU_PER_REPLICA,
        "decode_engine_num_gpu": DEFAULT_GPU_PER_REPLICA,
        "max_gpu_budget": DEFAULT_MAX_GPU_BUDGET,
        "report_filename": report_filename,
    }
    cfg.update(overrides)
    return cfg


def static_gpu_hours(params: dict[str, Any]) -> float | None:
    """Compute GPU-hours analytically for a static (no-planner) run.

    Uses the params stamped onto a `ReplayInvocation` row:
      - `mode == "agg"` + `num_workers`            → workers × TP × duration
      - `mode == "disagg"` + `num_prefill, num_decode` → (P+D) × TP × duration
    """
    mode = params.get("mode")
    duration_h = TRACE_DURATION_S / 3600.0
    if mode == "agg":
        n = params.get("num_workers")
        if n is None:
            return None
        return n * DEFAULT_GPU_PER_REPLICA * duration_h
    if mode == "disagg":
        p = params.get("num_prefill")
        d = params.get("num_decode")
        if p is None or d is None:
            return None
        return (p + d) * DEFAULT_GPU_PER_REPLICA * duration_h
    return None


# --- Parsing -----------------------------------------------------------------

_METRIC_ROW_RE = re.compile(r"┃")


def _parse_float(s: str) -> float | None:
    s = s.strip().replace(",", "")
    if not s or s == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_metrics_row(stdout: str, metric_name: str) -> dict[str, float | None]:
    """Pull one row out of the AIPerf-style box-drawn metrics table.

    Columns, in order: avg, min, max, p99, p90, p75, std.
    """
    for line in stdout.splitlines():
        if metric_name in line and _METRIC_ROW_RE.search(line):
            parts = [p.strip() for p in line.split("┃") if p.strip()]
            # parts[0] is the metric name; the rest are numeric columns
            if len(parts) >= 8:
                return {
                    "avg": _parse_float(parts[1]),
                    "min": _parse_float(parts[2]),
                    "max": _parse_float(parts[3]),
                    "p99": _parse_float(parts[4]),
                    "p90": _parse_float(parts[5]),
                    "p75": _parse_float(parts[6]),
                    "std": _parse_float(parts[7]),
                }
    return {}


_GPU_HOURS_RE = re.compile(r"GPU hours:\s*([0-9.]+)")


def parse_gpu_hours(html_path: Path) -> float | None:
    """Grep the cumulative GPU-hours figure out of the planner HTML report."""
    if not html_path.exists():
        return None
    text = html_path.read_text()
    matches = _GPU_HOURS_RE.findall(text)
    return float(matches[-1]) if matches else None


def count_scaling_events(stdout: str) -> tuple[int, int]:
    """Count `scale_up` / `scale_down` log lines emitted by the planner."""
    up = sum(1 for line in stdout.splitlines() if "scale_up" in line)
    down = sum(1 for line in stdout.splitlines() if "scale_down" in line)
    return up, down


# --- Run primitive -----------------------------------------------------------


@dataclass
class ReplayInvocation:
    """One replay invocation in a sweep.

    `tag` becomes the per-run report filename stem and the JSON row id.
    `params` is opaque metadata stamped onto the result row (e.g. the swept value).
    """

    tag: str
    params: dict[str, Any]
    extra_engine_args: dict[str, Any] | None = None
    prefill_engine_args: dict[str, Any] | None = None
    decode_engine_args: dict[str, Any] | None = None
    planner_config: dict[str, Any] | None = None  # None = static (no planner)
    cli_extra: list[str] = field(default_factory=list)


def _build_cmd(inv: ReplayInvocation, exp_dir: Path) -> list[str]:
    cmd: list[str] = [str(PY), "-m", "dynamo.replay", str(TRACE)]
    if inv.planner_config is not None:
        cmd += ["--planner-config", json.dumps(inv.planner_config)]
    if inv.extra_engine_args is not None:
        cmd += ["--extra-engine-args", json.dumps(inv.extra_engine_args)]
    if inv.prefill_engine_args is not None:
        cmd += ["--prefill-engine-args", json.dumps(inv.prefill_engine_args)]
    if inv.decode_engine_args is not None:
        cmd += ["--decode-engine-args", json.dumps(inv.decode_engine_args)]
    cmd += inv.cli_extra
    cmd += ["--arrival-speedup-ratio", "1.0"]
    cmd += ["--report-json", str(exp_dir / f"{inv.tag}.json")]
    return cmd


def _execute(inv: ReplayInvocation, exp_dir: Path, html_dir: Path) -> dict[str, Any]:
    exp_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    cmd = _build_cmd(inv, exp_dir)
    log_path = exp_dir / f"{inv.tag}.log"
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    wall = time.time() - t0
    log_path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""))

    ttft = parse_metrics_row(proc.stdout, "Time to First Token")
    itl = parse_metrics_row(proc.stdout, "Inter Token Latency")
    req_lat = parse_metrics_row(proc.stdout, "Request Latency")
    n_up, n_down = count_scaling_events(proc.stdout)

    # Resolve the report HTML location. The replay command writes to
    # ./planner_reports/<report_filename> relative to its cwd (the repo root).
    report_filename = (
        (inv.planner_config or {}).get("report_filename")
        if inv.planner_config
        else None
    )
    gpu_hours = None
    if report_filename:
        gpu_hours = parse_gpu_hours(REPO / "planner_reports" / report_filename)
    if gpu_hours is None and inv.planner_config is None:
        # Static (no-planner) runs don't emit the diagnostics HTML — fall back
        # to the analytical formula based on the swept worker counts.
        gpu_hours = static_gpu_hours(inv.params)

    return {
        "tag": inv.tag,
        "params": inv.params,
        "wall_time_s": wall,
        "returncode": proc.returncode,
        "ttft_ms": ttft,
        "itl_ms": itl,
        "request_latency_ms": req_lat,
        "scale_up_events": n_up,
        "scale_down_events": n_down,
        "gpu_hours": gpu_hours,
    }


def run_sweep(
    invocations: Iterable[ReplayInvocation],
    exp_name: str,
    *,
    max_workers: int = 12,
    on_done: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run a list of replay invocations in a process pool and aggregate results.

    All per-run logs/JSON AND the aggregated `results.json` land under
    `planner_reports/blog_exp/<exp_name>/`.
    """
    exp_html_dir = REPORTS_ROOT / exp_name
    exp_html_dir.mkdir(parents=True, exist_ok=True)
    results_json = results_json_path(exp_name)

    invocations = list(invocations)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(_execute, inv, exp_html_dir, exp_html_dir): inv
            for inv in invocations
        }
        for fut in as_completed(futs):
            row = fut.result()
            results.append(row)
            results.sort(key=lambda r: r["tag"])
            results_json.write_text(json.dumps(results, indent=2))
            if on_done:
                on_done(row)
    return results
