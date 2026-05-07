# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plot Exp 1 results: planner runs against a static-deployment Pareto curve."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from common import RESULTS_ROOT  # noqa: E402

EXP_NAME = "exp1_setup_tradeoff"
NV_GREEN = "#76B900"
INK = "#1A1A1A"
GRAY = "#8C8C8C"

TARGET_MARKERS = {"throughput": "*", "latency": "P", "sla": "o"}
MODE_COLORS = {"agg": "#0072B2", "disagg": "#D55E00"}


def load() -> list[dict]:
    p = RESULTS_ROOT / EXP_NAME / "results.json"
    return json.loads(p.read_text())


def main() -> None:
    rows = load()
    by_kind: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        params = r["params"]
        if params["sweep"] == "static":
            key = (params["mode"], "static")
        else:
            key = (params["mode"], params["optimization_target"])
        by_kind.setdefault(key, []).append(r)

    fig, (ax_pareto, ax_itl) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Pareto: GPU-hours vs p90 TTFT ---
    for mode in ["agg", "disagg"]:
        static_rows = sorted(
            by_kind.get((mode, "static"), []),
            key=lambda r: r["gpu_hours"] or 0.0,
        )
        if static_rows:
            xs = [r["gpu_hours"] for r in static_rows]
            ys = [(r["ttft_ms"] or {}).get("p90") for r in static_rows]
            ax_pareto.plot(
                xs,
                ys,
                "-o",
                color=MODE_COLORS[mode],
                lw=2,
                markersize=6,
                alpha=0.6,
                label=f"{mode} static",
            )

        for target in ["throughput", "latency", "sla"]:
            pts = by_kind.get((mode, target), [])
            if not pts:
                continue
            marker = TARGET_MARKERS[target]
            xs = [r["gpu_hours"] for r in pts]
            ys = [(r["ttft_ms"] or {}).get("p90") for r in pts]
            ax_pareto.scatter(
                xs,
                ys,
                marker=marker,
                s=120 if target != "sla" else 70,
                edgecolors=MODE_COLORS[mode],
                facecolors="none" if target == "sla" else MODE_COLORS[mode],
                linewidths=2,
                label=f"{mode} planner ({target})",
                zorder=5,
            )

    ax_pareto.axhline(1500, ls="--", color=GRAY, alpha=0.6, label="TTFT SLA 1500ms")
    ax_pareto.set_yscale("log")
    ax_pareto.set_xlabel("Cumulative GPU-hours")
    ax_pareto.set_ylabel("p90 TTFT (ms, log)")
    ax_pareto.set_title("Pareto: planner vs static deployment")
    ax_pareto.grid(True, which="both", alpha=0.3)
    ax_pareto.legend(loc="best", fontsize=8)

    # --- Bars: p90 ITL per planner run ---
    planner_rows = [r for r in rows if r["params"]["sweep"] == "planner"]
    planner_rows.sort(key=lambda r: (r["params"]["mode"], r["tag"]))
    labels = [r["tag"].replace("planner_", "") for r in planner_rows]
    itl_p90 = [(r["itl_ms"] or {}).get("p90") or 0.0 for r in planner_rows]
    colors = [MODE_COLORS[r["params"]["mode"]] for r in planner_rows]
    ax_itl.barh(range(len(labels)), itl_p90, color=colors, alpha=0.8)
    ax_itl.set_yticks(range(len(labels)))
    ax_itl.set_yticklabels(labels, fontsize=8)
    ax_itl.axvline(50, ls="--", color=GRAY, alpha=0.6, label="ITL SLA 50ms")
    ax_itl.set_xlabel("p90 ITL (ms)")
    ax_itl.set_title("p90 ITL per planner config")
    ax_itl.grid(True, axis="x", alpha=0.3)
    ax_itl.legend()

    fig.suptitle(
        "Exp 1 — Qwen3-32B / TP=2 / H200 / vLLM 0.12.0 — toolagent_trace, startup=60s",
        y=1.0,
    )
    plt.tight_layout()
    out = RESULTS_ROOT / EXP_NAME / "plot.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
