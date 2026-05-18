#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate fig-6: KV-aware routing comparison (kv_router_exp redraw).

Left panel: mean TTFT (log y) vs concurrency. Right panel: throughput
(TPS/GPU) vs interactivity (TPS/User) Pareto. Round-robin baseline in
muted gray, KV Router in NV green.

Data: tools/data_kv_router.csv (copied from PR 9139).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly_dynamo import dynamo_template, load_tokens

HERE = Path(__file__).resolve().parent
TOKENS = load_tokens()

C = TOKENS["colors"]
TY = TOKENS["typography"]
TEXT_PRIMARY = C["text"]["primary"]
TEXT_SECONDARY = C["text"]["secondary"]
TEXT_MUTED = C["text"]["muted"]
BORDER_SUBTLE = C["border"]["subtle"]
NV_GREEN = C["accent"]["dynamo_green"]
CPU_BLUE = C["accent"]["cpu_blue"]
SANS = TY["font_family"]
MONO = TY["font_family_mono"]

# Series styling.
SERIES = {
    "round_robin": dict(name="Round Robin", color="#909090"),
    "kv_router":   dict(name="KV Router",   color=NV_GREEN),
}


def load_csv(path: Path) -> dict[str, list[dict]]:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_mode[row["router_mode"]].append({
                "concurrency": int(row["concurrency"]),
                "tps_gpu":     float(row["tps_gpu"]),
                "tps_user":    float(row["tps_user"]),
                "ttft_ms":     float(row["ttft_ms"]),
                "prefix":      float(row["prefix_cache_reused_ratio"]),
            })
    for mode in by_mode:
        by_mode[mode].sort(key=lambda r: r["concurrency"])
    return by_mode


def main() -> None:
    data = load_csv(HERE / "data_kv_router.csv")

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)

    # ---------------- Left: Mean TTFT vs Concurrency ----------------
    for mode in ("round_robin", "kv_router"):
        s = SERIES[mode]
        rows = data[mode]
        xs = [r["concurrency"] for r in rows]
        ys = [r["ttft_ms"] for r in rows]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=s["name"], legendgroup=mode,
                line=dict(color=s["color"], width=2.0),
                marker=dict(color=s["color"], symbol="diamond" if mode == "kv_router" else "circle",
                            size=10, line=dict(color=C["background"]["primary"], width=1)),
                hovertemplate=f"<b>{s['name']}</b><br>c=%{{x}}<br>%{{y:,.0f}} ms<extra></extra>",
            ),
            row=1, col=1,
        )

    # Per-concurrency reduction labels (NV green) below each KV-Router point.
    # Rendered as a text-mode Scatter trace so they bind to the subplot's
    # axes the same way the data does (annotation row/col binding behaves
    # inconsistently in this version of Plotly).
    rr = {r["concurrency"]: r for r in data["round_robin"]}
    kv = {r["concurrency"]: r for r in data["kv_router"]}
    pct_xs, pct_ys, pct_texts = [], [], []
    for c in sorted(kv):
        rr_ttft = rr[c]["ttft_ms"]
        kv_ttft = kv[c]["ttft_ms"]
        pct = (kv_ttft - rr_ttft) / rr_ttft * 100
        pct_xs.append(c)
        pct_ys.append(kv_ttft * 0.78)  # ~12% below the marker on log y
        pct_texts.append(f"<b>{pct:.0f}%</b>")
    fig.add_trace(
        go.Scatter(
            x=pct_xs, y=pct_ys, mode="text",
            text=pct_texts, textposition="middle center",
            textfont=dict(family=MONO, size=11, color=NV_GREEN),
            showlegend=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )

    # ---------------- Right: Throughput vs Interactivity Pareto ----------------
    for mode in ("round_robin", "kv_router"):
        s = SERIES[mode]
        rows = data[mode]
        xs = [r["tps_user"] for r in rows]
        ys = [r["tps_gpu"] for r in rows]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=s["name"], legendgroup=mode, showlegend=False,
                line=dict(color=s["color"], width=2.0),
                marker=dict(color=s["color"], symbol="diamond" if mode == "kv_router" else "circle",
                            size=10, line=dict(color=C["background"]["primary"], width=1)),
                hovertemplate=f"<b>{s['name']}</b><br>TPS/user %{{x:.1f}}<br>TPS/GPU %{{y:.0f}}<extra></extra>",
            ),
            row=1, col=2,
        )
        for r in rows:
            fig.add_annotation(
                x=r["tps_user"], y=r["tps_gpu"],
                xanchor="left", yanchor="middle",
                xshift=8,
                text=f"c={r['concurrency']}",
                showarrow=False,
                font=dict(family=MONO, size=9, color=s["color"]),
                row=1, col=2,
            )

    # Prefix-reuse note in the lower left of the Pareto panel.
    # Tufte block: subtle dark wash, hairline border, white display-sans.
    rr_prefix = sum(r["prefix"] for r in data["round_robin"]) / len(data["round_robin"])
    kv_prefix = sum(r["prefix"] for r in data["kv_router"]) / len(data["kv_router"])
    fig.add_annotation(
        x=0.02, y=0.04,
        xref="x2 domain", yref="y2 domain",
        xanchor="left", yanchor="bottom",
        align="left",
        text=f"<b>Prefix Reuse</b><br>"
             f"Round Robin {rr_prefix:.2f} · KV Router {kv_prefix:.2f}",
        showarrow=False,
        bgcolor="rgba(20,20,20,0.65)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        borderpad=10,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=12, color=TEXT_PRIMARY, weight=300),
    )

    # Inline panel labels.
    for i, label in enumerate(("Mean TTFT (ms, log)", "Throughput vs Interactivity")):
        idx = "" if i == 0 else str(i + 1)
        fig.add_annotation(
            x=0.0, y=1.04,
            xref=f"x{idx} domain", yref=f"y{idx} domain",
            xanchor="left", yanchor="bottom",
            text=label, showarrow=False,
            font=dict(family=SANS, size=11, color=TEXT_MUTED),
        )

    fig.update_layout(
        template=dynamo_template,
        title=dict(
            text="Round-Robin vs KV Router: Concurrency Sweep and Pareto Curve",
            x=0.02, xanchor="left",
            y=0.96, yanchor="top",
            font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                      size=42, color=TEXT_PRIMARY, weight=300),
        ),
        legend=dict(
            orientation="h",
            x=1.0, xanchor="right",
            y=-0.18, yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(family=SANS, size=12, color=TEXT_SECONDARY),
            itemsizing="constant",
            tracegroupgap=18,
        ),
        margin=dict(l=80, r=40, t=180, b=120),
        width=1240, height=560,
        shapes=[],
    )

    # Subtitle parked 5px below the title's bottom edge. Derived from
    # the standard formula in plotting.md:
    #   title_top    = (1 - 0.96) * 560 = 22.4
    #   title_bottom = 22.4 + 42 * 1.00 = 64.4   # 1.00 = cap + descender
    #   subtitle_top = 64.4 + 2         = 66.4   # +2 px = snug
    #   plot_h       = 560 - 180 - 120  = 260
    #   paper_y      = 1 + (180 - 66.4) / 260 = 1.437
    fig.add_annotation(
        x=-0.049, y=1.437,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text="B200 / MiniMax-M2.5 / TP=4 / 8 workers / Mooncake trace — KV Router reduces TTFT across all concurrencies and lifts throughput/GPU.",
        showarrow=False,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=22, color=TEXT_MUTED, weight=300),
    )

    # Left panel axes.
    concs = sorted({r["concurrency"] for r in data["round_robin"]})
    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=concs,
        ticktext=[f"c={v}" for v in concs],
        title=dict(text="Concurrency",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED), standoff=8),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        range=[1.7, 2.85],  # log10: covers c=50..c=700
        row=1, col=1,
    )
    fig.update_yaxes(
        type="log",
        tickmode="array",
        tickvals=[200, 500, 1000, 2000, 5000],
        ticktext=["200", "500", "1k", "2k", "5k"],
        title=dict(text="Mean TTFT (ms, Log)",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED)),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        range=[2.15, 3.85],
        row=1, col=1,
    )

    # Right panel axes.
    fig.update_xaxes(
        title=dict(text="Interactivity (Tok/s/User)",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED), standoff=8),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        range=[10, 95],
        row=1, col=2,
    )
    fig.update_yaxes(
        title=dict(text="Throughput (Tok/s/GPU)",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED)),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        range=[80, 220],
        row=1, col=2,
    )

    out_svg = HERE.parent / "images" / "fig-7-router-experiment.svg"
    out_png = HERE.parent / "images" / "fig-7-router-experiment.png"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_svg))
    fig.write_image(str(out_png), scale=2)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
