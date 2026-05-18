#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate fig-7: KVBM G2 host-memory cache tier (kvbm_g2_exp redraw).

Left panel: mean TTFT (log y) vs concurrency. Right panel: throughput
(TPS/GPU) vs interactivity (TPS/User) Pareto. Baseline G1-only in muted
gray, G2-enabled in NV green.

Data: tools/data_kvbm_g2.csv (copied from PR 9139).
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
SANS = TY["font_family"]
MONO = TY["font_family_mono"]

SERIES = {
    "baseline": dict(name="Baseline (G1 Only)",  color="#909090"),
    "g2":       dict(name="With G2 (32K Blocks)", color=NV_GREEN),
}


def load_csv(path: Path) -> dict[str, list[dict]]:
    by_src: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_src[row["source"]].append({
                "concurrency": int(row["concurrency"]),
                "tps_gpu":     float(row["tps_gpu"]),
                "tps_user":    float(row["tps_user"]),
                "ttft_ms":     float(row["ttft_ms"]),
            })
    for s in by_src:
        by_src[s].sort(key=lambda r: r["concurrency"])
    return by_src


def main() -> None:
    data = load_csv(HERE / "data_kvbm_g2.csv")

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)

    # ---------------- Left: Mean TTFT vs Concurrency ----------------
    for src in ("baseline", "g2"):
        s = SERIES[src]
        rows = data[src]
        fig.add_trace(
            go.Scatter(
                x=[r["concurrency"] for r in rows],
                y=[r["ttft_ms"] for r in rows],
                mode="lines+markers",
                name=s["name"], legendgroup=src,
                line=dict(color=s["color"], width=2.0),
                marker=dict(color=s["color"], symbol="diamond" if src == "g2" else "circle",
                            size=10, line=dict(color=C["background"]["primary"], width=1)),
                hovertemplate=f"<b>{s['name']}</b><br>c=%{{x}}<br>%{{y:,.0f}} ms<extra></extra>",
            ),
            row=1, col=1,
        )

    bl = {r["concurrency"]: r for r in data["baseline"]}
    g2 = {r["concurrency"]: r for r in data["g2"]}
    pct_xs, pct_ys, pct_texts = [], [], []
    for c in sorted(g2):
        pct = (g2[c]["ttft_ms"] - bl[c]["ttft_ms"]) / bl[c]["ttft_ms"] * 100
        pct_xs.append(c)
        pct_ys.append(g2[c]["ttft_ms"] * 0.78)
        pct_texts.append(f"<b>{pct:.1f}%</b>")
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
    for src in ("baseline", "g2"):
        s = SERIES[src]
        rows = data[src]
        fig.add_trace(
            go.Scatter(
                x=[r["tps_user"] for r in rows],
                y=[r["tps_gpu"] for r in rows],
                mode="lines+markers",
                name=s["name"], legendgroup=src, showlegend=False,
                line=dict(color=s["color"], width=2.0),
                marker=dict(color=s["color"], symbol="diamond" if src == "g2" else "circle",
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
            text="Baseline vs KVBM G2: Concurrency Sweep and Pareto Curve",
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

    # Subtitle: 22pt, parked 2px below the title's descender bottom (snug).
    # Uses font-size * 1.00 as the title height (cap + descender), not 0.80,
    # so titles with g/y/p/j don't collide with the subtitle.
    #   title_top    = (1 - 0.96) * 560 = 22.4
    #   title_bottom = 22.4 + 42 * 1.00 = 64.4
    #   subtitle_top = 64.4 + 2         = 66.4   # +2 px = snug
    #   plot_h       = 560 - 180 - 120  = 260
    #   paper_y      = 1 + (180 - 66.4) / 260 = 1.437
    fig.add_annotation(
        x=-0.049, y=1.437,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text="B200 / MiniMax-M2.5 / TP=4 / 1 worker / Mooncake trace — KVBM G2 reduces TTFT across all concurrencies vs baseline.",
        showarrow=False,
        font=dict(family="Helvetica Neue, HelveticaNeue, sans-serif",
                  size=22, color=TEXT_MUTED, weight=300),
    )

    concs = sorted({r["concurrency"] for r in data["baseline"]})
    fig.update_xaxes(
        type="log",
        tickmode="array",
        tickvals=concs,
        ticktext=[f"c={v}" for v in concs],
        title=dict(text="Concurrency",
                   font=dict(family=SANS, size=11, color=TEXT_MUTED), standoff=8),
        showline=True, linecolor=BORDER_SUBTLE, linewidth=0.5, mirror=True,
        ticks="", showgrid=True, gridcolor=BORDER_SUBTLE, gridwidth=0.5,
        range=[0.8, 1.9],  # log10: covers c=6..c=80
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
        range=[2.15, 3.45],
        row=1, col=1,
    )

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
        range=[80, 210],
        row=1, col=2,
    )

    out_svg = HERE.parent / "images" / "fig-8-kvbm-g2.svg"
    out_png = HERE.parent / "images" / "fig-8-kvbm-g2.png"
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_svg))
    fig.write_image(str(out_png), scale=2)
    print(f"wrote {out_svg}")
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
