#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Generate the 4-panel B200 MiniMax-M2.5 fidelity figure for the digital-twin blog.

Reads data.csv and writes images/hw_mocker_aic_4panel.svg via the shared
Dynamo Dark Plotly template.

Panels: TPS/GPU, TPS/User, TPOT, TTFT vs concurrency.
Sources: Hardware (white solid, ground truth), Mocker (NV green),
AIC (CPU blue dashed).

c=4 hardware TTFT is a known cold-start outlier; excluded by default.
Concurrency axis is log-2 across c={8, 16, 32, 64}.

The figure intentionally carries no callout boxes, no footer captions, no
MAPE-per-panel boxes. The fidelity claim lives in the legend label and the
contextual prose around the figure in the blog.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotly_dynamo import dynamo_template, load_tokens

HERE = Path(__file__).resolve().parent
TOKENS = load_tokens()


def load_csv(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    out: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            c = int(row["concurrency"])
            out[row["source"]][c] = {
                "tps_gpu": float(row["tps_gpu"]),
                "tps_user": float(row["tps_user"]),
                "tpot_ms": float(row["tpot_ms"]),
                "ttft_ms": float(row["ttft_ms"]),
            }
    return out


def mape(pred: list[float], truth: list[float]) -> float:
    return 100.0 * sum(abs(p - t) / t for p, t in zip(pred, truth)) / len(truth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-c4", action="store_true")
    parser.add_argument("--out", default=str(HERE.parent / "images" / "fig-3-fidelity.svg"))
    args = parser.parse_args()

    data = load_csv(HERE / "data.csv")
    concs = sorted(data["hardware"].keys())
    if not args.keep_c4:
        concs = [c for c in concs if c != 4]

    colors = TOKENS["colors"]
    accent = colors["accent"]
    typo = TOKENS["typography"]
    text_primary = colors["text"]["primary"]
    text_secondary = colors["text"]["secondary"]
    text_muted = colors["text"]["muted"]

    # Compute MAPE ranges across the 4 metrics, used in legend labels.
    metrics = ["tps_gpu", "tps_user", "tpot_ms", "ttft_ms"]
    mocker_mapes = [mape([data["mocker"][c][m] for c in concs],
                         [data["hardware"][c][m] for c in concs]) for m in metrics]
    aic_mapes = [mape([data["aic"][c][m] for c in concs],
                      [data["hardware"][c][m] for c in concs]) for m in metrics]
    mocker_lo, mocker_hi = min(mocker_mapes), max(mocker_mapes)
    aic_lo,    aic_hi    = min(aic_mapes),    max(aic_mapes)

    # All series solid -- color (white/green/blue) carries the distinction.
    # Lines are thin so markers carry the data and the line just traces order.
    SOURCE_STYLE = {
        "hardware": dict(name="Hardware",
                         color=text_primary,           dash="solid", symbol="circle",  width=1.6),
        "mocker":   dict(name=f"DynoSim · MAPE {mocker_lo:.0f}–{mocker_hi:.0f}%",
                         color=accent["dynamo_green"], dash="solid", symbol="diamond", width=1.4),
        "aic":      dict(name=f"AIC · MAPE {aic_lo:.0f}–{aic_hi:.0f}%",
                         color=accent["cpu_blue"],     dash="solid", symbol="square",  width=1.4),
    }
    SOURCE_ORDER = ["hardware", "mocker", "aic"]

    PANELS = [
        ("tps_gpu",  "Output TPS / GPU"),
        ("tps_user", "Output TPS / User"),
        ("tpot_ms",  "Mean TPOT (ms)"),
        ("ttft_ms",  "Mean TTFT (ms)"),
    ]
    # Y-axis ranges chosen so 0 is always on the grid and tick spacing
    # yields a clean evenly-divided grid for each panel.
    YAXIS = {
        (1, 1): dict(range=[0,  800], dtick=200),
        (1, 2): dict(range=[0,  100], dtick=25),
        (2, 1): dict(range=[0,   25], dtick=5),
        (2, 2): dict(range=[0,  250], dtick=50),
    }

    fig = make_subplots(
        rows=2, cols=2,
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    # Plot
    for i, (metric, _label) in enumerate(PANELS):
        row, col = i // 2 + 1, i % 2 + 1
        for src in SOURCE_ORDER:
            s = SOURCE_STYLE[src]
            ys = [data[src][c][metric] for c in concs]
            fig.add_trace(
                go.Scatter(
                    x=concs, y=ys,
                    mode="lines+markers",
                    name=s["name"],
                    legendgroup=src,
                    showlegend=(i == 0),
                    line=dict(color=s["color"], dash=s["dash"], width=s["width"]),
                    marker=dict(symbol=s["symbol"], size=8,
                                color=s["color"],
                                line=dict(color=colors["background"]["primary"], width=1.0)),
                    hovertemplate=f"<b>{src.title()}</b><br>c=%{{x}}<br>%{{y:.2f}}<extra></extra>",
                ),
                row=row, col=col,
            )

    # Inline panel label, top-left of each panel — small, muted, regular weight.
    # Sits in the gap above each panel; high enough to clear the panel itself
    # but below the figure title.
    for i, (_metric, label) in enumerate(PANELS):
        idx = "" if i == 0 else str(i + 1)
        fig.add_annotation(
            x=0.0, y=1.04,
            xref=f"x{idx} domain", yref=f"y{idx} domain",
            xanchor="left", yanchor="bottom",
            text=label,
            showarrow=False,
            font=dict(family=typo["font_family"], size=11, color=text_muted),
        )

    # Title at top, panels below with inline labels, legend at the very bottom.
    fig.update_layout(
        template=dynamo_template,
        title=dict(
            text="Concurrency Sweep: DynoSim, AIC, Hardware",
            x=0.02, xanchor="left",
            y=0.96, yanchor="top",
            font=dict(
                family="Helvetica Neue, HelveticaNeue, sans-serif",
                size=42, color=text_primary, weight=300,
            ),
        ),
        legend=dict(
            orientation="h",
            x=1.0, xanchor="right",
            y=-0.14, yanchor="top",
            bgcolor="rgba(0,0,0,0)",
            font=dict(family=typo["font_family"], size=12, color=text_secondary),
            itemsizing="constant",
            tracegroupgap=18,
        ),
        margin=dict(l=80, r=40, t=130, b=70),
        width=1240, height=620,
        shapes=[],
    )

    # Subtitle: 22pt, parked exactly 5px below the title's bottom edge.
    # Empirically y=1.155 sits ~5px below the rendered title bottom for a
    # 42pt title at figure-y=0.96 with height=620 / top_margin=130 / plot_h=420.
    #   paper_x = (title_x*W - margin_l)/plot_w = (0.02*1240 - 80)/1120 = -0.049
    fig.add_annotation(
        x=-0.049, y=1.155,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text="B200 / MiniMax-M2.5 / TP=4 / ISL/OSL 1K/1K — DynoSim closes the AIC–hardware gap to ~50 ms on TTFT.",
        showarrow=False,
        font=dict(
            family="Helvetica Neue, HelveticaNeue, sans-serif",
            size=22, color=text_muted, weight=300,
        ),
    )

    # Log-x on every subplot; throughput panels start at zero, latency panels auto-fit.
    # Each panel carries its own "Concurrency" x-axis title so the unit is
    # unambiguous next to each chart, instead of one shared label at the bottom.
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(
                type="log",
                tickmode="array",
                tickvals=concs,
                ticktext=[str(v) for v in concs],
                title=dict(
                    text="Concurrency",
                    font=dict(family=typo["font_family"], size=11, color=text_muted),
                    standoff=8,
                ),
                showline=True,
                linecolor=colors["border"]["subtle"],
                linewidth=0.5,
                mirror=True,
                ticks="",
                showgrid=True,
                gridcolor=colors["border"]["subtle"],
                gridwidth=0.5,
                row=r, col=c,
            )
            yaxis_cfg = YAXIS[(r, c)]
            fig.update_yaxes(
                range=yaxis_cfg["range"],
                tick0=0,
                dtick=yaxis_cfg["dtick"],
                showline=True,
                linecolor=colors["border"]["subtle"],
                linewidth=0.5,
                mirror=True,
                ticks="",
                showgrid=True,
                gridcolor=colors["border"]["subtle"],
                gridwidth=0.5,
                row=r, col=c,
            )

    # Scheduler-simulation callout, parked in the bottom-right corner of
    # the TTFT panel (subplot 4 -> x4/y4). Tufte-style block: subtle dark
    # wash, hairline border, white display-sans text, left-aligned.
    # The leader Scatter still binds to row=2, col=2 and points from the
    # callout up toward the AIC/HW divergence near (c=64, ttft~190).
    # Vertical "]" bracket marking the AIC -> DynoSim/HW TTFT gap at c=64.
    # AIC c=64 ttft=167.8ms, HW c=64 ttft=220.4ms. Bracket sits just to the
    # right of the data markers with the open side facing left (toward the
    # data). Feet are ~5px wide — a tight tick mark, not a wide span. The
    # gap interpretation lives in the prose callout box below.
    #
    # Drawn as fig.add_shape lines bound to x4/y4 so the bracket is an
    # overlay and the x-axis tick set stays at concs=[8,16,32,64]. Shape
    # coords are RAW data values; Plotly applies the log10 transform
    # automatically when xref binds to a log axis.
    BRACKET_X_SPINE = 68    # just right of the c=64 data markers
    BRACKET_X_FOOT  = 66    # ~5px to the left of the spine on the log axis
    AIC_Y           = 167.8
    HW_Y            = 220.4
    bracket_line = dict(color="rgba(255,255,255,0.85)", width=2.0)
    for x0, y0, x1, y1 in (
        (BRACKET_X_SPINE, HW_Y,  BRACKET_X_FOOT,  HW_Y),   # top foot, opens left
        (BRACKET_X_SPINE, HW_Y,  BRACKET_X_SPINE, AIC_Y),  # spine on the right
        (BRACKET_X_SPINE, AIC_Y, BRACKET_X_FOOT,  AIC_Y),  # bottom foot, opens left
    ):
        fig.add_shape(
            type="line",
            xref="x4", yref="y4",
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=bracket_line,
            layer="above",
        )
    # Pin the TTFT panel x-range so the bracket overlay can't expand it.
    # Log range covers concs=[8..64] plus a tiny pad on the right for the
    # bracket spine at x=68.
    import math
    fig.update_xaxes(
        range=[math.log10(7.2), math.log10(74)],
        row=2, col=2,
    )

    fig.add_annotation(
        xref="x4 domain", yref="y4 domain",
        x=0.98, y=0.04,
        xanchor="right", yanchor="bottom",
        align="left",
        text="<b>Scheduler Simulation</b><br>"
             "DynoSim's scheduler effects close<br>"
             "the gap between AIC and hardware",
        showarrow=False,
        bgcolor="rgba(20,20,20,0.65)",
        bordercolor="rgba(255,255,255,0.18)",
        borderwidth=1,
        borderpad=10,
        font=dict(
            family="Helvetica Neue, HelveticaNeue, sans-serif",
            size=12, color=text_primary, weight=300,
        ),
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out))
    fig.write_image(str(out.with_suffix(".png")), scale=2)
    print(f"wrote {out}")
    print(f"wrote {out.with_suffix('.png')}")
    print()
    print(f"{'Metric':<18} {'Mocker MAPE':>13} {'AIC MAPE':>12}")
    for metric, label in PANELS:
        truth = [data["hardware"][c][metric] for c in concs]
        m_m = mape([data["mocker"][c][metric] for c in concs], truth)
        m_a = mape([data["aic"][c][metric]    for c in concs], truth)
        print(f"{label:<18} {m_m:>12.2f}% {m_a:>11.2f}%")


if __name__ == "__main__":
    main()
