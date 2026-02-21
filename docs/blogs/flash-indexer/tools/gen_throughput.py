#!/usr/bin/env python3
"""Plot achieved vs. offered throughput from mooncake_bench sweep data.

Uses the Dynamo dark Plotly template (design_tokens.yaml + plotly_dynamo.py).

Usage:
    python3 gen_throughput.py ../data/sweep_plot.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

BACKENDS = {
    "nested-map": {
        "label": "Concurrent Positional Indexer (Flash Indexer)",
        "color": "#76b900",  # NVIDIA Green -- best performer
        "version": "Dynamo v1.0.0",
    },
    "concurrent-radix-tree": {
        "label": "Concurrent Radix Tree",
        "color": "#008564",  # Emerald -- strong second
        "version": "Dynamo v1.0.0",
    },
    "radix-tree": {
        "label": "Radix Tree",
        "color": "#fac200",  # Fluorite -- middle
        "version": "Dynamo v0.1.0",
    },
    "inverted-index": {
        "label": "Inverted Index",
        "color": "#969696",  # Silver Gray (7.1:1 on black) -- weak
        "version": "Naive",
    },
    "naive-nested-map": {
        "label": "Naive Nested Map",
        "color": "#767676",  # Muted Gray (4.6:1 on black) -- worst
        "version": "Naive",
    },
}

PLOT_ORDER = list(BACKENDS.keys())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    images_dir = Path(__file__).resolve().parent.parent / "images"
    default_out = str(images_dir / "fig-6-indexer-throughput.png")
    parser.add_argument("json_path", help="Path to sweep_plot.json")
    parser.add_argument(
        "-o", "--output", default=default_out, help="Output path (.png, .svg, .pdf)"
    )
    parser.add_argument("--width", type=int, default=775, help="Figure width in px")
    parser.add_argument("--height", type=int, default=650, help="Figure height in px")
    args = parser.parse_args()

    with open(args.json_path) as f:
        data = json.load(f)

    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    series_colors = tokens["colors"]["chart_series"]
    border_subtle = tokens["colors"]["border"]["subtle"]

    ordered = [(k, data[k]) for k in PLOT_ORDER if k in data]
    for k in data:
        if k not in PLOT_ORDER:
            ordered.append((k, data[k]))

    fig = go.Figure()

    global_min = float("inf")
    global_max = float("-inf")
    for _, steps in ordered:
        for s in steps:
            offered = s["offered_block_throughput"]
            achieved = s["block_throughput"]
            global_min = min(global_min, offered, achieved)
            global_max = max(global_max, offered, achieved)

    axis_min = 1e5  # 100k fixed floor
    x_max = 2e9  # 2.0G
    y_max = 5e8  # 500M

    fig.add_trace(
        go.Scatter(
            x=[axis_min, x_max],
            y=[axis_min, x_max],
            mode="lines",
            line=dict(color=border_subtle, width=1, dash="dash"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    for i, (name, steps) in enumerate(ordered):
        meta = BACKENDS.get(name, {})
        color = meta.get("color", series_colors[i % len(series_colors)])
        display = meta.get("label", name.replace("-", " ").title())

        offered = [s["offered_block_throughput"] for s in steps]
        achieved = [s["block_throughput"] for s in steps]

        fig.add_trace(
            go.Scatter(
                x=offered,
                y=achieved,
                mode="lines+markers",
                name=display,
                line=dict(color=color, width=2.5),
                marker=dict(size=7, color=color),
                hovertemplate=(
                    f"<b>{display}</b><br>"
                    "Offered: %{x:.2s} ops/s<br>"
                    "Achieved: %{y:.2s} ops/s"
                    "<extra></extra>"
                ),
            )
        )

    peak_annotations = []
    peak_shapes = []
    peak_values = {}

    label_x = 1e9  # 1.0G mark

    peak_header_added = False

    for name, steps in ordered:
        meta = BACKENDS.get(name, {})
        color = meta.get("color", "#ffffff")
        version = meta.get("version", "")
        best = max(steps, key=lambda s: s["block_throughput"])
        peak_y = best["block_throughput"]
        peak_x = best["offered_block_throughput"]

        if not peak_header_added:
            peak_annotations.append(
                dict(
                    x=_log10(label_x),
                    y=_log10(peak_y),
                    xref="x",
                    yref="y",
                    text="PEAK THROUGHPUT",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    yshift=24,
                    font=dict(
                        family=tokens["typography"]["font_family"],
                        size=10,
                        color="#ffffff",
                    ),
                )
            )
            peak_header_added = True

        label_text = f"<b>{_fmt_si(peak_y)}</b>"
        peak_annotations.append(
            dict(
                x=_log10(label_x),
                y=_log10(peak_y),
                xref="x",
                yref="y",
                text=label_text,
                showarrow=False,
                xanchor="center",
                yanchor="bottom",
                yshift=6,
                font=dict(
                    family=tokens["typography"]["font_family_mono"],
                    size=12,
                    color=color,
                ),
            )
        )

        if version:
            peak_annotations.append(
                dict(
                    x=_log10(label_x),
                    y=_log10(peak_y),
                    xref="x",
                    yref="y",
                    text=f"<b>{version}</b>",
                    showarrow=False,
                    xanchor="center",
                    yanchor="top",
                    yshift=4,
                    font=dict(
                        family=tokens["typography"]["font_family"],
                        size=9,
                        color=color,
                    ),
                    opacity=1.0,
                )
            )

        peak_values[name] = peak_y

        y_up = peak_y * 1.06
        peak_shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=peak_x,
                x1=label_x,
                y0=y_up,
                y1=y_up,
                line=dict(color=color, width=2, dash="dot"),
                layer="above",
            )
        )

    # ] bracket from pixel editor (bracket_editor_px.html)
    # Pixel coords on 775x650 canvas → Plotly paper-x / data-y
    top_peak = peak_values.get("nested-map", 170e6)
    bot_peak = peak_values.get("radix-tree", 4e6)
    improvement = top_peak / bot_peak

    _margin_l, _margin_t, _margin_b = 60, 70, 60
    _plot_w = args.width - _margin_l - 55  # r=55
    _plot_h = args.height - _margin_t - _margin_b
    _y_log_lo = _log10(axis_min)
    _y_log_hi = _log10(y_max)

    def _px_to_paper_x(px: float) -> float:
        return float((px - _margin_l) / _plot_w)

    def _px_to_data_y(px: float) -> float:
        frac = (_margin_t + _plot_h - px) / _plot_h
        return float(10 ** (_y_log_lo + frac * (_y_log_hi - _y_log_lo)))

    br_x = 723
    br_y_top = 131
    br_y_bot = 369
    br_tick = 16
    br_lx_shift = 6

    bx_paper = _px_to_paper_x(br_x)
    tick_paper = br_tick / _plot_w
    top_y = _px_to_data_y(br_y_top)
    bot_y = _px_to_data_y(br_y_bot)
    mid_y = _px_to_data_y((br_y_top + br_y_bot) / 2)

    for y_val in [top_y, bot_y]:
        peak_shapes.append(
            dict(
                type="line",
                xref="paper",
                yref="y",
                x0=bx_paper,
                x1=bx_paper - tick_paper,
                y0=y_val,
                y1=y_val,
                line=dict(color="#cdcdcd", width=1.5),
                layer="above",
            )
        )
    peak_shapes.append(
        dict(
            type="line",
            xref="paper",
            yref="y",
            x0=bx_paper,
            x1=bx_paper,
            y0=bot_y,
            y1=top_y,
            line=dict(color="#cdcdcd", width=1.5),
            layer="above",
        )
    )
    peak_annotations.append(
        dict(
            xref="paper",
            yref="y",
            x=bx_paper,
            y=_log10(mid_y),
            text=f"{improvement:.0f}×",
            showarrow=False,
            xanchor="left",
            xshift=br_lx_shift,
            font=dict(
                family=tokens["typography"]["font_family"],
                size=10,
                color="#ffffff",
            ),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(text="ACHIEVED VS. OFFERED THROUGHPUT  (HIGHER IS BETTER)"),
        xaxis=dict(
            title="Offered Throughput (block ops/s)",
            type="log",
            range=[_log10(axis_min), _log10(x_max)],
            tickformat=".2s",
        ),
        yaxis=dict(
            title="Achieved Throughput (block ops/s)",
            type="log",
            range=[_log10(axis_min), _log10(y_max)],
            tickformat=".2s",
        ),
        legend=dict(
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
        ),
        width=args.width,
        height=args.height,
        margin=dict(l=60, r=55, t=70, b=60),
        annotations=peak_annotations,
        shapes=peak_shapes,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    png = out.with_suffix(".png")
    svg = out.with_suffix(".svg")
    fig.write_image(str(png), scale=3)
    fig.write_image(str(svg))
    print(f"Wrote {png.name}  ({args.width}x{args.height})")
    print(f"Wrote {svg.name}  ({args.width}x{args.height})")


def _log10(x: float) -> float:
    import math

    return math.log10(x)


def _fmt_si(v: float) -> str:
    if v >= 1e9:
        return f"{v / 1e9:.1f}G"
    if v >= 1e6:
        return f"{v / 1e6:.0f}M"
    if v >= 1e3:
        return f"{v / 1e3:.0f}K"
    return f"{v:.0f}"


if __name__ == "__main__":
    main()
