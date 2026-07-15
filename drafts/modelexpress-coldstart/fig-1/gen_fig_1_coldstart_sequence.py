#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""fig-1 — ModelExpress cold-start sequence, in the Dynamo Dark aesthetic.

A sequence diagram (rethought from a Mermaid-style reference, not a pixel copy)
showing how a new replica warms up under ModelExpress: it registers a source and
discovers candidates *through the metadata server* (metadata plane, grey), then
pulls weights *straight from a peer over RDMA* (data plane, green). The bytes
never touch the server.

Three lifelines:
  - Source replica (already serving)    -- neutral GPU actor        (left)
  - ModelExpress server (metadata only) -- cpu-blue control spine    (center)
  - New replica (target)                -- neutral GPU actor         (right)

Two semantic accents only: cpu_blue marks the metadata-only server; dynamo green
marks the single peer-to-peer data path. Everything else is neutral grey, so the
green RDMA edge reads as the one thing that matters -- and it crosses the server
spine *without terminating on it*, which is the whole argument of the figure.
Every metadata message lands on the server with a small contact node; the green
weight-transfer passes clean through it.

Title uses the Dynamo Dark display/hero treatment (Helvetica Neue Light, title
case). All labels are protocol/API terms carried over from the reference
(PublishMetadata, ListSources, GetMetadata, UpdateStatus, NIXL, mx_source_id,
tensor descriptors) -- no invented numbers. Renders deterministically.

Usage:
    python3 gen_fig_1_coldstart_sequence.py   # -> images/fig-1-modelexpress-coldstart.{png,svg}
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# ── Lifeline x-positions on a 0..100 canvas ──────────────────────────────────
X_SRC = 19.0  # Source replica (already serving)
X_SRV = 52.0  # ModelExpress server (metadata only)
X_TGT = 85.0  # New replica (target)

LIFE_TOP = 90.0  # lifelines drop from the bottom edge of the header cards
LIFE_BOT = 12.5  # ... to a common baseline
GAP = 1.1  # shaft end-gap / arrowhead offset from a lifeline


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]

    green = colors["accent"]["dynamo_green"]  # data plane (single accent)
    blue = colors["accent"]["cpu_blue"]  # metadata-only server
    blue_fill = colors["fills"]["blue"]  # server card interior
    grey = colors["text"]["medium"]  # neutral metadata flows
    subtle = colors["border"]["subtle"]  # hairlines, loop frame
    surface = colors["background"]["surface"]  # neutral card fill
    surface_alt = colors["background"]["surface_alt"]  # loop tag fill
    text_primary = colors["text"]["primary"]
    text_secondary = colors["text"]["secondary"]
    text_muted = colors["text"]["muted"]
    font_sans = tokens["typography"]["font_family"]
    font_mono = tokens["typography"]["font_family_mono"]

    fig = go.Figure()
    shapes: list[dict] = []
    annotations: list[dict] = []
    heads: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"x": [], "y": []}
    )
    nodes: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"x": [], "y": []})

    # ── Primitives ───────────────────────────────────────────────────────────
    def node(cx: float, y: float, color: str) -> None:
        nodes[color]["x"].append(cx)
        nodes[color]["y"].append(y)

    def lifeline(cx: float, color: str, dash: str) -> None:
        shapes.append(
            dict(
                type="line",
                x0=cx,
                y0=LIFE_TOP,
                x1=cx,
                y1=LIFE_BOT,
                line=dict(color=color, width=1.3, dash=dash),
                layer="below",
            )
        )

    def header(cx: float, name: str, sub: str, accent: str, fill: str) -> None:
        hw = 15.0
        shapes.append(
            dict(
                type="rect",
                x0=cx - hw,
                x1=cx + hw,
                y0=90.0,
                y1=98.0,
                line=dict(color=accent, width=1.5),
                fillcolor=fill,
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=cx,
                y=95.4,
                text=f"<b>{name}</b>",
                showarrow=False,
                font=dict(family=font_sans, size=15, color=text_primary),
                xanchor="center",
                yanchor="middle",
            )
        )
        annotations.append(
            dict(
                x=cx,
                y=92.2,
                text=sub,
                showarrow=False,
                font=dict(family=font_sans, size=12, color=text_muted),
                xanchor="center",
                yanchor="middle",
            )
        )

    def message(
        y: float,
        x_from: float,
        x_to: float,
        color: str,
        label: str,
        *,
        dashed: bool = False,
        width: float = 1.7,
        label_color: str | None = None,
        italic: bool = False,
        label_dy: float = 2.1,
        cross_node: bool = True,
        sublabel: str | None = None,
        sublabel_color: str | None = None,
    ) -> None:
        """One horizontal message: shaft (line shape) + arrowhead (marker) +
        contact nodes at both lifelines + a centered label above the shaft."""
        rightward = x_to > x_from
        ax = x_from + GAP if rightward else x_from - GAP
        xh = x_to - GAP if rightward else x_to + GAP
        shapes.append(
            dict(
                type="line",
                x0=ax,
                y0=y,
                x1=xh,
                y1=y,
                line=dict(color=color, width=width, dash="dot" if dashed else "solid"),
                layer="above",
            )
        )
        sym = "triangle-right" if rightward else "triangle-left"
        heads[(color, sym)]["x"].append(xh)
        heads[(color, sym)]["y"].append(y)
        if cross_node:
            node(x_from, y, color)
            node(x_to, y, color)
        lc = label_color or text_secondary
        txt = f"<i>{label}</i>" if italic else label
        annotations.append(
            dict(
                x=(x_from + x_to) / 2,
                y=y + label_dy,
                text=txt,
                showarrow=False,
                font=dict(family=font_mono, size=13, color=lc),
                xanchor="center",
                yanchor="middle",
            )
        )
        if sublabel:
            annotations.append(
                dict(
                    x=(x_from + x_to) / 2,
                    y=y - label_dy + 0.3,
                    text=sublabel,
                    showarrow=False,
                    font=dict(family=font_sans, size=12, color=sublabel_color or lc),
                    xanchor="center",
                    yanchor="middle",
                )
            )

    # ── Lifelines + participant headers ───────────────────────────────────────
    lifeline(X_SRC, subtle, "solid")
    lifeline(X_SRV, blue, "dash")  # metadata-only spine reads as control plane
    lifeline(X_TGT, subtle, "solid")
    header(X_SRC, "Source replica", "already serving", subtle, surface)
    header(X_SRV, "ModelExpress server", "metadata only", blue, blue_fill)
    header(X_TGT, "New replica", "target", subtle, surface)

    # ── Phase 1 · REGISTER — source announces itself to the metadata server ────
    # M1: source self-action (load + register with NIXL), drawn as a self-loop.
    y1 = 84.5
    loop_r = X_SRC + 7.0
    y1b = y1 - 2.6
    fig.add_trace(
        go.Scatter(
            x=[X_SRC, loop_r, loop_r, X_SRC],
            y=[y1, y1, y1b, y1b],
            mode="lines",
            line=dict(color=grey, width=1.5),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    heads[(grey, "triangle-left")]["x"].append(X_SRC + GAP)
    heads[(grey, "triangle-left")]["y"].append(y1b)
    node(X_SRC, y1, grey)
    node(X_SRC, y1b, grey)
    annotations.append(
        dict(
            x=loop_r + 1.8,
            y=(y1 + y1b) / 2,
            text="Load + post-process weights,<br>register GPU memory with NIXL",
            showarrow=False,
            font=dict(family=font_mono, size=13, color=text_secondary),
            xanchor="left",
            yanchor="middle",
        )
    )

    # M2: PublishMetadata -> the server mints an mx_source_id for this worker.
    message(77.0, X_SRC, X_SRV, grey, "PublishMetadata(identity) → mx_source_id")

    # M3: UpdateStatus(READY), repeated — wrapped in a UML loop fragment.
    lb_x0, lb_x1, lb_y0, lb_y1 = X_SRC - 6.0, X_SRV + 6.0, 65.0, 73.0
    shapes.append(
        dict(
            type="rect",
            x0=lb_x0,
            x1=lb_x1,
            y0=lb_y0,
            y1=lb_y1,
            line=dict(color=subtle, width=1),
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        )
    )
    tab_w, tab_h = 9.0, 2.6
    shapes.append(
        dict(
            type="rect",
            x0=lb_x0,
            x1=lb_x0 + tab_w,
            y0=lb_y1 - tab_h,
            y1=lb_y1,
            line=dict(color=subtle, width=1),
            fillcolor=surface_alt,
            layer="below",
        )
    )
    annotations.append(
        dict(
            x=lb_x0 + tab_w / 2,
            y=lb_y1 - tab_h / 2,
            text="loop",
            showarrow=False,
            font=dict(family=font_sans, size=11, color=text_secondary),
            xanchor="center",
            yanchor="middle",
        )
    )
    message(69.0, X_SRC, X_SRV, grey, "UpdateStatus(READY)")

    # ── Phase 2 · DISCOVER — target queries the metadata server ────────────────
    # M4: ListSources request (target -> server), filtered to the target's rank.
    message(56.0, X_TGT, X_SRV, grey, "ListSources(mx_source_id, READY) · own rank")
    # M5: server returns the candidate source workers (dashed = response).
    message(
        50.0, X_SRV, X_TGT, grey, "candidate source workers", dashed=True, italic=True
    )
    # M6: GetMetadata request for a chosen worker.
    message(44.0, X_TGT, X_SRV, grey, "GetMetadata(worker)")
    # M7: server returns tensor descriptors + NIXL connection info (dashed).
    message(
        38.0,
        X_SRV,
        X_TGT,
        grey,
        "tensor descriptors + NIXL connection info",
        dashed=True,
        italic=True,
    )

    # ── Phase 3 · TRANSFER — the data plane, peer-to-peer, around the server ───
    # M8: RDMA read of the weights, GPU to GPU. THE single green accent. It runs
    # target -> source and crosses the server spine WITHOUT a contact node
    # (cross_node=False for the server) -- the bytes bypass the server.
    y_rdma = 27.0
    message(
        y_rdma,
        X_TGT,
        X_SRC,
        green,
        "RDMA read of weights · GPU to GPU",
        width=3.0,
        label_color=green,
        sublabel="the bytes bypass the ModelExpress server",
        sublabel_color=green,
    )
    node(X_TGT, y_rdma, green)
    node(X_SRC, y_rdma, green)

    # M9: target publishes its own metadata — it becomes a new source (the swarm
    # grows). Back on the metadata plane, so grey again.
    message(18.0, X_TGT, X_SRV, grey, "PublishMetadata() · target becomes a new source")

    # ── Phase rail (left) ──────────────────────────────────────────────────────
    phases = [
        ("1", "REGISTER", 73.0, 87.0, grey),
        ("2", "DISCOVER", 36.0, 60.0, grey),
        ("3", "TRANSFER", 15.0, 33.0, green),
    ]
    for num, name, yb0, yb1, col in phases:
        ymid = (yb0 + yb1) / 2
        shapes.append(
            dict(
                type="line",
                x0=2.2,
                y0=yb0,
                x1=2.2,
                y1=yb1,
                line=dict(color=col, width=3),
                layer="above",
            )
        )
        annotations.append(
            dict(
                x=4.6,
                y=ymid + 2.4,
                text=f"<b>{num}</b>",
                showarrow=False,
                font=dict(family=font_sans, size=19, color=col, weight=700),
                xanchor="left",
                yanchor="middle",
            )
        )
        annotations.append(
            dict(
                x=4.6,
                y=ymid - 2.1,
                text=f"<b>{name}</b>",
                showarrow=False,
                font=dict(family=font_sans, size=12, color=col, weight=700),
                xanchor="left",
                yanchor="middle",
            )
        )

    # ── Legend (bottom-center): the two planes ─────────────────────────────────
    def legend(items: list[tuple[str, str]], y: float) -> None:
        sw, sw_gap, item_gap, char_w = 3.4, 1.4, 5.0, 0.52
        widths = [sw + sw_gap + len(lbl) * char_w for _, lbl in items]
        total = sum(widths) + item_gap * (len(items) - 1)
        x = (100.0 - total) / 2.0
        for (col, lbl), w in zip(items, widths):
            shapes.append(
                dict(
                    type="line",
                    x0=x,
                    x1=x + sw,
                    y0=y,
                    y1=y,
                    line=dict(color=col, width=3),
                    layer="above",
                )
            )
            annotations.append(
                dict(
                    x=x + sw + sw_gap,
                    y=y,
                    text=lbl,
                    showarrow=False,
                    font=dict(family=font_sans, size=12.5, color=text_secondary),
                    xanchor="left",
                    yanchor="middle",
                )
            )
            x += w + item_gap

    legend(
        [
            (grey, "Metadata plane — routed through ModelExpress"),
            (green, "Data plane — peer-to-peer RDMA, GPU to GPU"),
        ],
        7.5,
    )

    # ── Arrowheads + contact nodes as marker traces (crisp squares/triangles) ──
    for (col, sym), pts in heads.items():
        fig.add_trace(
            go.Scatter(
                x=pts["x"],
                y=pts["y"],
                mode="markers",
                marker=dict(symbol=sym, size=12, color=col),
                showlegend=False,
                hoverinfo="skip",
            )
        )
    for col, pts in nodes.items():
        fig.add_trace(
            go.Scatter(
                x=pts["x"],
                y=pts["y"],
                mode="markers",
                marker=dict(symbol="square", size=8, color=col),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        template=template,
        title=dict(
            text="Cold Start: The Data Plane Bypasses the Metadata Server",
            font=dict(family=HERO_FONT, size=40, color=text_primary, weight=300),
            subtitle=dict(
                text=(
                    "ModelExpress cold start — weights move peer-to-peer over "
                    "RDMA; the server only ever brokers metadata."
                ),
                font=dict(family=HERO_FONT, size=18, color=text_muted, weight=300),
            ),
            x=0.02,
            xanchor="left",
            y=0.955,
            yanchor="top",
        ),
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(range=[4, 100], visible=False),
        width=1600,
        height=950,
        margin=dict(l=30, r=30, t=150, b=40),
        shapes=shapes,
        annotations=annotations,
        showlegend=False,
    )

    out_dir = Path(__file__).parent / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "fig-1-modelexpress-coldstart.png"
    svg = out_dir / "fig-1-modelexpress-coldstart.svg"
    fig.write_image(str(png), scale=2)
    fig.write_image(str(svg))
    print(f"Wrote {png}")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
