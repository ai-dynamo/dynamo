#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""fig-3 — ModelExpress cold-start architecture in the Dynamo Dark aesthetic.

A DIAGRAM (not a chart), so this is a rethought composition of the reference
rather than a pixel copy. The reference is a two-plane ModelExpress diagram:

  Control plane (top):  Inference Engine (Source) + MX Client, MX Server,
                        Metadata Store (Redis / K8s), Inference Engine (New)
                        + MX Client, wired by thin control links.
  Data plane (bottom):  Object Storage (Remote) and File Storage (Local /
                        Network) feed model weights INTO the newly-scheduled
                        engine over three GPU-direct paths — GPUDirect RDMA
                        (peer engine), ModelStreamer (object store), and
                        GPUDirect Storage / GDS (file store).

Dynamo Dark rethink:
  - Two recessive plane bands (#1a1a1a surface, #3a3a3a hairline) on pure
    black, so the plane separation reads structurally instead of via tinted
    backgrounds.
  - Green (#76b900) is the single selective accent, reserved for the
    data-plane weight-transfer flows (GPUDirect RDMA + ModelStreamer + GDS) —
    the fast path ModelExpress accelerates, matching fig-1's data-plane green.
  - The ModelExpress software components (MX Server + MX Clients) take a
    neutral, elevated structural surface (never green fills), so green stays
    the one thing that reads as the accelerated flow.
  - cpu_blue marks the control-plane metadata store; thin grey lines carry the
    recessive control wiring.
  - All connectors are orthogonal (right angles only); every coordinate is
    computed from named constants; the three data paths converge on the New
    engine's bottom edge with arrowheads on exact edges.

Title uses the Dynamo Dark display / hero treatment (Helvetica Neue Light,
title case). Layout is representative of the ModelExpress design; it carries
no measured numbers. Renders deterministically.

Usage:
    python3 gen_fig_3_modelexpress.py   # -> images/fig-3-modelexpress-coldstart.{png,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"


def _box(shapes, x0, y0, x1, y1, *, border, fill, width=1.0, layer="above"):
    """Append a rectangular card (border-radius 0 is implicit in Plotly rects).

    Plane bands use ``layer="below"`` so the control wires and data-flow lines
    (Scatter traces) render on top of the band fill; component cards use the
    default ``layer="above"`` so flow lines stop cleanly at their edges.
    """
    shapes.append(
        dict(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color=border, width=width),
            fillcolor=fill,
            layer=layer,
        )
    )


def _label(
    annotations, x, y, text, *, color, size=14, weight=400, font=None, anchor="center"
):
    annotations.append(
        dict(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(
                family=font or "Arial, Helvetica, sans-serif",
                size=size,
                color=color,
                weight=weight,
            ),
            xanchor=anchor,
            yanchor="middle",
        )
    )


def _wire(fig, xs, ys, *, color, width=1.5, dash=None):
    """A recessive control-plane connector (orthogonal polyline, no arrowhead)."""
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _flow(fig, annotations, waypoints, *, color, width=2.0):
    """A data-plane weight-transfer flow: orthogonal polyline + arrowhead at the
    tip. All tips approach the target from directly below (vertical up)."""
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    tip_x, tip_y = waypoints[-1]
    # Stop the line just short of the tip; the arrowhead annotation lands on it.
    ys[-1] = tip_y - 1.8
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(color=color, width=width),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    annotations.append(
        dict(
            x=tip_x,
            y=tip_y,
            ax=tip_x,
            ay=tip_y - 1.8,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.3,
            arrowwidth=width,
            arrowcolor=color,
            text="",
        )
    )


def main() -> None:
    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]

    green = colors["accent"]["dynamo_green"]  # data-plane weight-transfer flows
    cpu_blue = colors["accent"]["cpu_blue"]  # control-plane metadata store
    blue_fill = colors["fills"]["blue"]
    subtle = colors["border"]["subtle"]  # hairline borders
    surface = colors["background"]["surface"]  # plane band fill
    surface_alt = colors["background"]["surface_alt"]  # component card fill
    elevated = colors["background"]["elevated"]  # MX component cards (neutral)
    text_primary = colors["text"]["primary"]
    text_secondary = colors["text"]["secondary"]
    text_medium = colors["text"]["medium"]
    text_muted = colors["text"]["muted"]  # neutral border for MX component cards

    fig = go.Figure()
    shapes: list[dict] = []
    annotations: list[dict] = []

    # ── Plane bands ──────────────────────────────────────────────────────────
    CTRL = (3.0, 51.0, 97.0, 96.0)  # x0, y0, x1, y1
    DATA = (3.0, 4.0, 97.0, 46.0)
    for x0, y0, x1, y1 in (CTRL, DATA):
        _box(
            shapes,
            x0,
            y0,
            x1,
            y1,
            border=subtle,
            fill=surface,
            width=1.0,
            layer="below",
        )
    _label(
        annotations,
        CTRL[0] + 2.5,
        CTRL[3] - 3.0,
        "CONTROL PLANE",
        color=text_medium,
        size=14,
        weight=600,
        anchor="left",
    )
    _label(
        annotations,
        DATA[0] + 2.5,
        DATA[3] - 3.0,
        "DATA PLANE",
        color=text_medium,
        size=14,
        weight=600,
        anchor="left",
    )

    # ── Control-plane components ─────────────────────────────────────────────
    # Shared mid-line for MX Client / MX Server so control wires stay straight.
    MX_Y0, MX_Y1 = 61.0, 72.0
    MX_YMID = (MX_Y0 + MX_Y1) / 2

    # Inference Engine (Source) container with nested MX Client.
    SRC = (5.0, 57.0, 29.0, 79.0)
    NEW = (71.0, 57.0, 95.0, 79.0)
    for (x0, y0, x1, y1), title in (
        (SRC, "Inference Engine<br>(Source)"),
        (NEW, "Inference Engine<br>(New)"),
    ):
        _box(shapes, x0, y0, x1, y1, border=subtle, fill=surface_alt, width=1.0)
        _label(
            annotations,
            (x0 + x1) / 2,
            y1 - 3.5,
            title,
            color=text_secondary,
            size=13,
            weight=400,
        )

    SRC_CLIENT = (9.0, MX_Y0, 25.0, MX_Y1)
    NEW_CLIENT = (75.0, MX_Y0, 91.0, MX_Y1)
    for x0, y0, x1, y1 in (SRC_CLIENT, NEW_CLIENT):
        _box(shapes, x0, y0, x1, y1, border=text_muted, fill=elevated, width=1.5)
        _label(
            annotations,
            (x0 + x1) / 2,
            (y0 + y1) / 2,
            "MX Client",
            color=text_primary,
            size=13,
            weight=400,
        )

    # MX Server (center).
    SRV = (42.0, MX_Y0, 58.0, MX_Y1)
    _box(shapes, *SRV, border=text_muted, fill=elevated, width=1.5)
    _label(
        annotations,
        (SRV[0] + SRV[2]) / 2,
        MX_YMID,
        "MX Server",
        color=text_primary,
        size=13,
        weight=400,
    )

    # Metadata Store (control-plane coordination backend → cpu_blue), top-center.
    META = (43.0, 82.0, 57.0, 92.0)
    _box(shapes, *META, border=cpu_blue, fill=blue_fill, width=1.5)
    _label(
        annotations,
        (META[0] + META[2]) / 2,
        (META[1] + META[3]) / 2,
        "Metadata Store",
        color=text_primary,
        size=13,
        weight=400,
    )
    _label(
        annotations,
        (META[0] + META[2]) / 2,
        META[3] + 2.4,
        "Redis / K8s backend",
        color=text_muted,
        size=11,
        weight=400,
    )

    # ── Control-plane wiring (recessive grey, orthogonal) ────────────────────
    meta_cx = (META[0] + META[2]) / 2
    _wire(
        fig, [meta_cx, meta_cx], [META[1], SRV[3]], color=text_medium
    )  # store → server
    _wire(
        fig, [SRC_CLIENT[2], SRV[0]], [MX_YMID, MX_YMID], color=text_medium
    )  # source client → server
    _wire(
        fig, [SRV[2], NEW_CLIENT[0]], [MX_YMID, MX_YMID], color=text_medium
    )  # server → new client

    # ── Data-plane components ────────────────────────────────────────────────
    OBJ = (28.0, 10.0, 50.0, 24.0)
    FILE = (60.0, 10.0, 92.0, 24.0)
    _box(shapes, *OBJ, border=subtle, fill=surface_alt, width=1.0)
    _label(
        annotations,
        (OBJ[0] + OBJ[2]) / 2,
        (OBJ[1] + OBJ[3]) / 2,
        "Object Storage<br>(Remote)",
        color=text_secondary,
        size=13,
        weight=400,
    )
    _box(shapes, *FILE, border=subtle, fill=surface_alt, width=1.0)
    _label(
        annotations,
        (FILE[0] + FILE[2]) / 2,
        (FILE[1] + FILE[3]) / 2,
        "File Storage<br>(Local / Network)",
        color=text_secondary,
        size=13,
        weight=400,
    )

    # ── Data-plane weight-transfer flows (fluorite, converge on New engine) ──
    NEW_BOTTOM = NEW[1]  # y = 57
    TIP_RDMA, TIP_MS, TIP_GDS = 76.0, 82.0, 88.0

    # GPUDirect RDMA: Source engine GPU memory → New engine (peer-to-peer).
    y_rdma = 38.0
    src_out_x = 15.0
    _flow(
        fig,
        annotations,
        [
            (src_out_x, SRC[1]),
            (src_out_x, y_rdma),
            (TIP_RDMA, y_rdma),
            (TIP_RDMA, NEW_BOTTOM),
        ],
        color=green,
    )

    # ModelStreamer: Object storage → New engine.
    y_ms = 30.0
    obj_cx = (OBJ[0] + OBJ[2]) / 2
    _flow(
        fig,
        annotations,
        [(obj_cx, OBJ[3]), (obj_cx, y_ms), (TIP_MS, y_ms), (TIP_MS, NEW_BOTTOM)],
        color=green,
    )

    # GPUDirect Storage (GDS): File storage → New engine (straight vertical).
    _flow(fig, annotations, [(TIP_GDS, FILE[3]), (TIP_GDS, NEW_BOTTOM)], color=green)

    # Flow labels (bound to the fluorite flow color).
    _label(
        annotations,
        (src_out_x + TIP_RDMA) / 2,
        y_rdma + 2.6,
        "<b>GPUDirect RDMA</b>",
        color=green,
        size=13,
        weight=700,
    )
    _label(
        annotations,
        (obj_cx + TIP_MS) / 2,
        y_ms + 2.6,
        "<b>ModelStreamer</b>",
        color=green,
        size=13,
        weight=700,
    )
    _label(
        annotations,
        TIP_GDS - 2.0,
        41.0,
        "<b>GPUDirect Storage<br>(GDS)</b>",
        color=green,
        size=13,
        weight=700,
        anchor="right",
    )

    # Cold-start phase tag over the receiving engine.
    _label(
        annotations,
        (NEW[0] + NEW[2]) / 2,
        NEW[3] + 2.4,
        "<b>COLD START</b>",
        color=text_secondary,
        size=12,
        weight=700,
    )

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        template=template,
        title=dict(
            text="ModelExpress Splits Coordination from Weight Transfer",
            font=dict(family=HERO_FONT, size=40, color=text_primary, weight=300),
            subtitle=dict(
                text=(
                    "Cold start — the control plane tracks where weights live; the data plane "
                    "moves them over GPUDirect RDMA, ModelStreamer, or GDS."
                ),
                font=dict(family=HERO_FONT, size=19, color=text_muted, weight=300),
            ),
            x=0.025,
            xanchor="left",
            y=0.955,
            yanchor="top",
        ),
        xaxis=dict(range=[0, 100], visible=False),
        yaxis=dict(range=[0, 102], visible=False),
        width=1600,
        height=1000,
        margin=dict(l=40, r=40, t=175, b=30),
        shapes=shapes,
        annotations=annotations,
    )

    out_png = Path(__file__).parent / "images" / "fig-3-modelexpress-coldstart.png"
    out_svg = Path(__file__).parent / "images" / "fig-3-modelexpress-coldstart.svg"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_png), scale=2)
    fig.write_image(str(out_svg))
    print(f"Wrote {out_png}")
    print(f"Wrote {out_svg}")


if __name__ == "__main__":
    main()
