#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Hero: ModelExpress "The Fast Path" flow diagram (Dynamo Dark).

The canonical wide (16:9) hero for the ModelExpress / cold-start figure set. It
blends two explored concepts: the FLOW DIAGRAM of concept B is the base, and the
EDITORIAL STAT TREATMENT of concept C supplies the annotations.

Composition (from concept B): model weights stream peer-to-peer over a single
bold, glowing green RDMA lane straight from a Source GPU to a New GPU (the fast
path, the visual spine). A long, dim, dashed coral detour climbs from a cold
object store up into the New GPU — the slow baseline it bypasses. A small
cpu_blue Metadata Store sits on recessive dashed control-plane wires; it only
coordinates where weights live, it never moves them.

Annotations (from concept C): the two paths are labeled explicitly and bound to
their lanes — "Model load via MX P2P - 11s" on the green spine, "Cold
object-store pull - 70s" on the coral detour — and the win is stated as art: a
giant light-weight green 6.4x with a grounded "59 s reclaimed per cold start"
secondary line.

The one clear "aha": the green data-plane lane is short, straight, and bright;
the coral baseline detour is long, dim, and dashed — and the 6.4x names the gap.

Treatment follows the canonical Dynamo Dark tokens (design_tokens.yaml consumed
via plotly_dynamo.py): pure-black ground, token palette only, no rounded
corners. Uses the Dynamo Dark DISPLAY title treatment - Helvetica Neue Light,
title case - paired with a muted Helvetica subtitle. The giant numeral uses the
same light display face; the number is the display type.

Color convention (shared, set-wide):
    green   -> the fast RDMA data-plane weight-transfer path / the win ONLY. The
               single dominant accent: the spine, its 11 s label, and the 6.4x
               stat. Boxes are never green - green is the flow.
    coral   -> the slow baseline / cold object-store pull (detour + its store).
    cpu_blue-> control-plane (the Metadata Store / coordination node).
    (gold   -> warmup; reserved set-wide, not needed for this path-only hero.)

Numbers are the source of truth from the ModelExpress cold-start figure set
(fig-4 / fig-6): baseline object-store pull 70 s, ModelExpress P2P RDMA 11 s; the
6.4x and 59 s are derived (70 / 11, 70 - 11). Scoped to the weight-transfer
phase only - cold JIT-cache warmup is a separate cold-start cost.

Usage:
    python3 gen_hero.py            # -> ../images/hero-modelexpress-coldstart.{png,svg}
    python3 gen_hero.py -o out.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).parent))
from plotly_dynamo import build_template, load_tokens

# Hero / display title set: Helvetica Neue Light, title case (falls back to the
# token sans stack where Helvetica Neue is unavailable). The giant numeral uses
# this same light-weight display face - the number is the display type.
HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# ---- data (source of truth: ModelExpress cold-start figure set) --------------
BASELINE_SEC = 70  # baseline: cold pull from object store (coral, slow path)
MX_SEC = 11  # ModelExpress: P2P RDMA weight transfer (green, fast path)
RECLAIMED_SEC = BASELINE_SEC - MX_SEC  # 59 s reclaimed on the weight-load phase
SPEEDUP = round(BASELINE_SEC / MX_SEC, 1)  # 6.4x faster weight load

# ---- geometry (x, y both in [0, 100] data space) -----------------------------
# GPU actors sit high-center; the green spine runs straight between them.
GPU_Y0, GPU_Y1 = 56.0, 78.0
GPU_YMID = (GPU_Y0 + GPU_Y1) / 2  # 67.0
SRC = (7.0, GPU_Y0, 29.0, GPU_Y1)  # Source GPU (weights resident)
NEW = (71.0, GPU_Y0, 93.0, GPU_Y1)  # New GPU (cold start)
SRC_CX = (SRC[0] + SRC[2]) / 2  # 18.0
NEW_CX = (NEW[0] + NEW[2]) / 2  # 82.0

# Green fast path: Source right edge -> New left edge, along the GPU mid-line.
SPINE_Y = GPU_YMID
SPINE_X0 = SRC[2]  # 29.0  (source right edge)
SPINE_X1 = NEW[0]  # 71.0  (new left edge)
SPINE_CX = (SPINE_X0 + SPINE_X1) / 2  # 50.0

# Control-plane metadata node, small + quiet, top-center.
META = (43.0, 85.0, 57.0, 94.0)
META_CX = (META[0] + META[2]) / 2  # 50.0
META_BOT = META[1]  # 85.0
CTRL_BUS_Y = 81.5  # recessive coordination bus, above the GPUs

# Cold object store, bottom-center; the slow baseline source.
OBJ = (39.0, 8.0, 61.0, 24.0)
OBJ_CX = (OBJ[0] + OBJ[2]) / 2  # 50.0
OBJ_TOP = OBJ[3]  # 24.0

# Coral baseline route: long, indirect climb from the store up into the New GPU.
CORAL_MID_Y = 40.0
NEW_BOT = NEW[1]  # 56.0

# Editorial win-stat block (concept C), anchored in the open lower-left.
STAT_X = 6.0  # shared left anchor for the numeral + its captions
STAT_NUM_Y = 34.0  # vertical center of the giant 6.4x numeral
STAT_SUB_Y = 22.5  # "Faster Model Load" descriptor, under the numeral
STAT_RECLAIM_Y = 15.5  # grounded secondary stat, quiet


def _box(shapes, box, *, border, fill, width=1.0, layer="above"):
    x0, y0, x1, y1 = box
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
    annotations, x, y, text, *, color, size=14, weight=400, font, anchor="center"
):
    annotations.append(
        dict(
            x=x,
            y=y,
            text=text,
            showarrow=False,
            font=dict(family=font, size=size, color=color, weight=weight),
            xanchor=anchor,
            yanchor="middle",
        )
    )


def _polyline(fig, waypoints, *, color, width, dash=None, opacity=1.0):
    fig.add_trace(
        go.Scatter(
            x=[p[0] for p in waypoints],
            y=[p[1] for p in waypoints],
            mode="lines",
            line=dict(color=color, width=width, dash=dash),
            opacity=opacity,
            showlegend=False,
            hoverinfo="skip",
        )
    )


def _arrowhead(annotations, tail, tip, *, color, width, size=1.6):
    annotations.append(
        dict(
            x=tip[0],
            y=tip[1],
            ax=tail[0],
            ay=tail[1],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=size,
            arrowwidth=width,
            arrowcolor=color,
            text="",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = str(
        Path(__file__).resolve().parent.parent
        / "images"
        / "hero-modelexpress-coldstart.png"
    )
    parser.add_argument("-o", "--output", default=default_out, help="Output PNG path")
    parser.add_argument("--width", type=int, default=1600, help="Canvas width (px)")
    parser.add_argument("--height", type=int, default=900, help="Canvas height (px)")
    args = parser.parse_args()

    tokens = load_tokens(Path(__file__).parent / "design_tokens.yaml")
    template = build_template(tokens)
    colors = tokens["colors"]
    font_sans = tokens["typography"]["font_family"]
    font_mono = tokens["typography"]["font_family_mono"]

    green = colors["accent"]["dynamo_green"]  # fast RDMA data-plane path (accent)
    coral = colors["accent"]["coral"]  # slow baseline / object-store pull
    coral_dim = colors["chart_fills"][7]  # muted coral for the receding route
    cpu_blue = colors["accent"]["cpu_blue"]  # control-plane metadata node
    blue_fill = colors["fills"]["blue"]
    red_fill = colors["fills"]["red"]
    subtle = colors["border"]["subtle"]
    surface_alt = colors["background"]["surface_alt"]  # neutral GPU boxes
    text_primary = colors["text"]["primary"]
    text_secondary = colors["text"]["secondary"]
    text_medium = colors["text"]["medium"]
    text_muted = colors["text"]["muted"]

    fig = go.Figure()
    shapes: list[dict] = []
    annotations: list[dict] = []

    # -- Control-plane wiring (recessive dashed grey, drawn first / underneath) -
    for gpu_cx in (SRC_CX, NEW_CX):
        _polyline(
            fig,
            [
                (META_CX, META_BOT),
                (META_CX, CTRL_BUS_Y),
                (gpu_cx, CTRL_BUS_Y),
                (gpu_cx, GPU_Y1),
            ],
            color=text_medium,
            width=1.0,
            dash="4 4",
        )

    # -- Slow baseline path (coral, long + indirect + dim, dashed) -------------
    coral_route = [
        (OBJ_CX, OBJ_TOP),
        (OBJ_CX, CORAL_MID_Y),
        (NEW_CX, CORAL_MID_Y),
        (NEW_CX, NEW_BOT - 1.6),
    ]
    _polyline(fig, coral_route, color=coral_dim, width=2.0, dash="6 5", opacity=0.9)
    _arrowhead(
        annotations,
        (NEW_CX, NEW_BOT - 1.6),
        (NEW_CX, NEW_BOT),
        color=coral_dim,
        width=2.0,
        size=1.4,
    )

    # -- Fast path (green): the spine - soft glow + bold core + arrowhead ------
    _polyline(
        fig,
        [(SPINE_X0, SPINE_Y), (SPINE_X1, SPINE_Y)],
        color=green,
        width=20.0,
        opacity=0.16,  # the one permitted gradient/glow, on the single accent
    )
    _polyline(
        fig,
        [(SPINE_X0, SPINE_Y), (SPINE_X1 - 2.4, SPINE_Y)],
        color=green,
        width=6.0,
    )
    _arrowhead(
        annotations,
        (SPINE_X1 - 2.4, SPINE_Y),
        (SPINE_X1, SPINE_Y),
        color=green,
        width=6.0,
        size=1.7,
    )

    # -- GPU actors (neutral surfaces - never green) ---------------------------
    for box, cx, title, sub in (
        (SRC, SRC_CX, "Source GPU", "weights resident"),
        (NEW, NEW_CX, "New GPU", "cold start"),
    ):
        _box(shapes, box, border=subtle, fill=surface_alt, width=1.0)
        _label(
            annotations,
            cx,
            GPU_YMID + 3.2,
            title,
            color=text_primary,
            size=17,
            weight=400,
            font=font_sans,
        )
        _label(
            annotations,
            cx,
            GPU_YMID - 3.4,
            sub,
            color=text_muted,
            size=12,
            weight=400,
            font=font_sans,
        )

    # -- Control-plane metadata node (cpu_blue) --------------------------------
    # Coordination only: it tracks where weights live; it never moves them. The
    # "coordination" sub-caption names that role; the dashed grey wires carry it.
    _box(shapes, META, border=cpu_blue, fill=blue_fill, width=1.5)
    _label(
        annotations,
        META_CX,
        (META[1] + META[3]) / 2 + 1.9,
        "Metadata Store",
        color=text_primary,
        size=13,
        weight=400,
        font=font_sans,
    )
    _label(
        annotations,
        META_CX,
        (META[1] + META[3]) / 2 - 2.1,
        "coordination",
        color=text_secondary,
        size=10.5,
        weight=400,
        font=font_sans,
    )

    # -- Cold object store (slow baseline source) ------------------------------
    _box(shapes, OBJ, border=coral, fill=red_fill, width=1.5)
    _label(
        annotations,
        OBJ_CX,
        (OBJ[1] + OBJ[3]) / 2 + 1.8,
        "Cold Object Store",
        color=text_secondary,
        size=14,
        weight=400,
        font=font_sans,
    )
    _label(
        annotations,
        OBJ_CX,
        (OBJ[1] + OBJ[3]) / 2 - 3.0,
        "remote pull, cold",
        color=text_muted,
        size=11,
        weight=400,
        font=font_sans,
    )

    # -- Path labels (concept C: explicit, bound to each lane) -----------------
    # Green fast path: the win, bold, above the spine. Green on black is AA-safe
    # (8.8:1) at any size, so the 11 s lives inside the green label.
    _label(
        annotations,
        SPINE_CX,
        SPINE_Y + 5.6,
        "Model load via MX P2P \u2014 11s",
        color=green,
        size=17,
        weight=600,
        font=font_sans,
    )
    _label(
        annotations,
        SPINE_CX,
        SPINE_Y - 5.4,
        "RDMA peer-to-peer",
        color=text_medium,
        size=12,
        weight=400,
        font=font_sans,
    )
    # Coral baseline: recessive. The coral role is carried by the dashed detour
    # line it rides + the coral-bordered store below it; the label text itself is
    # a light token (AA-safe on black), keeping the baseline quiet under the win.
    _label(
        annotations,
        (OBJ_CX + NEW_CX) / 2,
        CORAL_MID_Y + 2.7,
        "Cold object-store pull \u2014 70s",
        color=text_secondary,
        size=14,
        weight=400,
        font=font_sans,
    )
    _label(
        annotations,
        (OBJ_CX + NEW_CX) / 2,
        CORAL_MID_Y - 2.6,
        "baseline, bypassed",
        color=text_medium,
        size=11,
        weight=400,
        font=font_sans,
    )

    # -- The win as art (concept C): giant light-weight green numeral ----------
    # Anchored in the open lower-left. Green = the win; grounded 59 s beneath it.
    _label(
        annotations,
        STAT_X,
        STAT_NUM_Y,
        f"{SPEEDUP}\u00d7",  # 6.4x
        color=green,
        size=100,
        weight=300,
        font=HERO_FONT,
        anchor="left",
    )
    _label(
        annotations,
        STAT_X + 0.5,
        STAT_SUB_Y,
        "Faster Model Load",
        color=text_primary,
        size=22,
        weight=300,
        font=HERO_FONT,
        anchor="left",
    )
    _label(
        annotations,
        STAT_X + 0.7,
        STAT_RECLAIM_Y,
        f"{RECLAIMED_SEC} s reclaimed per cold start",
        color=text_muted,
        size=15,
        weight=400,
        font=font_mono,
        anchor="left",
    )

    # -- Footnote (scopes the claim; grounds the config) -----------------------
    _label(
        annotations,
        3.0,
        2.6,
        (
            "Model weights only \u00b7 DeepSeek-V4-Pro / vLLM on one 8-GPU node "
            f"\u2014 {BASELINE_SEC} s cold pull vs {MX_SEC} s RDMA transfer; "
            "JIT-cache warmup is a separate cold-start cost."
        ),
        color=text_muted,
        size=13,
        weight=400,
        font=font_mono,
        anchor="left",
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="The Fast Path to Warm GPUs",
            font=dict(family=HERO_FONT, size=44, color=text_primary, weight=300),
            subtitle=dict(
                text=(
                    "ModelExpress streams model weights GPU-to-GPU over RDMA "
                    "\u2014 bypassing the cold object-store pull."
                ),
                font=dict(family=HERO_FONT, size=21, color=text_muted, weight=300),
            ),
            x=0.03,
            xanchor="left",
            y=0.955,
            yanchor="top",
        ),
        xaxis=dict(range=[0, 100], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 100], visible=False, fixedrange=True),
        width=args.width,
        height=args.height,
        margin=dict(l=40, r=40, t=200, b=48),
        shapes=shapes,
        annotations=annotations,
        showlegend=False,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out), scale=2)
    svg = out.with_suffix(".svg")
    fig.write_image(str(svg))
    print(f"Wrote {out}  ({args.width}x{args.height} @2x)")
    print(f"Wrote {svg}")


if __name__ == "__main__":
    main()
