#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Hero: ModelExpress collapsing model cold start (Dynamo Dark).

A wide (16:9) headline hero for the ModelExpress / cold-start figure set. A
clean title lockup over a single supporting motif: the model-weight-transfer
collapse. ModelExpress delivers weights peer-to-peer over RDMA, so the
weight-load phase drops from a 70 s object-store pull (coral, the slow
baseline) to an 11 s RDMA transfer (the single green accent / fast path) — a
6.4x faster weight load. A dashed green "finish line" at 11 s and a green delta
bracket over the reclaimed span carry the collapse; the baseline coral bar
blows straight past the finish line.

Treatment follows the canonical Dynamo Dark tokens (design_tokens.yaml,
consumed via plotly_dynamo.py): pure-black ground, token palette only, no
rounded corners. The hero uses the Dynamo Dark DISPLAY title treatment —
Helvetica Neue Light, title case — paired with a muted Helvetica subtitle,
distinct from the compact 18 px uppercase chart title used for dense
dashboards.

Color convention (shared with the harmonized figure set):
    green -> the fast RDMA data-plane weight-transfer path (what ModelExpress
             accelerates); the single accent.
    coral -> the slow baseline / object-store pull.

Numbers are the source of truth from the ModelExpress cold-start figure set
(fig-4 / fig-6): baseline model load 70 s, ModelExpress P2P model load 11 s;
the 6.4x and 59 s are derived (70 / 11, 70 - 11). Scoped to the weight-transfer
phase only — cold JIT-cache warmup is a separate cold-start cost.

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
# token sans stack where Helvetica Neue is unavailable).
HERO_FONT = "Helvetica Neue, Helvetica, Arial, sans-serif"

# ---- data (source of truth: ModelExpress cold-start figure set) --------------
BASELINE_SEC = 70  # baseline model load: cold pull from object store (no P2P)
MX_SEC = 11  # ModelExpress model load: P2P RDMA weight transfer
RECLAIMED_SEC = BASELINE_SEC - MX_SEC  # 59 s reclaimed on the weight-load phase
SPEEDUP = round(BASELINE_SEC / MX_SEC, 1)  # 6.4x faster weight load
X_MAX = 82  # seconds; leaves room for the right-end duration labels

# ---- geometry (x = seconds [0, X_MAX]; y = plot-area paper [0, 1]) -----------
BASE_Y0, BASE_Y1 = 0.575, 0.720  # baseline (coral) bar
MX_Y0, MX_Y1 = 0.300, 0.445  # ModelExpress (green) bar
BRACKET_Y = 0.508  # green delta bracket, in the gap between the bars


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

    green = colors["accent"]["dynamo_green"]  # ModelExpress fast path (accent)
    coral = colors["accent"]["coral"]  # slow baseline / object-store pull
    subtle = colors["border"]["subtle"]  # hairline separators
    text_primary = colors["text"]["primary"]
    text_secondary = colors["text"]["secondary"]
    text_muted = colors["text"]["muted"]

    fig = go.Figure()
    shapes: list[dict] = []
    annotations: list[dict] = []

    # ---- header divider (hairline under the title block) ---------------------
    shapes.append(
        dict(
            type="line",
            xref="paper",
            yref="paper",
            x0=0.0,
            x1=1.0,
            y0=1.02,
            y1=1.02,
            line=dict(color=subtle, width=1),
            layer="above",
        )
    )

    # ---- section heading (scopes the motif to weight transfer) ---------------
    annotations.append(
        dict(
            xref="x",
            yref="paper",
            x=0,
            y=0.895,
            xanchor="left",
            yanchor="bottom",
            text="WEIGHT TRANSFER — MODEL LOAD",
            showarrow=False,
            font=dict(family=font_sans, size=15, color=text_secondary, weight=600),
        )
    )

    # ---- the two weight-load bars (fills, no borders) ------------------------
    bars = [
        dict(sec=BASELINE_SEC, y0=BASE_Y0, y1=BASE_Y1, color=coral),
        dict(sec=MX_SEC, y0=MX_Y0, y1=MX_Y1, color=green),
    ]
    for b in bars:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=0,
                x1=b["sec"],
                y0=b["y0"],
                y1=b["y1"],
                fillcolor=b["color"],
                line=dict(width=0),
                layer="above",
            )
        )

    # ---- row labels (above each bar, left-aligned to the origin) -------------
    annotations.append(
        dict(
            xref="x",
            yref="paper",
            x=0,
            y=BASE_Y1 + 0.028,
            xanchor="left",
            yanchor="bottom",
            text="Baseline — weights pulled cold from object store",
            showarrow=False,
            font=dict(family=font_sans, size=18, color=text_secondary),
        )
    )
    annotations.append(
        dict(
            xref="x",
            yref="paper",
            x=0,
            y=MX_Y1 + 0.028,
            xanchor="left",
            yanchor="bottom",
            text="ModelExpress — RDMA peer-to-peer weight transfer",
            showarrow=False,
            font=dict(family=font_sans, size=18, color=text_secondary),
        )
    )

    # ---- right-end duration numbers (mono; role color) -----------------------
    annotations.append(
        dict(
            xref="x",
            yref="paper",
            x=BASELINE_SEC,
            y=(BASE_Y0 + BASE_Y1) / 2,
            xshift=14,
            xanchor="left",
            yanchor="middle",
            text="70s",
            showarrow=False,
            font=dict(family=font_mono, size=30, color=coral, weight=500),
        )
    )
    annotations.append(
        dict(
            xref="x",
            yref="paper",
            x=MX_SEC,
            y=(MX_Y0 + MX_Y1) / 2,
            xshift=14,
            xanchor="left",
            yanchor="middle",
            text="11s",
            showarrow=False,
            font=dict(family=font_mono, size=30, color=green, weight=500),
        )
    )

    # ---- green "finish line": where ModelExpress completes -------------------
    # A dashed green vertical guide at x = MX_SEC spanning both rows. The coral
    # baseline bar blows straight past it -> the collapse reads at a glance.
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=MX_SEC,
            x1=MX_SEC,
            y0=MX_Y0 - 0.03,
            y1=BASE_Y1 + 0.03,
            line=dict(color=green, width=1.5, dash="4 4"),
            layer="above",
        )
    )

    # ---- green delta bracket over the reclaimed span [MX_SEC, BASELINE_SEC] --
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=MX_SEC,
            x1=BASELINE_SEC,
            y0=BRACKET_Y,
            y1=BRACKET_Y,
            line=dict(color=green, width=2),
            layer="above",
        )
    )
    # right down-tick, tying the bracket to the baseline bar's near edge
    shapes.append(
        dict(
            type="line",
            xref="x",
            yref="paper",
            x0=BASELINE_SEC,
            x1=BASELINE_SEC,
            y0=BRACKET_Y,
            y1=BASE_Y0,
            line=dict(color=green, width=2),
            layer="above",
        )
    )
    # the punch line, centered above the bracket (scoped to weight load)
    annotations.append(
        dict(
            xref="x",
            yref="paper",
            x=(MX_SEC + BASELINE_SEC) / 2,
            y=BRACKET_Y + 0.012,
            xanchor="center",
            yanchor="bottom",
            text=f"{SPEEDUP}x faster weight load",
            showarrow=False,
            font=dict(family=font_sans, size=20, color=green, weight=500),
        )
    )
    annotations.append(
        dict(
            xref="x",
            yref="paper",
            x=(MX_SEC + BASELINE_SEC) / 2,
            y=BRACKET_Y - 0.070,
            xanchor="center",
            yanchor="top",
            text=f"{RECLAIMED_SEC} s reclaimed per cold start",
            showarrow=False,
            font=dict(family=font_mono, size=15, color=text_muted),
        )
    )

    # ---- footnote (scopes the claim; grounds the config) --------------------
    annotations.append(
        dict(
            xref="x",
            yref="paper",
            x=0,
            y=0.055,
            xanchor="left",
            yanchor="middle",
            text=(
                "Model weights only · DeepSeek-V4-Pro / vLLM on one 8-GPU node "
                "— cold JIT-cache warmup is a separate cold-start cost."
            ),
            showarrow=False,
            font=dict(family=font_mono, size=14, color=text_muted),
        )
    )

    fig.update_layout(
        template=template,
        title=dict(
            text="ModelExpress: Collapsing Model Cold Start",
            font=dict(family=HERO_FONT, size=44, color=text_primary, weight=300),
            subtitle=dict(
                text=(
                    "RDMA peer-to-peer weight transfer — model weights arrive "
                    "in seconds, not minutes."
                ),
                font=dict(family=HERO_FONT, size=22, color=text_muted, weight=300),
            ),
            x=0.03,
            xanchor="left",
            y=0.955,
            yanchor="top",
        ),
        xaxis=dict(range=[0, X_MAX], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        width=args.width,
        height=args.height,
        margin=dict(l=56, r=56, t=210, b=48),
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
