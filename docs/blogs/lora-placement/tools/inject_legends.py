#!/usr/bin/env python3
#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
"""Inject legends and apply padding to LoRA Placement SVGs.

Reads each diagram SVG, adjusts the viewBox for padding, and injects
a legend group at the specified coordinates. Fully idempotent: uses
marker comments to strip previous injections before re-applying.

Usage:
    python3 inject_legends.py
"""

import re
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
OUT = HERE.parent / "images"

ROW_H = 22
LABEL_W = 80
LINE_W = 40
ARROW_W = 6
PAD = 8

# Marker comments for idempotent injection
BG_START = "<!-- INJ:BG -->"
BG_END = "<!-- /INJ:BG -->"
LEG_START = "<!-- INJ:LEGEND -->"
LEG_END = "<!-- /INJ:LEGEND -->"
ORIG_VB_ATTR = "data-orig-vb"

CONFIG: dict[str, dict[str, Any]] = {
    "architecture-overview": {
        "file": "architecture-overview.svg",
        "out": "fig-1-architecture-overview.svg",
        "legend": {"x": 823, "y": 50},
        "padding": {"top": -50, "right": -75, "bottom": -50, "left": -76},
        "entries": [
            ("Data Flow", "#c4a035", "solid"),
            ("Control Flow", "#5a90c0", "dashed"),
            ("Events", "#c08050", "dashed"),
        ],
    },
    "control-loop": {
        "file": "control-loop.svg",
        "out": "fig-2-control-loop.svg",
        "legend": {"x": 588, "y": 20},
        "padding": {"top": -30, "right": -30, "bottom": -30, "left": -30},
        "entries": [
            ("Per-Request", "#5a90c0", "solid"),
            ("Periodic", "#76b900", "dashed"),
            ("Feedback", "#c08050", "dashed"),
        ],
    },
    "mcf-bipartite": {
        "file": "mcf-bipartite.svg",
        "out": "fig-3-mcf-bipartite.svg",
        "legend": {"x": 588, "y": 20},
        "padding": {"top": -30, "right": -30, "bottom": -30, "left": -30},
        "entries": [
            ("Flow Edge", "#c4a035", "solid"),
            ("Overflow", "#d06060", "dashed"),
        ],
    },
}


def build_legend_group(entries: list[tuple[str, str, str]], x: int, y: int) -> str:
    """Build SVG elements for a legend at the given position."""
    parts = [
        f'<g transform="translate({x},{y})">',
    ]

    for i, (label, color, style) in enumerate(entries):
        ey = PAD + i * ROW_H + 14
        parts.append(
            f'<text x="{PAD}" y="{ey}" fill="{color}" font-size="13" '
            f'font-family="Helvetica, Arial, sans-serif">{label}</text>'
        )
        lx = PAD + LABEL_W
        dash = ""
        if style == "dashed":
            dash = ' stroke-dasharray="6,4"'
        elif style == "dotted":
            dash = ' stroke-dasharray="3,4"'
        parts.append(
            f'<line x1="{lx}" y1="{ey - 4}" x2="{lx + LINE_W}" y2="{ey - 4}" '
            f'stroke="{color}" stroke-width="2"{dash}/>'
        )
        ax = lx + LINE_W + 4
        ay = ey - 4
        parts.append(
            f'<polygon points="{ax},{ay - 4} {ax},{ay + 4} {ax + ARROW_W},{ay}" fill="{color}"/>'
        )

    parts.append("</g>")
    return "".join(parts)


def strip_injections(svg: str) -> str:
    """Remove all previously injected content and restore original viewBox."""
    svg = re.sub(
        re.escape(BG_START) + r".*?" + re.escape(BG_END),
        "",
        svg,
        flags=re.DOTALL,
    )
    svg = re.sub(
        re.escape(LEG_START) + r".*?" + re.escape(LEG_END),
        "",
        svg,
        flags=re.DOTALL,
    )

    m = re.search(rf'{ORIG_VB_ATTR}="([^"]*)"', svg)
    if m:
        orig_vb = m.group(1)
        svg = re.sub(rf'\s*{ORIG_VB_ATTR}="[^"]*"', "", svg)
        svg = re.sub(r'viewBox="[^"]*"', f'viewBox="{orig_vb}"', svg, count=1)

    return svg


def inject(svg: str, cfg: dict) -> str:
    """Inject legend and padding into an SVG string."""
    svg = strip_injections(svg)

    vb_m = re.search(r'viewBox="([^"]*)"', svg)
    if not vb_m:
        raise ValueError("No viewBox found in SVG")
    orig_vb = vb_m.group(1)
    vbx, vby, vbw, vbh = (float(v) for v in orig_vb.split())

    p = cfg["padding"]
    nx = vbx - p["left"]
    ny = vby - p["top"]
    nw = vbw + p["left"] + p["right"]
    nh = vbh + p["top"] + p["bottom"]
    new_vb = f"{nx:g} {ny:g} {nw:g} {nh:g}"

    svg = re.sub(
        r'viewBox="[^"]*"',
        f'viewBox="{new_vb}" {ORIG_VB_ATTR}="{orig_vb}"',
        svg,
        count=1,
    )

    bg_rect = (
        f'{BG_START}<rect x="{nx:g}" y="{ny:g}" width="{nw:g}" height="{nh:g}" '
        f'fill="#0a0a0a"/>{BG_END}'
    )

    legend = build_legend_group(cfg["entries"], cfg["legend"]["x"], cfg["legend"]["y"])
    legend_block = f"{LEG_START}{legend}{LEG_END}"

    nested = "</svg></svg>" in svg
    if nested:
        inner_pos = svg.index("<svg", svg.index("<svg") + 1)
        svg = svg[:inner_pos] + bg_rect + svg[inner_pos:]
        outer_close = svg.rfind("</svg></svg>") + len("</svg>")
        svg = svg[:outer_close] + legend_block + svg[outer_close:]
    else:
        open_end = svg.index(">", svg.index("<svg")) + 1
        svg = svg[:open_end] + bg_rect + svg[open_end:]
        close_pos = svg.rfind("</svg>")
        svg = svg[:close_pos] + legend_block + svg[close_pos:]

    return svg


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for cfg in CONFIG.values():
        src = HERE / cfg["file"]
        if not src.exists():
            print(f"  SKIP: {cfg['file']} not found")
            continue

        svg = src.read_text()
        result = inject(svg, cfg)

        dst = OUT / cfg["out"]
        dst.write_text(result)

        p = cfg["padding"]
        print(
            f"  {cfg['file']} -> {cfg['out']}: legend at ({cfg['legend']['x']}, {cfg['legend']['y']}), "
            f"padding T{p['top']:+d} R{p['right']:+d} B{p['bottom']:+d} L{p['left']:+d}"
        )


if __name__ == "__main__":
    main()
