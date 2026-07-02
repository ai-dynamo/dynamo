#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render deterministic SVG figures from normalized PFaS result data."""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
from pathlib import Path
from typing import Any

BACKGROUND = "#111111"
PANEL = "#1b1b1b"
TEXT = "#f5f5f5"
MUTED = "#b3b3b3"
GRID = "#3a3a3a"
GREEN = "#76b900"
COLORS = (GREEN, "#00a6a6", "#fac200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("normalized_json", type=Path)
    parser.add_argument("--latency-output", required=True, type=Path)
    parser.add_argument("--throughput-output", required=True, type=Path)
    parser.add_argument("--timeline-output", required=True, type=Path)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def text_element(
    x: float,
    y: float,
    value: object,
    *,
    size: int = 14,
    color: str = TEXT,
    anchor: str = "start",
    weight: int = 400,
    rotate: int | None = None,
) -> str:
    transform = (
        "" if rotate is None else f' transform="rotate({rotate} {x:.1f} {y:.1f})"'
    )
    return (
        f'<text x="{x:.1f}" y="{y:.1f}"{transform} fill="{color}" '
        f'font-family="Arial, sans-serif" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}">{esc(value)}</text>'
    )


def svg_document(width: int, height: int, title: str, body: list[str]) -> str:
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
                f'aria-label="{esc(title)}">'
            ),
            f"<title>{esc(title)}</title>",
            f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>',
            *body,
            "</svg>",
            "",
        ]
    )


def write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def load_normalized(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    require(data.get("schema_version") == 1, "unsupported normalized schema")
    require(bool(data.get("spec_fingerprint")), "specification fingerprint is absent")
    require(bool(data.get("requests")), "normalized request data is absent")
    require(bool(data.get("repetitions")), "normalized repetition data is absent")
    return data


def group_requests(data: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for request in data["requests"]:
        repetition = int(request["repetition"])
        grouped.setdefault(repetition, []).append(request)
    return dict(sorted(grouped.items()))


def nice_maximum(value: float) -> float:
    require(value > 0, "plot maximum must be positive")
    magnitude = 10 ** math.floor(math.log10(value))
    scaled = value / magnitude
    rounded = next(
        candidate
        for candidate in (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10)
        if scaled <= candidate
    )
    return float(rounded * magnitude)


def format_axis_tick(value: float, maximum: float) -> str:
    decimals = max(0, -math.floor(math.log10(maximum / 4)))
    return f"{value:.{decimals}f}"


def render_ecdf_panel(
    grouped: dict[int, list[dict[str, Any]]],
    metric: str,
    title: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    values = [
        float(item[metric]) / 1000 for group in grouped.values() for item in group
    ]
    x_max = nice_maximum(max(values))
    left, right, top, bottom = x + 58, x + width - 18, y + 38, y + height - 48
    plot_width, plot_height = right - left, bottom - top
    body = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="8" fill="{PANEL}"/>',
        text_element(x + 18, y + 26, title, size=16, weight=600),
    ]
    for tick in range(5):
        fraction = tick / 4
        tick_x = left + fraction * plot_width
        tick_y = bottom - fraction * plot_height
        body.append(
            f'<line x1="{tick_x:.1f}" y1="{top:.1f}" x2="{tick_x:.1f}" y2="{bottom:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        body.append(
            f'<line x1="{left:.1f}" y1="{tick_y:.1f}" x2="{right:.1f}" y2="{tick_y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        body.append(
            text_element(
                tick_x,
                bottom + 20,
                format_axis_tick(fraction * x_max, x_max),
                size=11,
                color=MUTED,
                anchor="middle",
            )
        )
        body.append(
            text_element(
                left - 10,
                tick_y + 4,
                f"{fraction:.2f}",
                size=11,
                color=MUTED,
                anchor="end",
            )
        )
    body.append(
        text_element(
            (left + right) / 2,
            bottom + 40,
            "seconds",
            size=12,
            color=MUTED,
            anchor="middle",
        )
    )
    body.append(
        text_element(
            left - 42,
            (top + bottom) / 2,
            "ECDF",
            size=12,
            color=MUTED,
            anchor="middle",
            rotate=-90,
        )
    )

    for color_index, (repetition, records) in enumerate(grouped.items()):
        ordered = sorted(float(record[metric]) / 1000 for record in records)
        denominator = max(len(ordered) - 1, 1)
        points = [
            (
                left + min(value / x_max, 1.0) * plot_width,
                bottom - index / denominator * plot_height,
            )
            for index, value in enumerate(ordered)
        ]
        path = " ".join(
            ("M" if index == 0 else "L") + f" {point_x:.1f} {point_y:.1f}"
            for index, (point_x, point_y) in enumerate(points)
        )
        color = COLORS[color_index % len(COLORS)]
        body.append(
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        legend_x = right - 88 + color_index * 30
        body.append(
            f'<circle cx="{legend_x:.1f}" cy="{y + 22:.1f}" r="4" fill="{color}"/>'
        )
        body.append(
            text_element(legend_x + 7, y + 26, str(repetition), size=11, color=MUTED)
        )
    return body


def render_latency(data: dict[str, Any]) -> str:
    grouped = group_requests(data)
    configured_isl = data["locked_workload"]["configured_input_sequence_length"]
    server_input_lengths = sorted(
        {int(request["server_input_tokens"]) for request in data["requests"]}
    )
    server_isl = (
        str(server_input_lengths[0])
        if len(server_input_lengths) == 1
        else f"{server_input_lengths[0]}–{server_input_lengths[-1]}"
    )
    body = [
        text_element(
            30, 38, "Client-observed latency distributions", size=24, weight=700
        ),
        text_element(
            30,
            62,
            (
                f"{len(grouped)} repetitions · configured/server ISL "
                f"{configured_isl}/{server_isl} tokens · fingerprint "
                f"{data['spec_fingerprint'][:12]}"
            ),
            size=13,
            color=MUTED,
        ),
    ]
    body.extend(
        render_ecdf_panel(grouped, "ttft_ms", "Time to first token", 30, 84, 405, 280)
    )
    body.extend(
        render_ecdf_panel(
            grouped,
            "itl_ms",
            "Inter-token latency",
            465,
            84,
            405,
            280,
        )
    )
    body.extend(
        render_ecdf_panel(
            grouped,
            "request_latency_ms",
            "End-to-end request latency",
            30,
            386,
            840,
            260,
        )
    )
    body.append(
        text_element(
            30,
            674,
            "Each curve contains one repetition; timing is measured at the AIPerf client.",
            size=12,
            color=MUTED,
        )
    )
    return svg_document(900, 700, "Client-observed PFaS latency distributions", body)


def render_bar_panel(
    repetitions: list[dict[str, Any]],
    metric: str,
    title: str,
    unit: str,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    values = [float(item["metrics"][metric]["avg"]) for item in repetitions]
    y_max = nice_maximum(max(values) * 1.1)
    left, right, top, bottom = x + 54, x + width - 18, y + 38, y + height - 48
    plot_width, plot_height = right - left, bottom - top
    body = [
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" rx="8" fill="{PANEL}"/>',
        text_element(x + 18, y + 26, title, size=16, weight=600),
    ]
    for tick in range(5):
        fraction = tick / 4
        tick_y = bottom - fraction * plot_height
        body.append(
            f'<line x1="{left:.1f}" y1="{tick_y:.1f}" x2="{right:.1f}" y2="{tick_y:.1f}" stroke="{GRID}" stroke-width="1"/>'
        )
        body.append(
            text_element(
                left - 8,
                tick_y + 4,
                f"{fraction * y_max:.2f}",
                size=11,
                color=MUTED,
                anchor="end",
            )
        )
    slot = plot_width / len(values)
    bar_width = min(slot * 0.52, 74)
    for index, (repetition, value) in enumerate(zip(repetitions, values, strict=True)):
        bar_height = value / y_max * plot_height
        bar_x = left + index * slot + (slot - bar_width) / 2
        bar_y = bottom - bar_height
        color = COLORS[index % len(COLORS)]
        body.append(
            f'<rect x="{bar_x:.1f}" y="{bar_y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3" fill="{color}"/>'
        )
        body.append(
            text_element(
                bar_x + bar_width / 2,
                bar_y - 7,
                f"{value:.3f}",
                size=11,
                anchor="middle",
            )
        )
        body.append(
            text_element(
                bar_x + bar_width / 2,
                bottom + 20,
                f"Run {repetition['repetition']}",
                size=11,
                color=MUTED,
                anchor="middle",
            )
        )
    body.append(
        text_element(
            (left + right) / 2, bottom + 40, unit, size=12, color=MUTED, anchor="middle"
        )
    )
    return body


def render_throughput(data: dict[str, Any]) -> str:
    repetitions = data["repetitions"]
    body = [
        text_element(30, 38, "Per-repetition throughput", size=24, weight=700),
        text_element(
            30,
            62,
            f"Closed-loop concurrency 4 · 64 requests per repetition · fingerprint {data['spec_fingerprint'][:12]}",
            size=13,
            color=MUTED,
        ),
    ]
    body.extend(
        render_bar_panel(
            repetitions,
            "request_throughput",
            "Request throughput",
            "requests / second",
            30,
            84,
            405,
            300,
        )
    )
    body.extend(
        render_bar_panel(
            repetitions,
            "output_token_throughput",
            "Output-token throughput",
            "tokens / second",
            465,
            84,
            405,
            300,
        )
    )
    body.append(
        text_element(
            30,
            412,
            "Bars report canonical AIPerf averages; no cross-cluster control arm was measured.",
            size=12,
            color=MUTED,
        )
    )
    return svg_document(900, 438, "PFaS throughput by locked repetition", body)


def representative_request(data: dict[str, Any]) -> dict[str, Any]:
    requests = data["requests"]
    median_latency = statistics.median(
        float(item["request_latency_ms"]) for item in requests
    )
    request = min(
        requests,
        key=lambda item: (
            abs(float(item["request_latency_ms"]) - median_latency),
            str(item["request_id"]),
        ),
    )
    require(
        float(request["request_latency_ms"]) >= float(request["ttft_ms"]),
        "representative request ends before its first token",
    )
    return request


def render_timeline(data: dict[str, Any]) -> str:
    request = representative_request(data)
    ttft = float(request["ttft_ms"]) / 1000
    total = float(request["request_latency_ms"]) / 1000
    streaming = total - ttft
    left, right, bar_y, bar_height = 70.0, 830.0, 118.0, 56.0
    scale = (right - left) / total
    split = left + ttft * scale
    body = [
        text_element(
            30,
            38,
            "Representative client-visible request timeline",
            size=24,
            weight=700,
        ),
        text_element(
            30,
            62,
            f"Run {request['repetition']} · nearest median request · fingerprint {data['spec_fingerprint'][:12]}",
            size=13,
            color=MUTED,
        ),
        f'<rect x="{left:.1f}" y="{bar_y:.1f}" width="{ttft * scale:.1f}" height="{bar_height:.1f}" rx="4" fill="{GREEN}"/>',
        f'<rect x="{split:.1f}" y="{bar_y:.1f}" width="{streaming * scale:.1f}" height="{bar_height:.1f}" rx="4" fill="#00a6a6"/>',
        text_element(
            left + ttft * scale / 2,
            bar_y + 34,
            f"TTFT {ttft:.2f}s",
            size=15,
            anchor="middle",
            weight=600,
        ),
        text_element(
            split + streaming * scale / 2,
            bar_y + 34,
            f"First-to-last token {streaming:.2f}s",
            size=15,
            anchor="middle",
            weight=600,
        ),
        text_element(
            left, bar_y + 82, "request start", size=12, color=MUTED, anchor="middle"
        ),
        text_element(
            split, bar_y + 82, "first token", size=12, color=MUTED, anchor="middle"
        ),
        text_element(
            right, bar_y + 82, "request complete", size=12, color=MUTED, anchor="middle"
        ),
        text_element(
            30,
            242,
            "These client intervals are not exact prefill, NIXL-transfer, network, or decode timings.",
            size=12,
            color=MUTED,
        ),
    ]
    return svg_document(
        900, 266, "Representative client-visible PFaS request timeline", body
    )


def render_all(
    data: dict[str, Any], latency: Path, throughput: Path, timeline: Path
) -> None:
    write_atomic(latency, render_latency(data))
    write_atomic(throughput, render_throughput(data))
    write_atomic(timeline, render_timeline(data))


def main() -> int:
    args = parse_args()
    data = load_normalized(args.normalized_json)
    render_all(data, args.latency_output, args.throughput_output, args.timeline_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
