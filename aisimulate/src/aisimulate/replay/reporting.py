# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo-neutral reporting helpers for single-run replay."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

TITLE = "NVIDIA AIPerf | LLM Metrics"
STAT_COLUMNS = ("avg", "min", "max", "p99", "p90", "p75", "std")


def default_report_path(prefix: str = "aisimulate_replay_report") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"{prefix}_{timestamp}.json"


def write_report_json(
    report: dict[str, Any],
    output_path: str | Path | None,
    *,
    default_prefix: str = "aisimulate_replay_report",
) -> Path:
    path = (
        Path(output_path)
        if output_path is not None
        else default_report_path(default_prefix)
    )
    if path.exists() and path.is_dir():
        path = path / default_report_path(default_prefix).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def write_per_request_jsonl(
    output_path: str | Path,
    records: list[dict[str, Any]] | None,
) -> None:
    if records is None:
        raise ValueError("replay report did not provide per_request records")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            output.write("\n")


def format_report_table(report: dict[str, Any]) -> str:
    rows: list[list[str]] = []
    for label, suffix in (
        ("Time to First Token (ms)", "ttft_ms"),
        ("Time to Second Token (ms)", "ttst_ms"),
        ("Request Latency (ms)", "e2e_latency_ms"),
        ("Inter Token Latency (ms)", "itl_ms"),
        (
            "Output Token Throughput Per User (tokens/sec/user)",
            "output_token_throughput_per_user",
        ),
    ):
        _append_stat_row(rows, report, label, suffix)
    rows.extend(
        [
            [
                "Output Token Throughput (tokens/sec)",
                _format_value(report.get("output_throughput_tok_s")),
                *["N/A"] * (len(STAT_COLUMNS) - 1),
            ],
            [
                "Request Throughput (requests/sec)",
                _format_value(report.get("request_throughput_rps")),
                *["N/A"] * (len(STAT_COLUMNS) - 1),
            ],
            [
                "Request Count (requests)",
                _format_value(
                    report.get("completed_requests", report.get("num_requests"))
                ),
                *["N/A"] * (len(STAT_COLUMNS) - 1),
            ],
        ]
    )
    lines = [TITLE, _render_table(rows)]
    wall_time_ms = report.get("wall_time_ms")
    if isinstance(wall_time_ms, int | float):
        lines.append(f"Wall Time (ms): {_format_value(wall_time_ms)}")
    prefix_ratio = report.get("prefix_cache_reused_ratio")
    if isinstance(prefix_ratio, int | float):
        lines.append(f"Prefix Cache Reused Ratio: {_format_value(prefix_ratio)}")
    first_admission_ratio = report.get("first_admission_prefix_cache_reused_ratio")
    if isinstance(first_admission_ratio, int | float):
        lines.append(
            "First Admission Prefix Cache Reused Ratio: "
            f"{_format_value(first_admission_ratio)}"
        )
    return "\n".join(lines)


def _append_stat_row(
    rows: list[list[str]],
    report: dict[str, Any],
    label: str,
    suffix: str,
) -> None:
    if f"mean_{suffix}" not in report:
        return
    rows.append(
        [
            label,
            *[
                _format_value(report.get(f"{prefix}_{suffix}"))
                for prefix in ("mean", "min", "max", "p99", "p90", "p75", "std")
            ],
        ]
    )


def _render_table(rows: list[list[str]]) -> str:
    headers = ["Metric", *STAT_COLUMNS]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def render_separator(left: str, mid: str, right: str) -> str:
        return left + mid.join("━" * (width + 2) for width in widths) + right

    def render_row(row: list[str]) -> str:
        padded = []
        for index, value in enumerate(row):
            if index == 0:
                padded.append(f" {value.ljust(widths[index])} ")
            else:
                padded.append(f" {value.rjust(widths[index])} ")
        return "┃" + "┃".join(padded) + "┃"

    lines = [
        render_separator("┏", "┳", "┓"),
        render_row(headers),
        render_separator("┡", "╇", "┩"),
    ]
    lines.extend(render_row(row) for row in rows)
    lines.append(render_separator("└", "┴", "┘"))
    return "\n".join(lines)


def _format_value(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int | float):
        return f"{value:,.2f}"
    return str(value)
