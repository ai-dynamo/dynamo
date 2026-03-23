# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

TITLE = "NVIDIA AIPerf | LLM Metrics"
STAT_COLUMNS = ("avg", "min", "max", "p99", "p90", "p75", "std")


def default_report_path() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / f"dynamo_replay_report_{timestamp}.json"


def write_report_json(
    report: dict[str, object], output_path: str | Path | None
) -> Path:
    path = Path(output_path) if output_path is not None else default_report_path()
    if path.exists() and path.is_dir():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = path / f"dynamo_replay_report_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def format_report_table(report: dict[str, object]) -> str:
    rows = _build_rows(report)
    table = _render_table(rows)
    lines = [TITLE, table]

    wall_time_ms = report.get("wall_time_ms")
    if isinstance(wall_time_ms, int | float):
        lines.append(f"Wall Time (ms): {_format_value(wall_time_ms)}")

    prefix_cache_reused_ratio = report.get("prefix_cache_reused_ratio")
    if isinstance(prefix_cache_reused_ratio, int | float):
        lines.append(
            f"Prefix Cache Reused Ratio: {_format_value(prefix_cache_reused_ratio)}"
        )

    return "\n".join(lines)


def _build_rows(report: dict[str, object]) -> list[list[str]]:
    rows: list[list[str]] = []
    _append_stat_row(rows, report, "Time to First Token (ms)", "ttft_ms")
    _append_stat_row(rows, report, "Time to Second Token (ms)", "ttst_ms")
    _append_stat_row(rows, report, "Request Latency (ms)", "e2e_latency_ms")
    _append_stat_row(rows, report, "Inter Token Latency (ms)", "itl_ms")
    _append_stat_row(
        rows,
        report,
        "Output Token Throughput Per User (tokens/sec/user)",
        "output_token_throughput_per_user",
    )
    rows.append(
        [
            "Output Token Throughput (tokens/sec)",
            _format_value(report.get("output_throughput_tok_s")),
            *["N/A"] * (len(STAT_COLUMNS) - 1),
        ]
    )
    rows.append(
        [
            "Request Throughput (requests/sec)",
            _format_value(report.get("request_throughput_rps")),
            *["N/A"] * (len(STAT_COLUMNS) - 1),
        ]
    )
    rows.append(
        [
            "Request Count (requests)",
            _format_value(report.get("completed_requests", report.get("num_requests"))),
            *["N/A"] * (len(STAT_COLUMNS) - 1),
        ]
    )
    return rows


def _append_stat_row(
    rows: list[list[str]], report: dict[str, object], label: str, metric_suffix: str
) -> None:
    mean_key = f"mean_{metric_suffix}"
    if mean_key not in report:
        return

    rows.append(
        [
            label,
            _format_value(report.get(mean_key)),
            _format_value(report.get(f"min_{metric_suffix}")),
            _format_value(report.get(f"max_{metric_suffix}")),
            _format_value(report.get(f"p99_{metric_suffix}")),
            _format_value(report.get(f"p90_{metric_suffix}")),
            _format_value(report.get(f"p75_{metric_suffix}")),
            _format_value(report.get(f"std_{metric_suffix}")),
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
                continue
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
