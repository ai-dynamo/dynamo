#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

from dynamo.llm import run_mocker_trace_replay


def default_replay_output_path(trace_file: Path) -> Path:
    return trace_file.with_name(f"{trace_file.stem}.replay.json")


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def format_row(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    return "\n".join(
        [format_row(headers), separator, *(format_row(row) for row in rows)]
    )


def format_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def format_number(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}"


def print_replay_summary(report: dict[str, Any], output_file: Path) -> None:
    scalar_rows = [
        ["Request count", str(report["num_requests"])],
        ["Completed requests", str(report["completed_requests"])],
        ["Virtual duration (ms)", f"{report['duration_ms']:.3f}"],
        ["Wall time (ms)", f"{report['wall_time_ms']:.3f}"],
        ["Input tokens", str(report["total_input_tokens"])],
        ["Output tokens", str(report["total_output_tokens"])],
        ["Request throughput (req/s)", f"{report['request_throughput_rps']:.3f}"],
        ["Input throughput (tok/s)", f"{report['input_throughput_tok_s']:.3f}"],
        ["Output throughput (tok/s)", f"{report['output_throughput_tok_s']:.3f}"],
        ["Total throughput (tok/s)", f"{report['total_throughput_tok_s']:.3f}"],
        ["Prefix cache reused ratio", f"{report['prefix_cache_reused_ratio']:.6f}"],
    ]
    latency_rows = [
        [
            "TTFT",
            format_ms(report["mean_ttft_ms"]),
            format_ms(report["min_ttft_ms"]),
            format_ms(report["max_ttft_ms"]),
            format_ms(report["p99_ttft_ms"]),
            format_ms(report["p90_ttft_ms"]),
            format_ms(report["median_ttft_ms"]),
            format_ms(report["p75_ttft_ms"]),
            format_ms(report["std_ttft_ms"]),
        ],
        [
            "TTST",
            format_ms(report["mean_ttst_ms"]),
            format_ms(report["min_ttst_ms"]),
            format_ms(report["max_ttst_ms"]),
            format_ms(report["p99_ttst_ms"]),
            format_ms(report["p90_ttst_ms"]),
            format_ms(report["median_ttst_ms"]),
            format_ms(report["p75_ttst_ms"]),
            format_ms(report["std_ttst_ms"]),
        ],
        [
            "TPOT",
            format_ms(report["mean_tpot_ms"]),
            format_ms(report["min_tpot_ms"]),
            format_ms(report["max_tpot_ms"]),
            format_ms(report["p99_tpot_ms"]),
            format_ms(report["p90_tpot_ms"]),
            format_ms(report["median_tpot_ms"]),
            format_ms(report["p75_tpot_ms"]),
            format_ms(report["std_tpot_ms"]),
        ],
        [
            "ITL",
            format_ms(report["mean_itl_ms"]),
            format_ms(report["min_itl_ms"]),
            format_ms(report["max_itl_ms"]),
            format_ms(report["p99_itl_ms"]),
            format_ms(report["p90_itl_ms"]),
            format_ms(report["median_itl_ms"]),
            format_ms(report["p75_itl_ms"]),
            format_ms(report["std_itl_ms"]),
        ],
        [
            "E2E latency",
            format_ms(report["mean_e2e_latency_ms"]),
            format_ms(report["min_e2e_latency_ms"]),
            format_ms(report["max_e2e_latency_ms"]),
            format_ms(report["p99_e2e_latency_ms"]),
            format_ms(report["p90_e2e_latency_ms"]),
            format_ms(report["median_e2e_latency_ms"]),
            format_ms(report["p75_e2e_latency_ms"]),
            format_ms(report["std_e2e_latency_ms"]),
        ],
        [
            "Output TPS/User",
            format_number(report["mean_output_token_throughput_per_user"]),
            format_number(report["min_output_token_throughput_per_user"]),
            format_number(report["max_output_token_throughput_per_user"]),
            format_number(report["p99_output_token_throughput_per_user"]),
            format_number(report["p90_output_token_throughput_per_user"]),
            format_number(report["median_output_token_throughput_per_user"]),
            format_number(report["p75_output_token_throughput_per_user"]),
            format_number(report["std_output_token_throughput_per_user"]),
        ],
    ]
    lines = [
        "Replay Summary",
        format_table(["Metric", "Value"], scalar_rows),
        "",
        format_table(
            ["Metric", "avg", "min", "max", "p99", "p90", "p50", "p75", "std"],
            latency_rows,
        ),
        f"JSON report: {output_file}",
    ]
    print("\n".join(lines))


def write_replay_report(report: dict[str, Any], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)


def run_trace_replay(
    trace_file: Path,
    output_file: Path | None,
    extra_engine_args: Path,
    num_workers: int,
    replay_concurrency: int | None,
) -> None:
    resolved_output_file = output_file or default_replay_output_path(trace_file)
    report = run_mocker_trace_replay(
        trace_file=trace_file,
        extra_engine_args=extra_engine_args,
        num_workers=num_workers,
        replay_concurrency=replay_concurrency,
    )
    write_replay_report(report, resolved_output_file)
    print_replay_summary(report, resolved_output_file)
