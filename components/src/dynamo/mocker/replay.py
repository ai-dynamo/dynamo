#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any

from dynamo.llm import run_mocker_trace_replay


def default_replay_output_path(trace_file: Path) -> Path:
    return trace_file.with_name(f"{trace_file.stem}.replay.json")


def format_latency_summary(report: dict[str, Any], label: str, key_prefix: str) -> str:
    return (
        f"{label}: mean={report[f'mean_{key_prefix}_ms']:.3f} ms, "
        f"median={report[f'median_{key_prefix}_ms']:.3f} ms, "
        f"p95={report[f'p95_{key_prefix}_ms']:.3f} ms, "
        f"p99={report[f'p99_{key_prefix}_ms']:.3f} ms"
    )


def print_replay_summary(report: dict[str, Any], output_file: Path) -> None:
    lines = [
        "Replay Summary",
        f"Completed requests: {report['completed_requests']}/{report['num_requests']}",
        (
            f"Virtual duration: {report['duration_ms']:.3f} ms | "
            f"Wall time: {report['wall_time_ms']:.3f} ms"
        ),
        (
            f"Tokens: input={report['total_input_tokens']} "
            f"output={report['total_output_tokens']}"
        ),
        (
            "Throughput: "
            f"requests={report['request_throughput_rps']:.3f} req/s, "
            f"input={report['input_throughput_tok_s']:.3f} tok/s, "
            f"output={report['output_throughput_tok_s']:.3f} tok/s, "
            f"total={report['total_throughput_tok_s']:.3f} tok/s"
        ),
        f"Queue latency: mean={report['mean_queue_ms']:.3f} ms",
        format_latency_summary(report, "TTFT", "ttft"),
        format_latency_summary(report, "TPOT", "tpot"),
        format_latency_summary(report, "ITL", "itl"),
        format_latency_summary(report, "E2E latency", "e2e_latency"),
        (
            f"Prefix cache reused ratio: {report['prefix_cache_reused_ratio']:.6f} | "
            f"Max ITL: {report['max_itl_ms']:.3f} ms"
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
