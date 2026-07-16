# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail closed unless prefill and decode use CUDA graphs in an nsys trace."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

EXECUTE_CONTEXT_RE = re.compile(
    r"^execute_context_(?P<context_requests>\d+)\((?P<context_tokens>\d+)\)_"
    r"generation_(?P<generation_requests>\d+)\((?P<generation_tokens>\d+)\)$"
)


def row_count(connection: sqlite3.Connection, table: str) -> int:
    quoted = table.replace('"', '""')
    return int(connection.execute(f'SELECT COUNT(*) FROM "{quoted}"').fetchone()[0])


def load_execute_ranges(
    connection: sqlite3.Connection,
) -> list[dict[str, int | str]]:
    rows = connection.execute(
        """
        SELECT N.start, N."end", N.globalTid, COALESCE(N.text, S.value)
        FROM NVTX_EVENTS AS N
        LEFT JOIN StringIds AS S ON S.id = N.textId
        WHERE N."end" IS NOT NULL
          AND COALESCE(N.text, S.value) LIKE 'execute_context_%'
        """
    ).fetchall()
    ranges: list[dict[str, int | str]] = []
    for start, end, global_tid, name in rows:
        match = EXECUTE_CONTEXT_RE.match(str(name))
        if match is None:
            continue
        ranges.append(
            {
                "start": int(start),
                "end": int(end),
                "global_tid": int(global_tid),
                "name": str(name),
                **{key: int(value) for key, value in match.groupdict().items()},
            }
        )
    return ranges


def graph_launches(connection: sqlite3.Connection) -> list[tuple[int, int]]:
    return [
        (int(start), int(global_tid))
        for start, global_tid in connection.execute(
            """
            SELECT R.start, R.globalTid
            FROM CUPTI_ACTIVITY_KIND_RUNTIME AS R
            JOIN StringIds AS S ON S.id = R.nameId
            WHERE S.value LIKE 'cudaGraphLaunch%'
            """
        )
    ]


def phase_graph_audit(
    ranges: list[dict[str, int | str]],
    launches: list[tuple[int, int]],
) -> dict[str, Any]:
    prefill_ranges = [
        item
        for item in ranges
        if item["context_requests"] > 0 and item["generation_requests"] == 0
    ]
    decode_ranges = [item for item in ranges if item["generation_requests"] > 0]

    def count_launches(item: dict[str, int | str]) -> int:
        return sum(
            1
            for launch_start, global_tid in launches
            if global_tid == item["global_tid"]
            and item["start"] <= launch_start <= item["end"]
        )

    prefill_counts = [count_launches(item) for item in prefill_ranges]
    decode_counts = [count_launches(item) for item in decode_ranges]
    return {
        "prefill_ranges": len(prefill_ranges),
        "prefill_tokens": sorted(
            {int(item["context_tokens"]) for item in prefill_ranges}
        ),
        "prefill_graph_launches": sum(prefill_counts),
        "prefill_ranges_without_graph": sum(count == 0 for count in prefill_counts),
        "decode_ranges": len(decode_ranges),
        "decode_graph_launches": sum(decode_counts),
        "decode_ranges_without_graph": sum(count == 0 for count in decode_counts),
    }


def audit(rep_path: Path, sqlite_path: Path, summary_path: Path) -> dict[str, Any]:
    """Validate a finalized report, its SQLite export, and experiment summary."""
    failures: list[str] = []
    if rep_path.suffix != ".nsys-rep" or not rep_path.is_file():
        failures.append(f"missing finalized nsys report: {rep_path}")
    elif rep_path.stat().st_size <= 0:
        failures.append(f"empty nsys report: {rep_path}")
    if not sqlite_path.is_file() or sqlite_path.stat().st_size <= 0:
        failures.append(f"missing nsys SQLite export: {sqlite_path}")
    if not summary_path.is_file():
        failures.append(f"missing experiment summary: {summary_path}")
    if failures:
        return {"accepted": False, "failures": failures}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_requests = int(summary["config"]["requests"])
    expected_output_tokens = int(summary["config"]["output_tokens"])
    expected_prefill_ranges = expected_requests
    expected_decode_ranges = expected_requests * (expected_output_tokens - 1)
    resolved = summary["resolved_engine"]
    prompt_tokens = int(summary["config"]["prompt_tokens"])
    block_size = int(summary["config"]["block_size"])
    prefix_cache_remainder = prompt_tokens % block_size or block_size
    requested_capture_sizes = {1, prefix_cache_remainder, prompt_tokens}

    with sqlite3.connect(sqlite_path) as connection:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required_tables = {
            "StringIds",
            "NVTX_EVENTS",
            "CUPTI_ACTIVITY_KIND_RUNTIME",
            "CUDA_GRAPH_NODE_EVENTS",
        }
        missing_tables = sorted(required_tables - tables)
        if missing_tables:
            failures.append(f"nsys SQLite is missing tables {missing_tables}")
            return {"accepted": False, "failures": failures}

        runtime_count = row_count(connection, "CUPTI_ACTIVITY_KIND_RUNTIME")
        kernel_tables = sorted(
            table for table in tables if table.startswith("CUPTI_ACTIVITY_KIND_KERNEL")
        )
        kernel_count = sum(row_count(connection, table) for table in kernel_tables)
        graph_node_count = row_count(connection, "CUDA_GRAPH_NODE_EVENTS")
        launches = graph_launches(connection)
        phase_graphs = phase_graph_audit(load_execute_ranges(connection), launches)

    if not summary.get("accepted"):
        failures.append("experiment summary was not accepted")
    if resolved["cuda_graph_mode"] != "FULL":
        failures.append(
            f"resolved CUDA graph mode is {resolved['cuda_graph_mode']!r}, not FULL"
        )
    if resolved["enforce_eager"]:
        failures.append("resolved engine enabled enforce_eager")
    if not resolved["prefix_caching"]:
        failures.append("resolved engine disabled prefix caching")
    missing_capture_sizes = sorted(
        requested_capture_sizes - set(resolved["cuda_graph_capture_sizes"])
    )
    if missing_capture_sizes:
        failures.append(f"resolved capture sizes are missing {missing_capture_sizes}")
    if runtime_count <= 0:
        failures.append("nsys trace has no CUDA runtime events")
    if kernel_count <= 0:
        failures.append("nsys trace has no CUDA kernel events")
    if graph_node_count <= 0:
        failures.append("nsys trace has no CUDA graph node events")
    if not launches:
        failures.append("nsys trace has no cudaGraphLaunch calls")
    if phase_graphs["prefill_ranges"] != expected_prefill_ranges:
        failures.append(
            f"prefill ranges={phase_graphs['prefill_ranges']}, "
            f"expected {expected_prefill_ranges}"
        )
    if phase_graphs["decode_ranges"] != expected_decode_ranges:
        failures.append(
            f"decode ranges={phase_graphs['decode_ranges']}, "
            f"expected {expected_decode_ranges}"
        )
    if phase_graphs["prefill_ranges_without_graph"]:
        failures.append(
            f"{phase_graphs['prefill_ranges_without_graph']} prefill ranges "
            "did not launch a CUDA graph"
        )
    if phase_graphs["decode_ranges_without_graph"]:
        failures.append(
            f"{phase_graphs['decode_ranges_without_graph']} decode ranges "
            "did not launch a CUDA graph"
        )

    return {
        "accepted": not failures,
        "report": str(rep_path.resolve()),
        "report_bytes": rep_path.stat().st_size,
        "sqlite": str(sqlite_path.resolve()),
        "cuda_runtime_events": runtime_count,
        "cuda_kernel_events": kernel_count,
        "cuda_graph_node_events": graph_node_count,
        "cuda_graph_launches": len(launches),
        "phase_graphs": phase_graphs,
        "kernel_tables": kernel_tables,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit prefill and decode CUDA graph use in an nsys trace."
    )
    parser.add_argument("rep", type=Path)
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = audit(args.rep, args.sqlite, args.summary)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not result["accepted"]:
        raise SystemExit("NSYS_AUDIT_FAILED: " + "; ".join(result["failures"]))
    phase_graphs = result["phase_graphs"]
    print(
        "NSYS_AUDIT=PASS "
        f"prefill_ranges={phase_graphs['prefill_ranges']} "
        f"prefill_graph_launches={phase_graphs['prefill_graph_launches']} "
        f"decode_ranges={phase_graphs['decode_ranges']} "
        f"decode_graph_launches={phase_graphs['decode_graph_launches']}"
    )


if __name__ == "__main__":
    main()
