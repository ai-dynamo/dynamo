# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a deterministic small-model WEKA fixture without committing trace data."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

from aiperf.dataset.loader.weka_trace import _trace_peak_context_length
from aiperf.dataset.loader.weka_trace_models import WekaSubagentEntry, WekaTrace
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-traces", type=int, required=True)
    parser.add_argument("--max-requests-per-trace", type=int, default=120)
    parser.add_argument(
        "--selection-mode",
        choices=("source-order", "highest-request-count"),
        default="source-order",
    )
    parser.add_argument("--max-source-span-seconds", type=float)
    parser.add_argument("--token-scale-factor", type=int, default=8)
    parser.add_argument("--time-scale-factor", type=float, default=0.01)
    parser.add_argument("--max-think-time", type=float, default=2.0)
    parser.add_argument("--max-context-length", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def request_count(trace: WekaTrace) -> int:
    return sum(
        len(request.requests) if isinstance(request, WekaSubagentEntry) else 1
        for request in trace.requests
    )


def has_completed_subagent_and_parent_resume(trace: WekaTrace) -> bool:
    root_times = [
        request.t
        for request in trace.requests
        if not isinstance(request, WekaSubagentEntry)
    ]
    return any(
        subagent.status == "completed"
        and bool(subagent.requests)
        and any(root_time > subagent.t for root_time in root_times)
        for subagent in trace.requests
        if isinstance(subagent, WekaSubagentEntry)
    )


def request_times(trace: WekaTrace) -> list[float]:
    return [
        request.t
        for entry in trace.requests
        for request in (
            entry.requests if isinstance(entry, WekaSubagentEntry) else [entry]
        )
    ]


def trace_span_seconds(trace: WekaTrace) -> float:
    times = request_times(trace)
    return max(times) - min(times) if times else 0.0


class HashGroupInterner:
    def __init__(self) -> None:
        self._group_to_id: dict[tuple[int, ...], int] = {}

    def intern_complete_groups(self, block_hashes: list[int], factor: int) -> list[int]:
        result: list[int] = []
        for offset in range(0, len(block_hashes) - factor + 1, factor):
            group = tuple(block_hashes[offset : offset + factor])
            result.append(self._group_to_id.setdefault(group, len(self._group_to_id)))
        return result


def scaled_positive(value: int, factor: int) -> int:
    return max(1, math.ceil(value / factor))


def scale_request(
    request: dict[str, Any],
    *,
    block_size: int,
    token_factor: int,
    time_factor: float,
    max_think_time: float,
    interner: HashGroupInterner,
) -> dict[str, Any]:
    scaled = dict(request)
    original_hashes = list(request.get("hash_ids") or [])
    scaled_hashes = interner.intern_complete_groups(original_hashes, token_factor)
    represented_tokens = len(original_hashes) * block_size
    unhashed_tokens = max(0, int(request["in"]) - represented_tokens)
    remainder_blocks = len(original_hashes) % token_factor
    scaled_tail = math.ceil(
        (remainder_blocks * block_size + unhashed_tokens) / token_factor
    )

    scaled["hash_ids"] = scaled_hashes
    scaled["in"] = max(1, len(scaled_hashes) * block_size + scaled_tail)
    scaled["out"] = scaled_positive(int(request["out"]), token_factor)
    scaled["t"] = float(request["t"]) * time_factor
    if request.get("think_time") is not None:
        scaled["think_time"] = min(
            float(request["think_time"]) * time_factor, max_think_time
        )
    if request.get("api_time") is not None:
        scaled["api_time"] = float(request["api_time"]) * time_factor
    if request.get("ttft") is not None:
        scaled["ttft"] = float(request["ttft"]) * time_factor
    return scaled


def scale_trace(
    trace: WekaTrace,
    *,
    token_factor: int,
    time_factor: float,
    max_think_time: float,
) -> WekaTrace:
    raw = trace.model_dump(by_alias=True)
    interner = HashGroupInterner()
    scaled_requests: list[dict[str, Any]] = []

    for request in raw["requests"]:
        if request["type"] != "subagent":
            scaled_requests.append(
                scale_request(
                    request,
                    block_size=trace.block_size,
                    token_factor=token_factor,
                    time_factor=time_factor,
                    max_think_time=max_think_time,
                    interner=interner,
                )
            )
            continue

        scaled_subagent = dict(request)
        scaled_subagent["t"] = float(request["t"]) * time_factor
        if request.get("duration_ms") is not None:
            scaled_subagent["duration_ms"] = max(
                1, math.ceil(int(request["duration_ms"]) * time_factor)
            )
        if request.get("total_tokens") is not None:
            scaled_subagent["total_tokens"] = scaled_positive(
                int(request["total_tokens"]), token_factor
            )
        scaled_subagent["tool_tokens"] = math.ceil(
            int(request.get("tool_tokens", 0)) / token_factor
        )
        scaled_subagent["system_tokens"] = math.ceil(
            int(request.get("system_tokens", 0)) / token_factor
        )
        scaled_subagent["requests"] = [
            scale_request(
                child,
                block_size=trace.block_size,
                token_factor=token_factor,
                time_factor=time_factor,
                max_think_time=max_think_time,
                interner=interner,
            )
            for child in request["requests"]
        ]
        scaled_requests.append(scaled_subagent)

    raw["tool_tokens"] = math.ceil(int(raw.get("tool_tokens", 0)) / token_factor)
    raw["system_tokens"] = math.ceil(int(raw.get("system_tokens", 0)) / token_factor)
    raw["requests"] = scaled_requests
    raw["totals"] = None
    return WekaTrace.model_validate(raw)


def trace_summary(
    row_index: int,
    source: WekaTrace,
    scaled: WekaTrace,
) -> dict[str, Any]:
    subagents = [
        request for request in source.requests if isinstance(request, WekaSubagentEntry)
    ]
    return {
        "row_index": row_index,
        "trace_id": source.id,
        "root_request_count": sum(
            not isinstance(request, WekaSubagentEntry) for request in source.requests
        ),
        "subagent_count": len(subagents),
        "subagent_request_count": sum(len(subagent.requests) for subagent in subagents),
        "subagent_types": sorted({subagent.subagent_type for subagent in subagents}),
        "source_span_seconds": trace_span_seconds(source),
        "scaled_span_seconds": trace_span_seconds(scaled),
        "source_peak_context_length": _trace_peak_context_length(source, max_osl=None),
        "scaled_peak_context_length": _trace_peak_context_length(scaled, max_osl=None),
    }


def main() -> None:
    args = parse_args()
    if args.token_scale_factor <= 0:
        raise ValueError("--token-scale-factor must be positive")
    if not 0 < args.time_scale_factor <= 1:
        raise ValueError("--time-scale-factor must be in (0, 1]")

    dataset = load_dataset(args.dataset, split=args.split)
    candidates: list[tuple[int, WekaTrace, WekaTrace]] = []
    for row_index, row in enumerate(dataset):
        source = WekaTrace.model_validate(row)
        if request_count(source) > args.max_requests_per_trace:
            continue
        if not has_completed_subagent_and_parent_resume(source):
            continue
        if (
            args.max_source_span_seconds is not None
            and trace_span_seconds(source) > args.max_source_span_seconds
        ):
            continue
        scaled = scale_trace(
            source,
            token_factor=args.token_scale_factor,
            time_factor=args.time_scale_factor,
            max_think_time=args.max_think_time,
        )
        if _trace_peak_context_length(scaled, max_osl=None) > args.max_context_length:
            continue
        candidates.append((row_index, source, scaled))
        if args.selection_mode == "source-order" and len(candidates) == args.num_traces:
            break

    if args.selection_mode == "highest-request-count":
        candidates.sort(
            key=lambda item: (
                -request_count(item[1]),
                trace_span_seconds(item[1]),
                item[0],
            )
        )
    selected = candidates[: args.num_traces]

    if len(selected) != args.num_traces:
        raise RuntimeError(
            f"selected {len(selected)} eligible traces, expected {args.num_traces}"
        )

    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    traces_dir = args.output_dir / "traces"
    traces_dir.mkdir(parents=True)

    jsonl_lines: list[str] = []
    summaries: list[dict[str, Any]] = []
    for row_index, source, scaled in selected:
        payload = scaled.model_dump(by_alias=True, exclude_none=True)
        encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        jsonl_lines.append(encoded)
        (traces_dir / f"{scaled.id}.json").write_text(encoded + "\n")
        summaries.append(trace_summary(row_index, source, scaled))

    (args.output_dir / "traces.jsonl").write_text("\n".join(jsonl_lines) + "\n")
    manifest = {
        "source_dataset": args.dataset,
        "source_split": args.split,
        "selection_mode": args.selection_mode,
        "selection": (
            "eligible traces have a completed subagent, a later parent request, "
            "at most max_requests_per_trace model requests, optional source-span "
            "limit, and transformed context no larger than max_context_length"
        ),
        "num_traces": args.num_traces,
        "max_requests_per_trace": args.max_requests_per_trace,
        "max_source_span_seconds": args.max_source_span_seconds,
        "token_scale_factor": args.token_scale_factor,
        "time_scale_factor": args.time_scale_factor,
        "max_think_time": args.max_think_time,
        "max_context_length": args.max_context_length,
        "hash_scaling": (
            "each complete group of token_scale_factor source hash blocks is "
            "interned as one logical 64-token block; incomplete groups become "
            "an unhashed tail"
        ),
        "traces": summaries,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
