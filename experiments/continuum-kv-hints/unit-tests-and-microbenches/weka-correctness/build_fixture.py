# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a small, deterministic lifecycle fixture from the WEKA trace corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_TRACE_ID = "7a0f824af65bca68027591b288ae5f3223a2"
DEFAULT_SUBAGENT_ID = "subagent_006_63824fc7"


def load_trace(source: Path, trace_id: str) -> dict[str, Any]:
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            trace = json.loads(line)
            if trace.get("id") == trace_id:
                return trace
    raise ValueError(f"trace {trace_id!r} not found in {source}")


def normalize_request(request: dict[str, Any], timestamp: float) -> dict[str, Any]:
    normalized = dict(request)
    normalized["t"] = timestamp
    normalized["out"] = 1
    normalized["api_time"] = 0.01
    normalized["think_time"] = 0.0
    if "ttft" in normalized:
        normalized["ttft"] = 0.005
    return normalized


def build_fixture(
    trace: dict[str, Any], subagent_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    requests = trace["requests"]
    subagent_index, subagent = next(
        (index, request)
        for index, request in enumerate(requests)
        if request.get("type") == "subagent" and request.get("agent_id") == subagent_id
    )
    child_requests = subagent.get("requests", [])
    if not child_requests:
        raise ValueError(f"subagent {subagent_id!r} has no requests")

    root_requests = [
        request
        for request in requests[:subagent_index]
        if request.get("type") != "subagent"
    ][:2]
    if len(root_requests) != 2:
        raise ValueError("the correctness fixture requires two parent turns")

    normalized_subagent = dict(subagent)
    normalized_subagent["t"] = 1.0
    normalized_subagent["duration_ms"] = len(child_requests) * 10
    normalized_subagent["requests"] = [
        normalize_request(request, float(index + 1))
        for index, request in enumerate(child_requests)
    ]

    fixture = {
        "id": trace["id"],
        "models": trace["models"],
        "block_size": trace["block_size"],
        "hash_id_scope": trace["hash_id_scope"],
        "requests": [
            normalize_request(request, float(index))
            for index, request in enumerate(root_requests)
        ]
        + [normalized_subagent],
    }
    provenance = {
        "source": str(trace.get("id")),
        "trace_id": trace["id"],
        "root_request_indices": [requests.index(request) for request in root_requests],
        "subagent_outer_request_index": subagent_index,
        "subagent_id": subagent_id,
        "subagent_status": subagent.get("status"),
        "subagent_request_count": len(child_requests),
        "root_request_count": len(root_requests),
        "source_input_tokens": [request["in"] for request in child_requests],
        "source_output_tokens": [request["out"] for request in child_requests],
        "transformations": [
            "kept the source input lengths and hash_ids",
            "selected one root request and one completed subagent",
            "set output lengths to one token",
            "compressed timestamps and request delays",
        ],
    }
    return fixture, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--trace-id", default=DEFAULT_TRACE_ID)
    parser.add_argument("--subagent-id", default=DEFAULT_SUBAGENT_ID)
    args = parser.parse_args()

    trace = load_trace(args.source, args.trace_id)
    fixture, provenance = build_fixture(trace, args.subagent_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trace.json").write_text(
        json.dumps(fixture, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )

    print(
        f"wrote {provenance['root_request_count'] + provenance['subagent_request_count']} "
        "requests to "
        f"{args.output_dir / 'trace.json'}"
    )


if __name__ == "__main__":
    main()
