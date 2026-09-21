# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate lifecycle headers captured from the WEKA correctness replay."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture", type=Path)
    args = parser.parse_args()

    records = load_records(args.capture)
    sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        headers = record["headers"]
        session_id = headers.get("x-dynamo-session-id")
        assert session_id, f"missing X-Dynamo-Session-ID: {headers}"
        sessions[session_id].append(headers)

    roots = [
        (session_id, headers)
        for session_id, entries in sessions.items()
        for headers in entries
        if "x-dynamo-parent-session-id" not in headers
    ]
    spawned_roots = [
        (session_id, headers)
        for session_id, headers in roots
        if headers.get("x-dynamo-subagent-spawn") == "true"
    ]
    assert spawned_roots, "no parent request carried X-Dynamo-Subagent-Spawn"

    parent_id = spawned_roots[0][0]
    child_sessions = {
        session_id: entries
        for session_id, entries in sessions.items()
        if any(
            headers.get("x-dynamo-parent-session-id") == parent_id
            for headers in entries
        )
    }
    assert child_sessions, f"no child session references parent {parent_id}"

    child_request_count = sum(len(entries) for entries in child_sessions.values())
    assert (
        child_request_count >= 2
    ), f"expected a multi-request child replay, got {child_request_count}"
    final_counts = Counter(
        headers.get("x-dynamo-session-final")
        for entries in child_sessions.values()
        for headers in entries
    )
    assert final_counts["false"] > 0, f"no continuing child request: {final_counts}"
    assert final_counts["true"] > 0, f"no final child request: {final_counts}"
    assert all(
        headers.get("x-dynamo-parent-session-id") == parent_id
        for entries in child_sessions.values()
        for headers in entries
    )

    compactions = sum(
        headers.get("x-dynamo-compaction") == "inferred"
        for entries in sessions.values()
        for headers in entries
    )
    summary = {
        "request_count": len(records),
        "session_count": len(sessions),
        "parent_session_id": parent_id,
        "child_session_count": len(child_sessions),
        "child_request_count": child_request_count,
        "child_final_markers": dict(final_counts),
        "compaction_markers": compactions,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
