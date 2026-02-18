#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Append a single router benchmark result to the history JSON used for trend visualization in CI.
Reads the artifact (results_summary.json format), adds run metadata, and appends to history
with a cap on total entries.
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Append benchmark result to history JSON")
    parser.add_argument("--artifact", required=True, help="Path to router_benchmark_results.json")
    parser.add_argument("--history", required=True, help="Path to benchmark_history.json")
    parser.add_argument("--run-id", required=True, help="GitHub run ID")
    parser.add_argument("--timestamp", required=True, help="ISO timestamp of the run")
    parser.add_argument("--ref", required=True, help="Git ref (e.g. refs/heads/main)")
    parser.add_argument("--branch", required=True, help="Branch name")
    parser.add_argument("--arch", required=True, help="Platform arch (e.g. amd64)")
    parser.add_argument("--commit", required=True, help="Commit SHA")
    parser.add_argument("--run-url", required=True, help="URL to the workflow run")
    parser.add_argument("--max-entries", type=int, default=200, help="Max history entries to keep")
    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    history_path = Path(args.history)

    if not artifact_path.exists():
        raise SystemExit(f"Artifact not found: {artifact_path}")

    with open(artifact_path) as f:
        results = json.load(f)

    record = {
        "run_id": args.run_id,
        "timestamp": args.timestamp,
        "ref": args.ref,
        "branch": args.branch,
        "platform_arch": args.arch,
        "commit": args.commit,
        "run_url": args.run_url,
        "results": results,
    }

    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        if not isinstance(history, list):
            history = []
    else:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history = []

    history.append(record)
    if len(history) > args.max_entries:
        history = history[-args.max_entries :]

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Appended run {args.run_id} to {history_path} ({len(history)} entries)")


if __name__ == "__main__":
    main()
