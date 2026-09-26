#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize selected frontend counter rates during the send window."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METRICS = (
    "dynamo_component_kv_cache_events_applied",
    "dynamo_component_kv_cache_event_warnings",
    "dynamo_component_router_kv_zmq_ingress_batches",
)


def metric_total(snapshot: dict[str, Any], name: str) -> float | None:
    values = snapshot.get("metrics", {}).get(name)
    if not isinstance(values, list):
        return None
    total = 0.0
    found = False
    for item in values:
        if isinstance(item, dict) and item.get("value") is not None:
            total += float(item["value"])
            found = True
    return total if found else None


def timestamp(path: Path) -> float:
    return dt.datetime.fromisoformat(path.read_text().strip()).timestamp()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    start = timestamp(args.run_dir / "MEASUREMENT_STARTED")
    end = timestamp(args.run_dir / "SENDING_ENDED")
    snapshots: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source = args.run_dir / "load_artifacts" / "server_metrics_export.jsonl"
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        observed = float(row["timestamp_ns"]) / 1_000_000_000.0
        if start <= observed <= end:
            snapshots[str(row["endpoint_url"])].append(row)
    endpoints: dict[str, Any] = {}
    for endpoint, rows in sorted(snapshots.items()):
        rows.sort(key=lambda row: row["timestamp_ns"])
        duration = (
            float(rows[-1]["timestamp_ns"] - rows[0]["timestamp_ns"])
            / 1_000_000_000.0
        )
        metrics: dict[str, Any] = {}
        for name in METRICS:
            first = metric_total(rows[0], name)
            last = metric_total(rows[-1], name)
            metrics[name] = {
                "first": first,
                "last": last,
                "delta": last - first if first is not None and last is not None else None,
                "per_second": (
                    (last - first) / duration
                    if duration > 0 and first is not None and last is not None
                    else None
                ),
            }
        endpoints[endpoint] = {
            "duration_seconds": duration,
            "snapshots": len(rows),
            "metrics": metrics,
        }
    result = {
        "measurement_started": start,
        "sending_ended": end,
        "endpoints": endpoints,
        "note": "Latest main exposes no direct KV-metrics application counter.",
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
