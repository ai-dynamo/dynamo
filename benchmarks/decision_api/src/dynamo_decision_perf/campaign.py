# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sequential campaign schedule and fail-closed gates; no automatic GPU launch."""

import argparse
import json
import math
from pathlib import Path

from .runner import campaign_points


def can_continue(result: dict, observations: dict) -> bool:
    """Require recorded live safety observations before escalating load."""
    counts = result.get("summary", {}).get("counts", {})
    return (
        result.get("audit", {}).get("status") == "valid"
        and counts.get("schema_errors") == 0
        and counts.get("incomplete_requests") == 0
        and observations.get("oom") is False
        and observations.get("backlog_bounded") is True
        and observations.get("outstanding_drained") is True
        and observations.get("client_saturated") is False
    )


def screen_capacity(results: list[dict]) -> float:
    candidates = []
    for result in results:
        summary = result.get("summary", {})
        counts = summary.get("counts", {})
        rate = summary.get("rates", {}).get("valid_http_requests_per_second")
        if (
            result.get("audit", {}).get("status") == "valid"
            and all(
                counts.get(name) == 0
                for name in (
                    "schema_errors",
                    "http_errors",
                    "timeouts",
                    "incomplete_requests",
                    "rejections",
                    "transport_errors",
                    "cancelled",
                    "pipeline_errors",
                )
            )
            and type(rate) in (int, float)
            and math.isfinite(rate)
            and rate > 0
        ):
            candidates.append(rate)
    if not candidates:
        raise ValueError("no valid error-free screening point; do not escalate")
    return max(candidates)


def planned_runs(shapes: list[str], capacity: float | None = None) -> tuple[dict, ...]:
    if "base" not in shapes:
        raise ValueError("qualified base shape is required")
    serial = (
        {
            "campaign": "serial",
            "shape": "base",
            "concurrency": 1,
            "requests": 8,
            "warmup": 0,
            "duration": 300,
        },
    )
    sensitivity = tuple(
        {
            "campaign": "shape",
            "shape": shape,
            "concurrency": 1,
            "requests": 8,
            "warmup": 0,
            "duration": 300,
        }
        for shape in shapes
        if shape != "base"
    )
    points = campaign_points(capacity or 1)
    screening = tuple(
        {**point, "shape": "base"}
        for point in points
        if point["campaign"] == "screening"
    )
    if capacity is None:
        return serial + sensitivity + screening
    sustained = tuple(
        {**point, "shape": "base"}
        for point in points
        if point["campaign"] == "sustained"
    )
    recovery = (
        {
            "campaign": "overload",
            "shape": "base",
            "rate": 1.25 * capacity,
            "requests": None,
            "duration": 30,
            "warmup": 0,
        },
        {
            "campaign": "recovery",
            "shape": "base",
            "rate": 0.25 * capacity,
            "requests": None,
            "duration": 60,
            "warmup": 0,
        },
    )
    return serial + sensitivity + screening + sustained + recovery


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload-manifest", type=Path, required=True)
    parser.add_argument("--screening-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.workload_manifest.read_text())
    shapes = sorted(
        {key.split(".")[0] for key in manifest["sha256"] if key.endswith(".oai.json")}
    )
    capacity = (
        screen_capacity(json.loads(args.screening_report.read_text())["runs"])
        if args.screening_report
        else None
    )
    with args.output.open("x") as output:
        json.dump(
            {
                "sequential_only": True,
                "requires_gpu_approval": True,
                "screened_capacity": capacity,
                "runs_per_dialect": planned_runs(shapes, capacity),
                "gate": "audit plus live OOM/backlog/drain/client-saturation evidence before each load escalation",
                "noise_pilot": "three same-configuration runs before small-delta claims",
            },
            output,
            indent=2,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
