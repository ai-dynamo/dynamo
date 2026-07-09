# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .protocol import LatencyRecorder, MODE_ALIASES, free_port
from .server import (
    Telemetry,
    capture_telemetry,
    shutdown,
    start_server,
    telemetry_summary,
    wait_ready,
)
from .validation import sample_result, validate_measured_state
from .workload import build_topology, execute, match_validator, run_phase, setup_workload

async def run_sample(
    args: argparse.Namespace,
    *,
    connections: int,
    pipeline: int,
    repetition: int,
    directory: Path,
) -> dict[str, Any]:
    mode = MODE_ALIASES.get(args.mode, args.mode)
    port = free_port()
    process = start_server(
        args.server,
        args.module,
        directory,
        port,
        args.appendonly,
        args.appendfsync,
        args.auto_aof_rewrite_percentage,
    )
    try:
        await wait_ready(port, process)
        topology = build_topology(args, connections, mode)
        setup = await setup_workload(
            port,
            args.key.encode(),
            topology,
            args.query_blocks or args.blocks_per_event,
            args.capacity,
            args.lease_ms,
            args.worker_lease_ms,
        )
        if args.warmup_seconds:
            await run_phase(
                port=port,
                setup=setup,
                mode=mode,
                connections=connections,
                pipeline=pipeline,
                blocks_per_event=args.blocks_per_event,
                churn_prefixes_per_connection=args.churn_prefixes_per_connection,
                count=None,
                duration_s=args.warmup_seconds,
                phase_id=0,
                latency=None,
            )

        latency = LatencyRecorder(args.latency_sample_limit)

        async def baseline() -> Telemetry:
            return await capture_telemetry(
                port,
                process,
                setup.key,
                before=True,
                include_gc_stats=mode == "churn_owned",
            )

        phase, before = await run_phase(
            port=port,
            setup=setup,
            mode=mode,
            connections=connections,
            pipeline=pipeline,
            blocks_per_event=args.blocks_per_event,
            churn_prefixes_per_connection=args.churn_prefixes_per_connection,
            count=None if args.duration_seconds is not None else args.events,
            duration_s=args.duration_seconds,
            phase_id=1,
            latency=latency,
            on_ready=baseline,
        )
        after = await capture_telemetry(
            port,
            process,
            setup.key,
            before=False,
            include_gc_stats=mode == "churn_owned",
        )
        telemetry = telemetry_summary(before, after, phase.elapsed_s)
        validate_measured_state(mode, phase.counters, before, after, telemetry, setup)
        if mode == "churn_owned":
            match_validator(setup)(
                await execute(port, b"DYNKV.MATCH", setup.key, setup.match_payload)
            )
        return sample_result(
            mode=mode,
            appendonly=args.appendonly,
            connections=connections,
            pipeline=pipeline,
            blocks_per_event=args.blocks_per_event,
            setup=setup,
            phase=phase,
            latency=latency,
            telemetry=telemetry,
            repetition=repetition,
            duration_s=args.duration_seconds,
        )
    finally:
        await shutdown(port, process)


def parse_positive_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("sweep values must be positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("sweep values must be unique")
    return values


def build_sweep_schedule(
    connections: Sequence[int], pipelines: Sequence[int], repetitions: int
) -> list[tuple[int, int, int]]:
    points = [
        (connection, pipeline) for connection in connections for pipeline in pipelines
    ]
    schedule = []
    for repetition in range(1, repetitions + 1):
        offset = (repetition - 1) % len(points)
        rotated = points[offset:] + points[:offset]
        schedule.extend(
            (repetition, connection, pipeline) for connection, pipeline in rotated
        )
    return schedule


def campaign_summary(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(
            (int(sample["connections"]), int(sample["pipeline"])), []
        ).append(sample)
    points = []
    for (connections, pipeline), values in sorted(grouped.items()):
        iterations = [float(value["iterations_per_s"]) for value in values]
        commands = [float(value["commands_per_s"]) for value in values]
        p99_values = []
        for value in values:
            p99 = sample_p99_ms(value)
            if p99 is not None:
                p99_values.append(p99)
        points.append(
            {
                "connections": connections,
                "pipeline": pipeline,
                "outstanding_iterations": connections * pipeline,
                "samples": len(values),
                "iterations_per_s_median": statistics.median(iterations),
                "iterations_per_s_min": min(iterations),
                "iterations_per_s_max": max(iterations),
                "commands_per_s_median": statistics.median(commands),
                "p99_ms_median_across_commands": (
                    statistics.median(p99_values) if p99_values else None
                ),
            }
        )
    peak = max(points, key=lambda point: point["iterations_per_s_median"])
    threshold = float(peak["iterations_per_s_median"]) * 0.95
    knee = min(
        (
            point
            for point in points
            if float(point["iterations_per_s_median"]) >= threshold
        ),
        key=lambda point: (
            point["outstanding_iterations"],
            point["connections"],
            point["pipeline"],
        ),
    )
    return {
        "points": points,
        "peak_observed": peak,
        "closed_loop_knee": knee,
        "knee_definition": (
            "smallest connection/pipeline window reaching at least 95% of the "
            "peak median logical-iteration rate; latency still determines whether "
            "this point meets an application SLO"
        ),
    }


def sample_p99_ms(sample: dict[str, Any]) -> float | None:
    """Return the conservative client p99 across commands in one sample."""

    values = []
    latency = sample.get("latency")
    if isinstance(latency, dict):
        for summary in latency.values():
            if not isinstance(summary, dict):
                continue
            value = summary.get("p99_ms")
            if isinstance(value, int | float) and math.isfinite(float(value)):
                values.append(float(value))
    return max(values) if values else None
