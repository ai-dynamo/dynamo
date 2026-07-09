# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from .protocol import (
    Counters,
    LEASED_REGISTRATION_VERSION,
    LIFECYCLE_MODES,
    QUERY_MODES,
    WIRE_VERSION,
    LatencyRecorder,
)
from .server import Telemetry
from .workload import PhaseResult, WorkloadSetup

def validate_measured_state(
    mode: str,
    counters: Counters,
    before: Telemetry,
    after: Telemetry,
    telemetry: dict[str, Any],
    setup: WorkloadSetup,
) -> None:
    if after.admission_reservations != 0 and mode in LIFECYCLE_MODES:
        raise RuntimeError(
            f"reservation lifecycle leaked {after.admission_reservations} entries"
        )
    node_delta = after.module_stats[0] - before.module_stats[0]
    mutation_delta = after.module_stats[2] - before.module_stats[2]
    if mode == "churn_owned":
        expected_nodes = len(setup.block_hashes)
        if before.module_stats[0] != expected_nodes:
            raise RuntimeError(
                "owned churn warmup retained radix state: "
                f"nodes={before.module_stats[0]}/{expected_nodes}"
            )
        if after.module_stats[0] != expected_nodes:
            raise RuntimeError(
                "owned churn retained radix state: "
                f"nodes={after.module_stats[0]}/{expected_nodes}"
            )
        expected_lifecycle = [len(setup.topology.workers), 0, 0]
        if (
            before.lifecycle_stats != expected_lifecycle
            or after.lifecycle_stats != expected_lifecycle
        ):
            raise RuntimeError(
                "owned churn changed worker lifecycle state: "
                f"before={before.lifecycle_stats}, after={after.lifecycle_stats}, "
                f"expected={expected_lifecycle}"
            )
        for label, stats in (("warmup", before.gc_stats), ("measured", after.gc_stats)):
            if stats is None or len(stats) != 8:
                raise RuntimeError(f"owned churn {label} GC diagnostics are missing")
            if stats[2] or stats[4] or stats[5] or stats[6]:
                raise RuntimeError(
                    f"owned churn {label} retained legacy/owner residue: {stats}"
                )
        expected_events = {"remove": counters.iterations, "store": counters.iterations}
        if counters.events_by_kind != expected_events:
            raise RuntimeError(
                "owned churn did not complete balanced STORE/REMOVE pairs: "
                f"{counters.events_by_kind!r} != {expected_events!r}"
            )
        if node_delta != 0 or mutation_delta != counters.events:
            raise RuntimeError(
                "owned churn state delta does not match accepted pairs: "
                f"nodes={node_delta}/0, mutations={mutation_delta}/{counters.events}"
            )
    elif mode in {"apply", "apply_owned", "mixed"}:
        if node_delta != counters.blocks or mutation_delta != counters.events:
            raise RuntimeError(
                "module state delta does not match accepted events: "
                f"nodes={node_delta}/{counters.blocks}, "
                f"mutations={mutation_delta}/{counters.events}"
            )
    elif mode in QUERY_MODES - {"mixed"}:
        if node_delta != 0 or mutation_delta != 0:
            raise RuntimeError(
                f"read-only mode mutated module state: nodes={node_delta}, "
                f"mutations={mutation_delta}"
            )
    elif mode in LIFECYCLE_MODES:
        minimum = counters.reservation_cycles * 2
        maximum = counters.reservation_cycles * (3 if mode == "renew" else 2)
        if not minimum <= mutation_delta <= maximum:
            raise RuntimeError(
                f"reservation mutation delta {mutation_delta} is outside "
                f"expected range [{minimum}, {maximum}]"
            )

    commandstats = telemetry["commandstats"]
    for command, expected_calls in counters.commands_by_kind.items():
        fields = commandstats.get(f"dynkv.{command}")
        actual_calls = fields.get("calls") if isinstance(fields, dict) else None
        if actual_calls != expected_calls:
            raise RuntimeError(
                f"commandstats call mismatch for {command}: "
                f"{actual_calls!r} != {expected_calls}"
            )


def rate(value: int, elapsed_s: float) -> float:
    return value / elapsed_s


def sample_result(
    *,
    mode: str,
    appendonly: bool,
    connections: int,
    pipeline: int,
    blocks_per_event: int,
    setup: WorkloadSetup,
    phase: PhaseResult,
    latency: LatencyRecorder,
    telemetry: dict[str, Any],
    repetition: int,
    duration_s: float | None,
) -> dict[str, Any]:
    counters = phase.counters
    elapsed_s = phase.elapsed_s
    event_mib_per_s = rate(counters.event_bytes, elapsed_s) / (1024 * 1024)
    request_wire_mib_per_s = rate(counters.request_wire_bytes, elapsed_s) / (
        1024 * 1024
    )
    response_wire_mib_per_s = rate(counters.response_wire_bytes, elapsed_s) / (
        1024 * 1024
    )
    max_outstanding = connections * pipeline
    if mode in {"mixed", "churn_owned"}:
        max_outstanding *= 2
    result = {
        "schema_version": 2,
        "status": "ok",
        "repetition": repetition,
        "mode": mode,
        "appendonly": appendonly,
        "connections": connections,
        "pipeline": pipeline,
        "max_outstanding_commands_per_batch": max_outstanding,
        "duration_target_s": duration_s,
        "commands": counters.commands,
        "events": counters.events,
        "event_bytes": counters.event_bytes,
        "elapsed_s": elapsed_s,
        "commands_per_s": rate(counters.commands, elapsed_s),
        "events_per_s": rate(counters.events, elapsed_s),
        "events_by_kind": dict(sorted(counters.events_by_kind.items())),
        "events_per_s_by_kind": {
            kind: rate(count, elapsed_s)
            for kind, count in sorted(counters.events_by_kind.items())
        },
        "event_mib_per_s": event_mib_per_s,
        "iterations": counters.iterations,
        "iterations_per_s": rate(counters.iterations, elapsed_s),
        "blocks": counters.blocks,
        "blocks_per_event": blocks_per_event,
        "blocks_per_s": rate(counters.blocks, elapsed_s),
        "queries": counters.queries,
        "queries_per_s": rate(counters.queries, elapsed_s),
        "selections": counters.selections,
        "selections_per_s": rate(counters.selections, elapsed_s),
        "reservation_cycles": counters.reservation_cycles,
        "reservation_cycles_per_s": rate(counters.reservation_cycles, elapsed_s),
        "logical_payload_bytes": counters.logical_payload_bytes,
        "logical_payload_mib_per_s": rate(counters.logical_payload_bytes, elapsed_s)
        / (1024 * 1024),
        "request_wire_bytes": counters.request_wire_bytes,
        "response_wire_bytes": counters.response_wire_bytes,
        "request_wire_mib_per_s": request_wire_mib_per_s,
        "response_wire_mib_per_s": response_wire_mib_per_s,
        "total_wire_mib_per_s": request_wire_mib_per_s + response_wire_mib_per_s,
        "commands_by_kind": dict(sorted(counters.commands_by_kind.items())),
        "latency": latency.summary(),
        "topology": {
            "preset": setup.topology.preset,
            "logical_workers": len(setup.topology.workers),
            "logical_frontends": setup.topology.frontends,
            "connections_per_frontend": [
                connections // setup.topology.frontends
                + int(index < connections % setup.topology.frontends)
                for index in range(setup.topology.frontends)
            ],
            "prefix_owners": len(setup.topology.owners),
            "leased_registration": setup.topology.leased,
            "candidates": len(setup.admission_candidates),
            "query_blocks": len(setup.local_hashes),
            "event_wire_version": WIRE_VERSION,
            "leased_registration_version": LEASED_REGISTRATION_VERSION,
        },
        "telemetry": telemetry,
    }
    return result
