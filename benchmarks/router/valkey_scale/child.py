# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import (
    LOGICAL_MOCKER_WORKERS,
    commandstats_fields,
    finite_number,
    nested_metric,
    replication_offset,
)

def build_child_command(
    args: argparse.Namespace, frontend_count: int, child_output_dir: Path
) -> list[str]:
    """Build one locked-down Valkey-HA authoritative child invocation."""

    command = [
        str(args.python),
        str(args.harness),
        "--arm",
        "valkey_ha",
        "--runs",
        "1",
        "--frontend-count",
        str(frontend_count),
        "--mocker-processes",
        str(args.mocker_processes),
        "--output-dir",
        str(child_output_dir),
        "--model",
        args.model,
        "--requests",
        str(args.requests),
        "--warmup-requests",
        str(args.warmup_requests),
        "--concurrency",
        str(args.concurrency),
        "--isl",
        str(args.isl),
        "--osl",
        str(args.osl),
        "--event-plane",
        args.event_plane,
        "--etcd-endpoints",
        args.etcd_endpoints,
        "--nats-server",
        args.nats_server,
        "--valkey-authoritative-admission",
        "--valkey-admission-lease-ms",
        str(args.valkey_admission_lease_ms),
        "--valkey-gc-interval-ms",
        str(args.valkey_gc_interval_ms),
        "--valkey-gc-inspection-budget",
        str(args.valkey_gc_inspection_budget),
        "--aiperf-timeout-seconds",
        str(args.aiperf_timeout_seconds),
        "--aiperf-request-timeout-seconds",
        str(args.aiperf_request_timeout_seconds),
        "--tcp-request-timeout-seconds",
        str(args.tcp_request_timeout_seconds),
        "--ready-timeout",
        str(args.ready_timeout),
        "--replica-ready-timeout",
        str(args.replica_ready_timeout),
        "--settle-seconds",
        str(args.settle_seconds),
    ]
    if args.tokenizer:
        command.extend(("--tokenizer", args.tokenizer))
    if args.aiperf:
        command.extend(("--aiperf", str(args.aiperf.expanduser().resolve())))
    if args.aiperf_workers_max is not None:
        command.extend(("--aiperf-workers-max", str(args.aiperf_workers_max)))
    if args.record_processors is not None:
        command.extend(("--record-processors", str(args.record_processors)))
    for option, value in (
        ("--frontend-cpus", args.frontend_cpus),
        ("--mocker-cpus", args.mocker_cpus),
        ("--valkey-cpus", args.valkey_cpus),
        ("--aiperf-cpus", args.aiperf_cpus),
    ):
        if value is not None:
            command.extend((option, value))
    command.extend(args.harness_extra_arg)
    return command


def stop_process_group(process: subprocess.Popen[str]) -> None:
    """Bound cleanup if the user interrupts the scale driver mid-child."""

    if process.poll() is not None:
        return
    for signal_to_send, grace_seconds in (
        (signal.SIGINT, 20),
        (signal.SIGTERM, 10),
        (signal.SIGKILL, 5),
    ):
        try:
            os.killpg(process.pid, signal_to_send)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=grace_seconds)
            return
        except subprocess.TimeoutExpired:
            continue


def release_core_provenance_error(provenance: Any) -> str | None:
    """Reject authoritative samples that cannot prove a release core build."""

    if not isinstance(provenance, Mapping):
        return "child summary does not contain benchmark provenance"
    dynamo_core = provenance.get("dynamo_core")
    if not isinstance(dynamo_core, Mapping):
        return "child benchmark provenance does not contain dynamo._core metadata"
    build_profile = dynamo_core.get("rust_build_profile")
    if build_profile != "release":
        return (
            "child benchmark provenance reports dynamo._core Rust build profile "
            f"{build_profile!r}; authoritative scale samples require 'release'"
        )
    return None


def read_child_result(
    child_output_dir: Path, *, frontend_count: int, args: argparse.Namespace
) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse and validate the one result a child harness is allowed to emit."""

    summary_path = child_output_dir / "summary.json"
    try:
        child_summary = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"could not read child summary {summary_path}: {error}"]
    if not isinstance(child_summary, Mapping):
        return None, ["child summary is not a JSON object"]

    errors: list[str] = []
    child_provenance = child_summary.get("provenance")
    provenance_error = release_core_provenance_error(child_provenance)
    if provenance_error is not None:
        errors.append(provenance_error)
    runs = child_summary.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        return None, [f"expected exactly one child run, got {runs!r}"]
    result = runs[0]
    if not isinstance(result, Mapping):
        return None, ["child run result is not a JSON object"]

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(result.get("status") == "ok", f"child status is {result.get('status')!r}")
    require(result.get("arm") == "valkey_ha", "child did not run the valkey_ha arm")
    require(
        result.get("valkey_authoritative_admission") is True,
        "child did not enable authoritative Valkey admission",
    )
    require(
        result.get("comparison_profile") == "valkey_authoritative_immediate",
        "child did not use the authoritative immediate-routing profile",
    )
    require(
        result.get("valkey_gc_interval_ms") == args.valkey_gc_interval_ms,
        "child lifecycle-GC interval does not match the requested value",
    )
    require(
        result.get("valkey_gc_inspection_budget") == args.valkey_gc_inspection_budget,
        "child lifecycle-GC inspection budget does not match the requested value",
    )
    topology = result.get("topology")
    require(
        isinstance(topology, Mapping), "child result does not contain topology metadata"
    )
    if isinstance(topology, Mapping):
        require(
            topology.get("frontend_processes") == frontend_count,
            "child frontend count does not match requested count "
            f"({topology.get('frontend_processes')!r} != {frontend_count})",
        )
        require(
            topology.get("mocker_processes") == args.mocker_processes,
            "child mocker process count does not match requested count "
            f"({topology.get('mocker_processes')!r} != {args.mocker_processes})",
        )
        require(
            topology.get("logical_mocker_workers") == LOGICAL_MOCKER_WORKERS,
            "child did not use the fixed four-logical-worker topology",
        )
        require(
            topology.get("discovered_worker_identity_count") == LOGICAL_MOCKER_WORKERS,
            "child did not discover four unique logical worker identities",
        )
        expected_affinity = {
            "frontend": args.frontend_cpus,
            "mocker": args.mocker_cpus,
            "valkey": args.valkey_cpus,
            "aiperf": args.aiperf_cpus,
        }
        require(
            topology.get("configured_cpu_affinity") == expected_affinity,
            "child CPU affinity metadata does not match the requested role masks "
            f"({topology.get('configured_cpu_affinity')!r} != {expected_affinity!r})",
        )
        process_layout = topology.get("mocker_process_layout")
        require(
            isinstance(process_layout, list)
            and len(process_layout) == args.mocker_processes,
            "child mocker process layout is missing or has the wrong shard count",
        )
    require(
        result.get("request_plane") == "tcp",
        "child request plane is not TCP",
    )
    require(
        result.get("aiperf_expected_profiling_requests") == args.requests,
        "child aiperf request count does not match the requested workload",
    )

    valkey_final_state = result.get("valkey_final_state")
    final_primary_offset: int | None = None
    final_replica_offset: int | None = None
    final_replication_offset_delta: int | None = None
    final_replica_at_least_primary_snapshot: bool | None = None
    final_primary_replid: str | None = None
    final_replica_replid: str | None = None
    final_connected_replicas: int | None = None
    final_good_replicas: int | None = None
    final_replica_synchronized = False
    renew_commandstats: str | None = None
    renew_failed_calls: int | None = None
    require(
        isinstance(valkey_final_state, Mapping),
        "child result has no final Valkey HA state",
    )
    if isinstance(valkey_final_state, Mapping):
        primary_replication = valkey_final_state.get("primary_replication")
        replica_replication = valkey_final_state.get("replica_replication")
        primary_commandstats = valkey_final_state.get("primary_commandstats")
        replication_errors: list[str] = []

        def require_replication(condition: bool, message: str) -> None:
            if not condition:
                replication_errors.append(message)

        require_replication(
            isinstance(primary_replication, Mapping),
            "final Valkey state has no primary replication INFO fields",
        )
        require_replication(
            isinstance(replica_replication, Mapping),
            "final Valkey state has no replica replication INFO fields",
        )
        if isinstance(primary_replication, Mapping) and isinstance(
            replica_replication, Mapping
        ):
            final_connected_replicas = replication_offset(
                primary_replication, "connected_slaves", "connected_replicas"
            )
            final_good_replicas = replication_offset(
                primary_replication, "min_slaves_good_slaves"
            )
            final_primary_offset = replication_offset(
                primary_replication, "master_repl_offset"
            )
            final_replica_offset = replication_offset(
                replica_replication, "slave_repl_offset", "master_repl_offset"
            )
            if final_primary_offset is not None and final_replica_offset is not None:
                final_replication_offset_delta = (
                    final_replica_offset - final_primary_offset
                )
                final_replica_at_least_primary_snapshot = final_replica_offset >= (
                    final_primary_offset
                )
            primary_replid = primary_replication.get("master_replid")
            replica_replid = replica_replication.get("master_replid")
            final_primary_replid = (
                primary_replid if isinstance(primary_replid, str) else None
            )
            final_replica_replid = (
                replica_replid if isinstance(replica_replid, str) else None
            )
            has_online_replica = any(
                key.startswith("slave") and "state=online" in str(value)
                for key, value in primary_replication.items()
            )
            require_replication(
                primary_replication.get("role") == "master",
                "final Valkey primary role is not master",
            )
            require_replication(
                final_connected_replicas is not None and final_connected_replicas >= 1,
                "final Valkey primary has no connected replica",
            )
            require_replication(
                (final_good_replicas is not None and final_good_replicas >= 1)
                or has_online_replica,
                "final Valkey primary has no online/good replica",
            )
            require_replication(
                replica_replication.get("role") in {"slave", "replica"},
                "final Valkey replica role is not replica/slave",
            )
            require_replication(
                replica_replication.get("master_link_status") == "up",
                "final Valkey replica master link is not up",
            )
            require_replication(
                replica_replication.get("master_sync_in_progress") == "0",
                "final Valkey replica is still synchronizing",
            )
            require_replication(
                bool(final_primary_replid)
                and bool(final_replica_replid)
                and final_primary_replid == final_replica_replid,
                "final Valkey primary/replica replication IDs do not match",
            )
            # The harness obtains primary and replica INFO replies separately.
            # Detached releases may replicate between those reads, so exact
            # offsets are observational data, not a safe validity predicate.
        final_replica_synchronized = not replication_errors
        errors.extend(replication_errors)

        # The module reports this command only after a request lives long
        # enough to renew its lease. Its absence is valid; if present, any
        # failed renewal invalidates the sample even if requests happened to
        # complete through retry/release paths.
        if isinstance(primary_commandstats, Mapping):
            renew_value = next(
                (
                    value
                    for command, value in primary_commandstats.items()
                    if str(command).casefold() == "dynkv.renew"
                ),
                None,
            )
            if renew_value is not None:
                renew_commandstats = (
                    renew_value if isinstance(renew_value, str) else str(renew_value)
                )
                renew_fields = commandstats_fields(renew_value)
                if renew_fields is None:
                    errors.append(
                        "could not parse final DYNKV.RENEW commandstats "
                        f"{renew_commandstats!r}"
                    )
                else:
                    try:
                        renew_failed_calls = int(renew_fields["failed_calls"])
                    except (KeyError, ValueError):
                        errors.append(
                            "final DYNKV.RENEW commandstats has no valid failed_calls field"
                        )
                    else:
                        require(
                            renew_failed_calls == 0,
                            "final DYNKV.RENEW commandstats reports "
                            f"failed_calls={renew_failed_calls}",
                        )

    aiperf = result.get("aiperf_metrics")
    require(isinstance(aiperf, Mapping), "child result has no aiperf metrics")
    summary: Mapping[str, Any] = {}
    records: Mapping[str, Any] = {}
    if isinstance(aiperf, Mapping):
        raw_summary = aiperf.get("summary")
        raw_records = aiperf.get("records")
        if isinstance(raw_summary, Mapping):
            summary = raw_summary
        else:
            errors.append("child result has no finalized aiperf summary")
        if isinstance(raw_records, Mapping):
            records = raw_records
        else:
            errors.append("child result has no incremental aiperf records")

    completed_records = finite_number(records.get("completed_profiling_records"))
    require(
        completed_records == float(args.requests),
        "child completed profiling record count does not match request count "
        f"({completed_records!r} != {args.requests})",
    )
    errored_records = finite_number(records.get("errored_profiling_records"))
    require(
        errored_records == 0.0,
        f"child recorded {errored_records!r} errored profiling requests",
    )
    observed_errors = finite_number(result.get("aiperf_observed_error_request_count"))
    require(
        observed_errors == 0.0,
        f"child reported {observed_errors!r} observed aiperf errors",
    )
    input_sha256 = result.get("aiperf_input_sha256")
    require(
        isinstance(input_sha256, str)
        and len(input_sha256) == 64
        and all(character in "0123456789abcdef" for character in input_sha256),
        f"child aiperf input dataset SHA-256 is missing or invalid ({input_sha256!r})",
    )

    metrics = {
        "request_throughput_rps": nested_metric(summary, "request_throughput", "avg"),
        "ttft_ms_p50": nested_metric(summary, "time_to_first_token", "p50"),
        "ttft_ms_p95": nested_metric(summary, "time_to_first_token", "p95"),
        "itl_ms_p50": nested_metric(summary, "inter_token_latency", "p50"),
        "itl_ms_p95": nested_metric(summary, "inter_token_latency", "p95"),
        "request_latency_ms_p50": nested_metric(summary, "request_latency", "p50"),
        "request_latency_ms_p95": nested_metric(summary, "request_latency", "p95"),
        "isl_tokens_avg": nested_metric(summary, "input_sequence_length", "avg"),
        "osl_tokens_avg": nested_metric(summary, "output_sequence_length", "avg"),
        "output_token_throughput_tps": nested_metric(
            summary, "output_token_throughput", "avg"
        ),
    }
    for metric_name, metric_value in metrics.items():
        require(
            metric_value is not None and metric_value >= 0.0,
            f"child aiperf metric {metric_name} is missing or invalid ({metric_value!r})",
        )
    require(
        (metrics["request_throughput_rps"] or 0.0) > 0.0,
        "child request throughput is not positive",
    )

    normalized = {
        "child_summary_path": str(summary_path),
        "child_result_path": str(child_output_dir / "run-01-valkey_ha" / "result.json"),
        "child_provenance": (
            dict(child_provenance) if isinstance(child_provenance, Mapping) else None
        ),
        "child_status": result.get("status"),
        "child_topology": dict(topology) if isinstance(topology, Mapping) else None,
        "valkey_registered_ranks": result.get("valkey_registered_ranks"),
        "valkey_expected_registered_ranks": result.get(
            "valkey_expected_registered_ranks"
        ),
        "valkey_final_replica_synchronized": final_replica_synchronized,
        "valkey_final_connected_replicas": final_connected_replicas,
        "valkey_final_good_replicas": final_good_replicas,
        "valkey_final_primary_replid": final_primary_replid,
        "valkey_final_replica_replid": final_replica_replid,
        "valkey_final_primary_replication_offset": final_primary_offset,
        "valkey_final_replica_replication_offset": final_replica_offset,
        "valkey_final_replication_offset_delta": final_replication_offset_delta,
        "valkey_final_replica_at_least_primary_snapshot": final_replica_at_least_primary_snapshot,
        "valkey_final_renew_commandstats": renew_commandstats,
        "valkey_final_renew_failed_calls": renew_failed_calls,
        "completed_profiling_records": completed_records,
        "errored_profiling_records": errored_records,
        "observed_aiperf_errors": observed_errors,
        "aiperf_input_sha256": input_sha256,
        "metrics": metrics,
    }
    return normalized, errors
