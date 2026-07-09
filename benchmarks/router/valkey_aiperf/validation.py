# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from typing import Any

from .common import SUMMARY_METRICS
from .metrics import aggregate_metric_value, percentile
from .schedule import event_plane_for_arm
from .valkey import valkey_ha_validation_errors, valkey_singleton_validation_errors


def validate_arm_result(
    result: Mapping[str, Any], args: argparse.Namespace
) -> list[str]:
    """Return every reason an arm is unsafe to include in an A/B comparison."""

    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    arm = result.get("arm")
    require(
        arm in {"inprocess", "inprocess_immediate", "valkey_ha"},
        f"unknown benchmark arm {arm!r}",
    )
    require(result.get("request_plane") == "tcp", "request plane is not TCP")
    require(
        result.get("offered_load")
        == {
            "mode": "closed_loop",
            "concurrency": args.concurrency,
            "request_rate_rps": "inf",
        },
        "offered-load metadata does not describe infinite-rate closed-loop traffic",
    )
    require(
        result.get("event_plane") == event_plane_for_arm(args, str(arm)),
        "event plane does not match the arm's requested value",
    )
    topology = result.get("topology")
    require(isinstance(topology, Mapping), "topology metadata is missing")
    if isinstance(topology, Mapping):
        expected_ranks = args.logical_mocker_workers * args.mocker_data_parallel_size
        expected_affinity = {
            "frontend": args.frontend_cpus,
            "mocker": args.mocker_cpus,
            "valkey": args.valkey_cpus,
            "aiperf": args.aiperf_cpus,
        }
        require(
            topology.get("frontend_processes") == args.frontend_count,
            "frontend topology count does not match the request",
        )
        require(
            topology.get("mocker_processes") == args.mocker_processes,
            "mocker process topology count does not match the request",
        )
        require(
            topology.get("logical_mocker_workers") == args.logical_mocker_workers,
            "logical mock-worker topology count does not match the request",
        )
        require(
            topology.get("data_parallel_ranks_per_worker")
            == args.mocker_data_parallel_size,
            "topology DP rank count does not match the request",
        )
        require(
            topology.get("routing_ranks") == expected_ranks,
            "topology routing-rank count is incorrect",
        )
        require(
            topology.get("configured_cpu_affinity") == expected_affinity,
            "topology CPU-affinity metadata does not match the request",
        )
        process_layout = topology.get("mocker_process_layout")
        require(
            isinstance(process_layout, list)
            and len(process_layout) == args.mocker_processes,
            "mocker process layout is missing or incomplete",
        )
        worker_ids = topology.get("discovered_worker_ids")
        valid_worker_ids = isinstance(worker_ids, list) and all(
            isinstance(worker_id, int) for worker_id in worker_ids
        )
        require(
            valid_worker_ids
            and len(worker_ids) == args.logical_mocker_workers
            and len(set(worker_ids)) == args.logical_mocker_workers,
            "discovery did not yield the requested unique worker identities",
        )
        require(
            topology.get("discovered_worker_identity_count")
            == args.logical_mocker_workers,
            "discovered worker identity count is incorrect",
        )

    aiperf_result = result.get("aiperf")
    require(isinstance(aiperf_result, Mapping), "aiperf process result is missing")
    if isinstance(aiperf_result, Mapping):
        require(aiperf_result.get("timed_out") is False, "aiperf timed out")
        require(aiperf_result.get("returncode") == 0, "aiperf returned nonzero")

    metrics = result.get("aiperf_metrics")
    require(isinstance(metrics, Mapping), "aiperf metrics are missing")
    summary: Mapping[str, Any] | None = None
    records: Mapping[str, Any] | None = None
    if isinstance(metrics, Mapping):
        raw_summary = metrics.get("summary")
        raw_records = metrics.get("records")
        if isinstance(raw_summary, Mapping) and "parse_error" not in raw_summary:
            summary = raw_summary
        else:
            errors.append("finalized aiperf summary is missing or invalid")
        if isinstance(raw_records, Mapping):
            records = raw_records
        else:
            errors.append("incremental aiperf records are missing")
    if records is not None:
        for field in (
            "cancelled_profiling_records",
            "errored_profiling_records",
            "malformed_records",
        ):
            require(records.get(field) == 0, f"aiperf {field} is not zero")
        require(
            records.get("completed_profiling_records") == args.requests,
            "completed aiperf record count does not match the request count",
        )
    if summary is not None:
        request_count = aggregate_metric_value(summary.get("request_count"))
        require(
            request_count == float(args.requests),
            "finalized aiperf request count does not match the request count",
        )
        error_count = aggregate_metric_value(summary.get("error_request_count"))
        require(
            error_count in {None, 0.0},
            f"finalized aiperf error count is {error_count!r}",
        )
    throughput = result_metric_value(result, "request_throughput_rps")
    require(
        throughput is not None and math.isfinite(throughput) and throughput > 0.0,
        f"request throughput is invalid: {throughput!r}",
    )
    require(
        isinstance(result.get("aiperf_input_sha256"), str),
        "aiperf input dataset SHA-256 is missing",
    )

    if arm == "valkey_ha":
        expected_ranks = args.logical_mocker_workers * args.mocker_data_parallel_size
        kill_primary = getattr(args, "kill_valkey_primary", False)
        require(
            result.get("valkey_gc_interval_ms") == args.valkey_gc_interval_ms,
            "Valkey lifecycle-GC interval metadata does not match the request",
        )
        require(
            result.get("valkey_gc_inspection_budget")
            == args.valkey_gc_inspection_budget,
            "Valkey lifecycle-GC inspection-budget metadata does not match the request",
        )
        require(
            result.get("valkey_expected_registered_ranks") == expected_ranks,
            "expected Valkey registered-rank metadata is incorrect",
        )
        require(
            result.get("valkey_registered_ranks") == expected_ranks,
            "Valkey did not register exactly the expected worker ranks",
        )
        client_pressure = result.get("valkey_client_pressure")
        client_pressure_ports = (
            client_pressure.get("ports")
            if isinstance(client_pressure, Mapping)
            else None
        )
        require(
            isinstance(client_pressure, Mapping)
            and "error" not in client_pressure
            and isinstance(client_pressure.get("successful_reads"), int)
            and client_pressure["successful_reads"] > 0,
            f"Valkey client-pressure sampling failed: {client_pressure!r}",
        )
        require(
            isinstance(client_pressure_ports, Mapping)
            and len(client_pressure_ports) == 2
            and all(
                isinstance(port_pressure, Mapping)
                and isinstance(port_pressure.get("peak_connected_clients"), int)
                and port_pressure["peak_connected_clients"] > 0
                and isinstance(port_pressure.get("maxclients"), int)
                and port_pressure["maxclients"] > 0
                for port_pressure in client_pressure_ports.values()
            ),
            f"Valkey client-pressure port coverage is invalid: {client_pressure_ports!r}",
        )
        if not kill_primary:
            require(
                isinstance(client_pressure, Mapping)
                and client_pressure.get("read_errors") == 0,
                f"Valkey client-pressure sampling had read errors: {client_pressure!r}",
            )
        if isinstance(topology, Mapping):
            require(
                topology.get("valkey_client_endpoint_count")
                == (2 if kill_primary else 1),
                (
                    "Valkey failover topology does not expose both data endpoints"
                    if kill_primary
                    else "Valkey topology does not use one stable primary endpoint"
                ),
            )
            require(
                topology.get("valkey_data_node_count") == 2,
                "Valkey topology does not contain two data nodes",
            )
            require(
                topology.get("valkey_required_replica_acks") == 1,
                "Valkey topology does not require one replica acknowledgement",
            )
            if kill_primary:
                require(
                    topology.get("valkey_sentinel_count") == 3,
                    "Valkey failover topology does not contain three Sentinels",
                )
        admission_stats = result.get("valkey_final_admission_stats")
        if kill_primary:
            fault = (
                aiperf_result.get("fault_injection", {})
                if isinstance(aiperf_result, Mapping)
                else {}
            )
            require(
                isinstance(fault, Mapping) and fault.get("status") == "promoted",
                f"Valkey primary fault injection did not promote: {fault!r}",
            )
            require(
                admission_stats == {"promoted": 0},
                f"Valkey admission reservations leaked after failover: {admission_stats!r}",
            )
            errors.extend(
                valkey_singleton_validation_errors(
                    result.get("valkey_final_state"),
                    expected_ranks=expected_ranks,
                    authoritative_admission=(
                        result.get("valkey_authoritative_admission") is True
                    ),
                )
            )
        else:
            require(
                admission_stats == {"primary": 0, "replica": 0},
                f"Valkey admission reservations leaked: {admission_stats!r}",
            )
            errors.extend(
                valkey_ha_validation_errors(
                    result.get("valkey_final_state"),
                    authoritative_admission=(
                        result.get("valkey_authoritative_admission") is True
                    ),
                )
            )
    return errors


def numeric_mapping_value(value: Any, key: str) -> float | None:
    """Return one numeric statistic from an aiperf metric mapping."""

    if not isinstance(value, Mapping):
        return None
    raw_value = value.get(key)
    return float(raw_value) if isinstance(raw_value, int | float) else None


def result_metric_value(result: Mapping[str, Any], metric_name: str) -> float | None:
    """Extract one normalized metric from an arm result.

    Prefer aiperf's finalized aggregate. When its finalizer is interrupted,
    fall back to the incrementally exported records so the incomplete arm is
    still inspectable, while the status checks separately prevent it from being
    treated as a valid result.
    """

    selector = SUMMARY_METRICS[metric_name]
    summary_tag, stat, records_tag = selector
    metrics = result.get("aiperf_metrics")
    if not isinstance(metrics, Mapping):
        return None

    summary = metrics.get("summary")
    if isinstance(summary, Mapping):
        value = numeric_mapping_value(summary.get(summary_tag), stat)
        if value is not None:
            return value

    records = metrics.get("records")
    if not isinstance(records, Mapping) or records_tag is None:
        return None
    record_value = records.get(records_tag)
    if isinstance(record_value, int | float):
        return float(record_value)
    return numeric_mapping_value(record_value, stat)


def arm_metric_summary(
    results: list[dict[str, Any]], arm: str, *, planned_runs: int | None = None
) -> dict[str, Any]:
    """Return median metrics for successful repetitions of one arm."""

    arm_results = [result for result in results if result.get("arm") == arm]
    successful = [result for result in arm_results if result.get("status") == "ok"]
    metric_summary: dict[str, float | None] = {}
    metric_samples: dict[str, list[float]] = {}
    for metric_name in SUMMARY_METRICS:
        values = [
            value
            for result in successful
            if (value := result_metric_value(result, metric_name)) is not None
        ]
        metric_samples[metric_name] = values
        metric_summary[metric_name] = percentile(values, 0.50)
    return {
        "planned_runs": len(arm_results) if planned_runs is None else planned_runs,
        "started_runs": len(arm_results),
        "successful_runs": len(successful),
        "statuses": [result.get("status") for result in arm_results],
        "median_metrics": metric_summary,
        "metric_samples": metric_samples,
    }


def compare_arm_medians(
    arm_summaries: Mapping[str, Mapping[str, Any]],
    *,
    baseline_arm: str,
    candidate_arm: str,
    policy_matched: bool,
    note: str,
) -> dict[str, Any] | None:
    """Compute signed deltas where positive latency is worse and positive RPS is better."""

    baseline = arm_summaries.get(baseline_arm)
    candidate = arm_summaries.get(candidate_arm)
    if baseline is None or candidate is None:
        return None
    if not baseline.get("successful_runs") or not candidate.get("successful_runs"):
        return None
    baseline_metrics = baseline.get("median_metrics")
    candidate_metrics = candidate.get("median_metrics")
    if not isinstance(baseline_metrics, Mapping) or not isinstance(
        candidate_metrics, Mapping
    ):
        return None

    metrics: dict[str, dict[str, float | None]] = {}
    for metric_name in SUMMARY_METRICS:
        baseline_value = baseline_metrics.get(metric_name)
        candidate_value = candidate_metrics.get(metric_name)
        if not isinstance(baseline_value, int | float) or not isinstance(
            candidate_value, int | float
        ):
            continue
        baseline_float = float(baseline_value)
        candidate_float = float(candidate_value)
        metrics[metric_name] = {
            "baseline_median": baseline_float,
            "candidate_median": candidate_float,
            "absolute_delta": candidate_float - baseline_float,
            "relative_percent": (
                (candidate_float - baseline_float) / baseline_float * 100.0
                if baseline_float != 0
                else None
            ),
        }
    return {
        "baseline_arm": baseline_arm,
        "candidate_arm": candidate_arm,
        "policy_matched": policy_matched,
        "note": note,
        "metrics": metrics,
    }


def campaign_validation_errors(
    results: list[dict[str, Any]], planned_schedule: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    if len(results) != len(planned_schedule):
        errors.append(
            f"started {len(results)} of {len(planned_schedule)} planned arm samples"
        )
    for result, planned in zip(results, planned_schedule):
        for field in ("sample_index", "run", "arm"):
            if result.get(field) != planned.get(field):
                errors.append(
                    f"sample {planned.get('sample_index')} {field} mismatch: "
                    f"{result.get(field)!r} != {planned.get(field)!r}"
                )
        if result.get("status") != "ok":
            errors.append(
                f"sample {planned.get('sample_index')} arm {planned.get('arm')} "
                f"status is {result.get('status')!r}"
            )
    input_hashes = [
        result.get("aiperf_input_sha256")
        for result in results
        if isinstance(result.get("aiperf_input_sha256"), str)
    ]
    if len(input_hashes) != len(planned_schedule):
        errors.append("not every planned arm has an input dataset SHA-256")
    if len(set(input_hashes)) > 1:
        errors.append("aiperf input dataset SHA-256 differs across arms")
    return errors


def summarize_results(
    results: list[dict[str, Any]], planned_schedule: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build comparisons only when the complete campaign is valid."""

    arm_order = list(
        dict.fromkeys(str(sample.get("arm")) for sample in planned_schedule)
    )
    planned_counts = {
        arm: sum(sample.get("arm") == arm for sample in planned_schedule)
        for arm in arm_order
    }
    arm_summaries = {
        arm: arm_metric_summary(results, arm, planned_runs=planned_counts[arm])
        for arm in arm_order
    }
    validation_errors = campaign_validation_errors(results, planned_schedule)
    campaign_valid = not validation_errors
    comparisons: list[dict[str, Any]] = []
    valkey_authoritative = any(
        all(
            (
                result.get("arm") == "valkey_ha",
                result.get("valkey_authoritative_admission") is True,
            )
        )
        for result in results
    )
    if campaign_valid:
        normal_comparison = compare_arm_medians(
            arm_summaries,
            baseline_arm="inprocess",
            candidate_arm="valkey_ha",
            policy_matched=not valkey_authoritative,
            note=(
                "Both arms use synchronized local admission and replica-sync policy; "
                "this is not the literal bare-CLI default, where replica sync is off."
                if not valkey_authoritative
                else (
                    "The Valkey arm uses module-owned immediate admission; compare it "
                    "primarily with inprocess_immediate below, not the synchronized "
                    "multi-frontend baseline. The bare-CLI default has replica sync off."
                )
            ),
        )
        if normal_comparison is not None:
            comparisons.append(normal_comparison)
        immediate_comparison = compare_arm_medians(
            arm_summaries,
            baseline_arm="inprocess_immediate",
            candidate_arm="valkey_ha",
            policy_matched=True,
            note=(
                "Matched immediate-routing policy: no router queue, replica sync, "
                "frontend prefill/output accounting, lower-tier credits, overlap decay, "
                "or temperature."
            ),
        )
        if immediate_comparison is not None:
            comparisons.append(immediate_comparison)
        policy_cost = compare_arm_medians(
            arm_summaries,
            baseline_arm="inprocess",
            candidate_arm="inprocess_immediate",
            policy_matched=False,
            note=(
                "Local-router policy change only; use this to separate admission-policy "
                "impact from the Valkey-module comparison."
            ),
        )
        if policy_cost is not None:
            comparisons.append(policy_cost)

    input_hashes = sorted(
        {
            str(result["aiperf_input_sha256"])
            for result in results
            if isinstance(result.get("aiperf_input_sha256"), str)
        }
    )

    return {
        "valid": campaign_valid,
        "validation_errors": validation_errors,
        "planned_samples": len(planned_schedule),
        "started_samples": len(results),
        "valid_samples": sum(result.get("status") == "ok" for result in results),
        "input_dataset_sha256": input_hashes[0] if len(input_hashes) == 1 else None,
        "input_dataset_consistent": (
            len(input_hashes) == 1 and len(results) == len(planned_schedule)
        ),
        "methodology_caveats": [
            "aiperf is closed-loop: this is a host-level throughput/latency result, not an open-loop server capacity limit.",
            "Processes share the runner CPU set unless the per-role CPU affinity options or external isolation pin them; inspect each run's topology metadata.",
            "Logical mock workers retain the same worker-scoped DP ranks regardless of how many independent dynamo.mocker OS processes host them.",
            "The inprocess arm enables replica sync as a synchronized multi-frontend baseline; it is not the literal bare-CLI default.",
            "Medians remain diagnostic during an incomplete campaign, but comparisons are suppressed unless every planned sample is valid and uses one identical input dataset.",
        ],
        "arms": arm_summaries,
        "comparisons": comparisons,
        "comparisons_suppressed": not campaign_valid,
    }
