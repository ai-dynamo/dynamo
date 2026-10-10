# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify a native Planner autonomous scale-from-zero batch run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

DISPATCHED = "llm_d_async_async_dispatched_requests_total"
SUCCESSFUL = "llm_d_async_async_successful_requests_total"
BACKLOG = "llm_d_async_async_broker_backlog"
INFLIGHT = "llm_d_async_async_inflight_requests"
QUEUE_DEPTH = "llm_d_async_async_queue_depth"
DRAIN_GAUGE = "llm_d_async_async_drain_limit_rps"
PLANNER_DECISION_PREFIX = "Batch scheduling decision:"
LEASE_PTTL_TOLERANCE_SECONDS = 5.0


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _read_data_lines(path: Path) -> list[str]:
    return [
        line
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    ]


def _read_scalar(path: Path) -> str:
    lines = _read_data_lines(path)
    if len(lines) != 1:
        raise ValueError(f"expected one data line in {path}, got {len(lines)}")
    return lines[0]


def _lease_cap(state: dict[str, Any]) -> float:
    fields = _lease_fields(state)
    return float(fields[2])


def _lease_fields(state: dict[str, Any]) -> list[str]:
    fields = next(csv.reader([state["lease_csv"]]))
    if len(fields) != 5:
        raise ValueError(f"expected five lease fields, got {fields!r}")
    return fields


def _lease_decision_id(state: dict[str, Any]) -> str:
    return _lease_fields(state)[4]


def _metric_value(payload: str, name: str, *, missing_is_zero: bool = False) -> float:
    for line in payload.splitlines():
        if not line.startswith((f"{name} ", f"{name}{{")):
            continue
        if "{" in line and 'pool_name="dynamo-batch"' not in line:
            continue
        return float(line.rsplit(maxsplit=1)[1])
    if missing_is_zero:
        # Prometheus CounterVec label series do not exist until their first
        # increment. A fresh, authoritatively idle Async pod therefore has a
        # semantic pre-run counter value of zero even though the series is
        # absent. Post-run reads remain strict so a missing result cannot pass.
        return 0.0
    raise ValueError(f"metric {name!r} not found")


def _parse_rfc3339(value: str) -> datetime:
    match = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z$", value)
    if match is None:
        raise ValueError(f"invalid RFC3339 timestamp: {value!r}")
    fraction = (match.group(2) or "")[:6].ljust(6, "0")
    return datetime.fromisoformat(f"{match.group(1)}.{fraction}+00:00")


def _normalize_observer_time(value: str) -> str:
    # BSD date preserves GNU's unsupported %3N token literally. The raw stream
    # remains untouched; derived output safely falls back to second precision.
    return re.sub(r"\.3NZ$", "Z", value)


def _planner_decisions(log: str) -> list[dict[str, Any]]:
    """Parse structured fields from native Planner decision log records.

    Historical evidence predates ``idle_replica_target`` and remains valid for
    the legacy verifier mode. New scale-to-idle evidence opts into an expected
    target and therefore requires that field to be present on the idle record.
    """

    decisions: list[dict[str, Any]] = []
    offset = 0
    for line in log.splitlines(keepends=True):
        marker = line.find(PLANNER_DECISION_PREFIX)
        if marker < 0:
            offset += len(line)
            continue
        fields = {
            key: value
            for token in line[marker + len(PLANNER_DECISION_PREFIX) :].split()
            if "=" in token
            for key, value in [token.split("=", maxsplit=1)]
        }
        try:
            cap = float(fields["max_admission_rps"])
        except (KeyError, ValueError):
            offset += len(line)
            continue
        timestamp_fields = line.split(maxsplit=1)
        timestamp_text = timestamp_fields[0] if timestamp_fields else ""
        try:
            observed_at = _parse_rfc3339(timestamp_text)
        except ValueError:
            # The timestamp is reporting-only. Decision identity and byte
            # order remain authoritative when the record is untimestamped.
            observed_at = None
        idle_text = fields.get("idle_replica_target")
        try:
            idle_target = None if idle_text in (None, "None") else int(idle_text)
        except ValueError:
            offset += len(line)
            continue
        decisions.append(
            {
                "offset": offset + marker,
                "replica_floor": fields.get("replica_floor"),
                "max_admission_rps": cap,
                "idle_replica_target": idle_target,
                "observed_at": observed_at,
                "decision_id": fields.get("decision_id"),
            }
        )
        offset += len(line)
    return decisions


def _dispatch_counter_interval(
    run_dir: Path, baseline: float
) -> tuple[str | None, str | None, float | None, bool]:
    """Bound the first observed increase and reject observed counter resets.

    A scrape timestamp says only when an increase was observed. The last
    unchanged scrape is therefore the lower bound of the possible dispatch
    interval. A counter reset breaks continuity with the pre-run baseline and
    makes that interval unsuitable for ordering evidence.
    """
    last_unchanged_at: str | None = None
    previous_value = baseline
    counter_continuous = True
    for prom_path in sorted((run_dir / "metrics" / "async").glob("*.prom")):
        value = _metric_value(prom_path.read_text(encoding="utf-8"), DISPATCHED)
        metadata_path = prom_path.with_suffix(".json")
        observed_at = _normalize_observer_time(_read_json(metadata_path)["observed_at"])
        if value < previous_value:
            counter_continuous = False
        if value == baseline and counter_continuous:
            last_unchanged_at = observed_at
        if value <= baseline:
            previous_value = value
            continue
        return last_unchanged_at, observed_at, value, counter_continuous
    return last_unchanged_at, None, None, counter_continuous


def _first_index(items: list[Any], predicate: Callable[[Any], bool]) -> int | None:
    return next((index for index, item in enumerate(items) if predicate(item)), None)


def _state_time(states: list[dict[str, Any]], index: int | None) -> str | None:
    if index is None:
        return None
    return _normalize_observer_time(states[index]["observed_at"])


def _state_started_at(state: dict[str, Any]) -> datetime:
    return _parse_rfc3339(
        _normalize_observer_time(state.get("sample_started_at", state["observed_at"]))
    )


def _state_observed_at(state: dict[str, Any]) -> datetime:
    return _parse_rfc3339(_normalize_observer_time(state["observed_at"]))


def _is_fully_active_state(state: dict[str, Any]) -> bool:
    return (
        _lease_cap(state) > 0
        and state["adapter_spec"] >= 1
        and state.get("adapter_status", state["adapter_spec"]) >= 1
        and state["ready_worker_pods"] >= 1
        and state["worker_ready_replicas"] >= 1
        and state["dgd_ready"] == "True"
    )


def _is_complete_idle_state(state: dict[str, Any], expected_replicas: int) -> bool:
    required_counts = (
        "adapter_spec",
        "adapter_status",
        "worker_pod_count",
        "nonterminating_worker_pods",
        "ready_worker_pods",
        "worker_ready_replicas",
    )
    return (
        all(state.get(field) == expected_replicas for field in required_counts)
        and _lease_cap(state) == 0
        and (expected_replicas == 0 or state.get("dgd_ready") == "True")
    )


def _idle_stability_suffix(
    states: list[dict[str, Any]],
    *,
    expected_replicas: int,
    evidence_t1: datetime | None,
    required_seconds: float,
) -> tuple[bool, int | None, float | None]:
    """Validate a complete, fresh terminal idle suffix on the observer clock."""

    if required_seconds <= 0:
        return True, None, None
    if not states or evidence_t1 is None:
        return False, None, None

    suffix_start = len(states)
    while suffix_start > 0 and _is_complete_idle_state(
        states[suffix_start - 1], expected_replicas
    ):
        suffix_start -= 1
    suffix = states[suffix_start:]
    if not suffix:
        return False, None, None

    starts = [_state_started_at(state) for state in suffix]
    ends = [_state_observed_at(state) for state in suffix]
    spans_valid = all(
        0 <= (end - start).total_seconds() <= 5
        for start, end in zip(starts, ends, strict=True)
    )
    gaps_valid = all(
        0 <= (right_start - left_end).total_seconds() <= 5
        for left_end, right_start in zip(ends[:-1], starts[1:], strict=True)
    )
    freshness_seconds = (evidence_t1 - ends[-1]).total_seconds()
    duration_seconds = (ends[-1] - starts[0]).total_seconds()
    valid = (
        spans_valid
        and gaps_valid
        and 0 <= freshness_seconds <= 5
        and duration_seconds >= required_seconds
    )
    return valid, suffix_start, duration_seconds


def _completed_observation_bounds(
    progress: list[dict[str, Any]],
) -> tuple[datetime | None, datetime | None]:
    completed = [
        observation
        for observation in progress
        if str(observation.get("status", "")).strip().lower() == "completed"
    ]
    if not completed:
        return None, None
    return (
        _parse_rfc3339(str(completed[0]["observed_at"])),
        _parse_rfc3339(str(completed[-1]["observed_at"])),
    )


def verify(
    run_dir: Path,
    evidence_dir: Path,
    *,
    worker_component: str = "worker",
    adapter_name: str = "qwen3-0-6b-batch-worker",
    expected_idle_replicas: int | None = None,
    expected_lease_duration_seconds: float,
    expected_idle_stability_seconds: float = 0,
) -> dict[str, Any]:
    if expected_idle_replicas is not None and expected_idle_replicas < 0:
        raise ValueError("expected_idle_replicas must be non-negative")
    if (
        isinstance(expected_lease_duration_seconds, bool)
        or not math.isfinite(expected_lease_duration_seconds)
        or expected_lease_duration_seconds <= 0
    ):
        raise ValueError("expected_lease_duration_seconds must be finite and positive")
    if (
        isinstance(expected_idle_stability_seconds, bool)
        or not math.isfinite(expected_idle_stability_seconds)
        or expected_idle_stability_seconds < 0
    ):
        raise ValueError(
            "expected_idle_stability_seconds must be finite and non-negative"
        )
    if expected_idle_stability_seconds > 0 and expected_idle_replicas is None:
        raise ValueError(
            "expected_idle_replicas is required when idle stability is requested"
        )
    maximum_expected_lease_pttl_ms = math.ceil(
        (expected_lease_duration_seconds + LEASE_PTTL_TOLERANCE_SECONDS) * 1_000
    )

    states = _read_jsonl(evidence_dir / "state.jsonl")
    caps = [_lease_cap(state) for state in states]
    scale_index = _first_index(states, lambda state: state["adapter_spec"] == 1)
    ready_index = _first_index(states, lambda state: state["ready_worker_pods"] >= 1)
    positive_index = _first_index(caps, lambda cap: cap > 0)
    terminal_zero_index = (
        next(
            (
                index
                for index in range(positive_index + 1, len(caps))
                if caps[index] == 0
            ),
            None,
        )
        if positive_index is not None
        else None
    )

    terminal = _read_json(run_dir / "terminal-batch.json")
    validation = _read_json(run_dir / "result-validation.json")
    adapter_before = _read_json(evidence_dir / "dgdsa.before.json")
    adapter_after = _read_json(evidence_dir / "dgdsa.after.json")
    watch_events = _read_jsonl(evidence_dir / "dgdsa.watch.jsonstream")
    planner_log = (evidence_dir / "planner.log").read_text(encoding="utf-8")
    planner_decisions = _planner_decisions(planner_log)
    gateway_completed_first_at: datetime | None = None
    gateway_completed_last_at: datetime | None = None
    if expected_idle_replicas is not None:
        progress_path = run_dir / "progress.jsonl"
        progress = _read_jsonl(progress_path) if progress_path.exists() else []
        (
            gateway_completed_first_at,
            gateway_completed_last_at,
        ) = _completed_observation_bounds(progress)

    redis_after = _read_data_lines(evidence_dir / "redis.after.txt")
    redis_pttl_ms = int(_read_scalar(evidence_dir / "redis.after.pttl-ms.txt"))

    metrics_before = (evidence_dir / "async-metrics.before.txt").read_text(
        encoding="utf-8"
    )
    metrics_after = (evidence_dir / "async-metrics.after.txt").read_text(
        encoding="utf-8"
    )
    dispatched_before = _metric_value(metrics_before, DISPATCHED, missing_is_zero=True)
    dispatched_after = _metric_value(metrics_after, DISPATCHED)
    successful_before = _metric_value(metrics_before, SUCCESSFUL, missing_is_zero=True)
    successful_after = _metric_value(metrics_after, SUCCESSFUL)
    (
        last_unchanged_at,
        first_increase_at,
        first_increase_value,
        dispatch_counter_continuous,
    ) = _dispatch_counter_interval(run_dir, dispatched_before)
    positive_lease_at = _state_time(states, positive_index)

    scale_up_log_needles = [
        "replica_floor=1 max_admission_rps=0.0",
        f"Updating decode component {worker_component} from 0 to desired replica count 1",
        f"Scaled DGDSA {adapter_name} to 1 replicas",
        "replica_floor=1 max_admission_rps=5.0",
    ]
    scale_up_log_positions: list[int] = []
    log_cursor = 0
    for needle in scale_up_log_needles:
        position = planner_log.find(needle, log_cursor)
        scale_up_log_positions.append(position)
        if position >= 0:
            log_cursor = position + len(needle)

    configured_idle_decisions = [
        decision
        for decision in planner_decisions
        if expected_idle_replicas is not None
        and decision["offset"] > scale_up_log_positions[-1]
        and decision["replica_floor"] == "None"
        and decision["max_admission_rps"] == 0.0
        and decision["idle_replica_target"] == expected_idle_replicas
    ]
    terminal_zero_decision_id = (
        _lease_decision_id(states[terminal_zero_index])
        if terminal_zero_index is not None
        else None
    )
    # Planner/CRI timestamps may come from a different host clock. Link the
    # authoritative first zero lease to its Planner record by decision ID and
    # use byte offsets only for sequencing within the Planner log.
    idle_decision = next(
        (
            decision
            for decision in configured_idle_decisions
            if decision["decision_id"] == terminal_zero_decision_id
            and terminal_zero_decision_id not in (None, "")
        ),
        None,
    )

    capture_valid_path = evidence_dir / "capture-valid.txt"
    capture_integrity_valid = (
        capture_valid_path.exists() and _read_scalar(capture_valid_path) == "1"
    )
    evidence_t1_path = evidence_dir / "T1.txt"
    evidence_t1 = (
        _parse_rfc3339(_read_scalar(evidence_t1_path))
        if evidence_t1_path.exists()
        else None
    )

    assertions = {
        "harness_exit_zero": _read_data_lines(run_dir / "exit_code.txt") == ["0"],
        "gateway_terminal_100_100_0": terminal.get("status") == "completed"
        and terminal.get("request_counts")
        == {"completed": 100, "failed": 0, "total": 100},
        "downloaded_results_valid": validation.get("valid") is True
        and validation.get("downloaded_output_lines") == 100
        and validation.get("unique_custom_ids") == 100,
        "adapter_started_at_zero": bool(states)
        and adapter_before["spec"]["replicas"] == 0
        and adapter_before["status"]["replicas"] == 0
        and states[0]["adapter_spec"] == 0,
        "watch_observed_zero_to_one": bool(watch_events)
        and watch_events[0]["object"]["spec"]["replicas"] == 0
        and any(event["object"]["spec"]["replicas"] == 1 for event in watch_events),
        "planner_logged_authoritative_scale": all(
            position >= 0 for position in scale_up_log_positions[:3]
        ),
        "scale_preceded_worker_readiness": scale_index is not None
        and ready_index is not None
        and scale_index <= ready_index,
        "lease_remained_closed_until_ready": positive_index is not None
        and ready_index is not None
        and positive_index >= ready_index
        and all(
            cap <= 0
            or (
                states[index]["ready_worker_pods"] >= 1
                and states[index]["worker_ready_replicas"] >= 1
                and states[index]["dgd_ready"] == "True"
            )
            for index, cap in enumerate(caps)
        ),
        "dispatch_started_after_positive_lease": first_increase_at is not None
        and last_unchanged_at is not None
        and positive_lease_at is not None
        and dispatch_counter_continuous
        and _parse_rfc3339(last_unchanged_at) >= _parse_rfc3339(positive_lease_at),
        "terminal_zero_observed_after_positive": terminal_zero_index is not None
        and positive_index is not None
        and terminal_zero_index > positive_index
        and caps[-1] == 0,
        "authoritative_terminal_lease_zero_and_fresh": len(redis_after) >= 5
        and redis_after[0] == "llm-d.ai/v1alpha1"
        and redis_after[1] == "dynamo-batch"
        and float(redis_after[2]) == 0
        and 0 < redis_pttl_ms <= maximum_expected_lease_pttl_ms,
        "exactly_100_dispatches_and_successes": dispatched_after - dispatched_before
        == 100
        and successful_after - successful_before == 100,
        "async_terminal_queues_empty": _metric_value(metrics_after, BACKLOG) == 0
        and _metric_value(metrics_after, INFLIGHT) == 0
        and _metric_value(metrics_after, QUEUE_DEPTH) == 0,
    }
    # Legacy captures predate this marker, but a marker that is present must
    # never be ignored. Strict idle-target verification additionally requires
    # the marker to exist so interrupted evidence collection cannot pass.
    if capture_valid_path.exists() or expected_idle_replicas is not None:
        assertions["capture_integrity_valid"] = capture_integrity_valid

    if expected_idle_replicas is None:
        terminal_log_position = planner_log.find(
            "replica_floor=0 max_admission_rps=0.0", log_cursor
        )
        assertions.update(
            {
                "adapter_finished_at_one": bool(states)
                and adapter_after["spec"]["replicas"] == 1
                and adapter_after["status"]["replicas"] == 1
                and states[-1]["adapter_spec"] == 1,
                "planner_policy_log_order": all(
                    position >= 0
                    for position in [*scale_up_log_positions, terminal_log_position]
                ),
            }
        )
    else:
        one_watch_index = _first_index(
            watch_events,
            lambda event: event["object"]["spec"]["replicas"] == 1,
        )
        idle_decision_position = (
            idle_decision["offset"] if idle_decision is not None else -1
        )
        if expected_idle_replicas == 1:
            # A warm-one target is a no-op. The first zero-lease sample must
            # itself show that the active worker stayed at the target.
            idle_target_state_index = (
                terminal_zero_index
                if terminal_zero_index is not None
                and states[terminal_zero_index]["adapter_spec"] == 1
                else None
            )
        else:
            # Ignore the pre-job baseline. Once admission became positive, the
            # first target observation is authoritative and cannot be masked by
            # a later recovery/renewal.
            idle_target_state_index = (
                next(
                    (
                        index
                        for index in range(positive_index + 1, len(states))
                        if states[index]["adapter_spec"] == expected_idle_replicas
                    ),
                    None,
                )
                if positive_index is not None
                else None
            )

        terminal_zero_after_completion = (
            terminal_zero_index is not None
            and gateway_completed_last_at is not None
            and _state_started_at(states[terminal_zero_index])
            > gateway_completed_last_at
        )
        adapter_target_after_completion = (
            idle_target_state_index is not None
            and gateway_completed_last_at is not None
            and _state_started_at(states[idle_target_state_index])
            > gateway_completed_last_at
        )
        barrier_end_candidates = [
            index
            for index in (terminal_zero_index, idle_target_state_index)
            if index is not None
        ]
        first_terminal_action_index = (
            min(barrier_end_candidates) if barrier_end_candidates else None
        )
        post_completion_active_index = (
            next(
                (
                    index
                    for index, state in enumerate(states)
                    if gateway_completed_last_at is not None
                    and first_terminal_action_index is not None
                    and index < first_terminal_action_index
                    and _state_started_at(state) > gateway_completed_last_at
                    and _is_fully_active_state(state)
                ),
                None,
            )
            if gateway_completed_last_at is not None
            else None
        )
        adapter_target_after_completion_and_decision = (
            adapter_target_after_completion
            and idle_decision is not None
            and terminal_zero_after_completion
        )
        (
            stability_valid,
            stability_suffix_index,
            stability_observed_seconds,
        ) = _idle_stability_suffix(
            states,
            expected_replicas=expected_idle_replicas,
            evidence_t1=evidence_t1,
            required_seconds=expected_idle_stability_seconds,
        )
        assertions.update(
            {
                "adapter_finished_at_expected_idle_target": bool(states)
                and adapter_after["spec"]["replicas"] == expected_idle_replicas
                and adapter_after["status"]["replicas"] == expected_idle_replicas
                and states[-1]["adapter_spec"] == expected_idle_replicas
                and states[-1]["ready_worker_pods"] == expected_idle_replicas
                and states[-1]["worker_ready_replicas"] == expected_idle_replicas
                and (
                    expected_idle_replicas == 0 or states[-1].get("dgd_ready") == "True"
                ),
                "gateway_completed_observation_present": gateway_completed_first_at
                is not None
                and gateway_completed_last_at is not None,
                "post_completion_active_barrier_observed": (
                    post_completion_active_index is not None
                ),
                "terminal_zero_observed_after_gateway_completion": (
                    terminal_zero_after_completion
                ),
                "first_idle_target_observed_after_gateway_completion": (
                    adapter_target_after_completion
                ),
                "planner_logged_configured_idle_target": idle_decision is not None,
                "adapter_target_observed_after_completion_and_idle_decision": (
                    adapter_target_after_completion_and_decision
                ),
                "terminal_redis_decision_matches_post_completion_idle": len(redis_after)
                >= 5
                and idle_decision is not None
                and any(
                    decision["decision_id"] == redis_after[4]
                    and decision["decision_id"] not in (None, "")
                    and decision["offset"] >= idle_decision_position
                    for decision in configured_idle_decisions
                ),
                "planner_policy_log_order": all(
                    position >= 0 for position in scale_up_log_positions
                )
                and idle_decision_position > scale_up_log_positions[-1],
            }
        )
        if expected_idle_stability_seconds > 0:
            assertions["terminal_idle_stability_observed"] = stability_valid
        # A warm target of one is already satisfied by the active-job worker,
        # so the idle decision is intentionally a scaling no-op. Other targets
        # must have an observable post-decision adapter transition.
        if expected_idle_replicas == 1:
            assertions[
                "watch_remained_at_expected_idle_target"
            ] = one_watch_index is not None and all(
                event["object"]["spec"]["replicas"] == 1
                for event in watch_events[one_watch_index:]
            )
        else:
            idle_watch_index = (
                next(
                    (
                        index
                        for index in range(one_watch_index + 1, len(watch_events))
                        if watch_events[index]["object"]["spec"]["replicas"]
                        == expected_idle_replicas
                    ),
                    None,
                )
                if one_watch_index is not None
                else None
            )
            watch_remained_idle = idle_watch_index is not None and all(
                event["object"]["spec"]["replicas"] == expected_idle_replicas
                for event in watch_events[idle_watch_index:]
            )
            idle_update_pattern = re.compile(
                rf"Updating decode component {re.escape(worker_component)} from \d+ "
                rf"to desired replica count {expected_idle_replicas}(?:\s|$)"
            )
            idle_update_match = idle_update_pattern.search(
                planner_log, max(idle_decision_position, 0)
            )
            idle_scaled_position = planner_log.find(
                f"Scaled DGDSA {adapter_name} to {expected_idle_replicas} replicas",
                max(idle_decision_position, 0),
            )
            assertions.update(
                {
                    "watch_observed_one_to_expected_idle_target": one_watch_index
                    is not None
                    and idle_watch_index is not None
                    and idle_watch_index > one_watch_index,
                    "watch_remained_at_expected_idle_target": watch_remained_idle,
                    "adapter_transition_observed_after_completion_and_idle_decision": (
                        adapter_target_after_completion_and_decision
                    ),
                    "planner_logged_idle_scale": idle_update_match is not None
                    and idle_scaled_position >= 0,
                    "planner_policy_log_order": assertions["planner_policy_log_order"]
                    and idle_update_match is not None
                    and idle_update_match.start() > idle_decision_position
                    and idle_scaled_position > idle_update_match.start(),
                }
            )

    notes = [
        "The Redis lease is authoritative; Async's drain gauge reports the last gate evaluation and can remain at 5 while the queue is idle.",
        "Dispatch ordering is certified only when a counter-unchanged scrape follows the first observed positive Redis lease, a later scrape increases, and no counter reset is observed between the baseline and that increase.",
    ]
    if expected_idle_replicas is None:
        notes.append(
            "Legacy evidence mode does not claim worker scale-down; pass --expected-idle-replicas to verify an exact post-batch warm target."
        )
    else:
        notes.append(
            "The configured post-batch warm target is accepted only after Gateway, frontend, Async, and worker inventory all report a safe idle state."
        )

    return {
        "schema_version": 1,
        "all_passed": all(assertions.values()),
        "assertions": assertions,
        "timeline": {
            "observer_first": _state_time(states, 0 if states else None),
            "adapter_one": _state_time(states, scale_index),
            "worker_ready": _state_time(states, ready_index),
            "positive_lease": _state_time(states, positive_index),
            "last_unchanged_dispatch_counter": last_unchanged_at,
            "first_dispatch_counter_increase": first_increase_at,
            "terminal_zero_lease": _state_time(states, terminal_zero_index),
            "post_completion_active_barrier": _state_time(
                states,
                post_completion_active_index
                if expected_idle_replicas is not None
                else None,
            ),
            "gateway_completed_first": (
                gateway_completed_first_at.isoformat().replace("+00:00", "Z")
                if gateway_completed_first_at is not None
                else None
            ),
            "gateway_completed_last": (
                gateway_completed_last_at.isoformat().replace("+00:00", "Z")
                if gateway_completed_last_at is not None
                else None
            ),
            "idle_decision": (
                idle_decision["observed_at"].isoformat().replace("+00:00", "Z")
                if idle_decision is not None
                and idle_decision["observed_at"] is not None
                else None
            ),
            "observer_last": _state_time(states, len(states) - 1 if states else None),
        },
        "observed": {
            "batch_id": terminal["id"],
            "state_samples": len(states),
            "adapter_generation": {
                "before": adapter_before["metadata"]["generation"],
                "after": adapter_after["metadata"]["generation"],
            },
            "adapter_resource_version": {
                "before": adapter_before["metadata"]["resourceVersion"],
                "after": adapter_after["metadata"]["resourceVersion"],
            },
            "dispatch_counter": {
                "before": dispatched_before,
                "after": dispatched_after,
                "continuous_through_first_increase": dispatch_counter_continuous,
                "last_unchanged_at": last_unchanged_at,
                "first_increase_value": first_increase_value,
            },
            "successful_counter": {
                "before": successful_before,
                "after": successful_after,
            },
            "redis_terminal_cap_rps": float(redis_after[2]),
            "redis_terminal_decision_id": redis_after[4],
            "first_terminal_zero_decision_id": terminal_zero_decision_id,
            "planner_selected_idle_decision_id": (
                idle_decision["decision_id"] if idle_decision is not None else None
            ),
            "redis_terminal_pttl_ms": redis_pttl_ms,
            "expected_lease_duration_seconds": expected_lease_duration_seconds,
            "lease_pttl_tolerance_seconds": LEASE_PTTL_TOLERANCE_SECONDS,
            "maximum_expected_lease_pttl_ms": maximum_expected_lease_pttl_ms,
            "async_idle_last_evaluated_drain_gauge_rps": _metric_value(
                metrics_after, DRAIN_GAUGE
            ),
            "expected_idle_replicas": expected_idle_replicas,
            "expected_idle_stability_seconds": expected_idle_stability_seconds,
            "idle_stability_suffix_start_index": (
                stability_suffix_index
                if expected_idle_replicas is not None
                and expected_idle_stability_seconds > 0
                else None
            ),
            "idle_stability_observed_seconds": (
                stability_observed_seconds
                if expected_idle_replicas is not None
                and expected_idle_stability_seconds > 0
                else None
            ),
            "planner_idle_replica_targets": [
                decision["idle_replica_target"] for decision in planner_decisions
            ],
        },
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--worker-component", default="worker")
    parser.add_argument("--adapter-name", default="qwen3-0-6b-batch-worker")
    parser.add_argument("--expected-idle-replicas", type=int)
    parser.add_argument(
        "--expected-idle-stability-seconds",
        type=float,
        default=0,
        help=(
            "required duration of the complete terminal idle-target state suffix; "
            "zero preserves legacy behavior"
        ),
    )
    parser.add_argument(
        "--expected-lease-duration-seconds",
        type=float,
        required=True,
        help=(
            "configured Planner drain lease duration; the terminal Redis PTTL "
            "must not exceed this duration plus five seconds of clock-skew "
            "tolerance"
        ),
    )
    args = parser.parse_args()

    result = verify(
        args.run_dir.resolve(),
        args.evidence_dir.resolve(),
        worker_component=args.worker_component,
        adapter_name=args.adapter_name,
        expected_idle_replicas=args.expected_idle_replicas,
        expected_lease_duration_seconds=args.expected_lease_duration_seconds,
        expected_idle_stability_seconds=args.expected_idle_stability_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
