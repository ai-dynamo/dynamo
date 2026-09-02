#!/usr/bin/env python3
"""Verify a native Planner autonomous scale-from-zero batch run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

DISPATCHED = "llm_d_async_async_dispatched_requests_total"
SUCCESSFUL = "llm_d_async_async_successful_requests_total"
BACKLOG = "llm_d_async_async_broker_backlog"
INFLIGHT = "llm_d_async_async_inflight_requests"
QUEUE_DEPTH = "llm_d_async_async_queue_depth"
DRAIN_GAUGE = "llm_d_async_async_drain_limit_rps"


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _lease_cap(state: dict[str, Any]) -> float:
    fields = next(csv.reader([state["lease_csv"]]))
    if len(fields) != 5:
        raise ValueError(f"expected five lease fields, got {fields!r}")
    return float(fields[2])


def _metric_value(payload: str, name: str) -> float:
    for line in payload.splitlines():
        if not line.startswith(name):
            continue
        if "{" in line and 'pool_name="dynamo-batch"' not in line:
            continue
        return float(line.rsplit(maxsplit=1)[1])
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


def _first_log_time(log: str, needle: str) -> datetime:
    for line in log.splitlines():
        if needle not in line:
            continue
        return _parse_rfc3339(line.split(maxsplit=1)[0])
    raise ValueError(f"Planner log entry not found: {needle!r}")


def _first_metric_increase(
    run_dir: Path, baseline: float
) -> tuple[str | None, float | None]:
    for prom_path in sorted((run_dir / "metrics" / "async").glob("*.prom")):
        value = _metric_value(prom_path.read_text(encoding="utf-8"), DISPATCHED)
        if value <= baseline:
            continue
        metadata_path = prom_path.with_suffix(".json")
        observed_at = _read_json(metadata_path)["observed_at"]
        return observed_at, value
    return None, None


def _first_index(items: list[Any], predicate: Any) -> int:
    return next(index for index, item in enumerate(items) if predicate(item))


def verify(run_dir: Path, evidence_dir: Path) -> dict[str, Any]:
    states = _read_jsonl(evidence_dir / "state.jsonl")
    caps = [_lease_cap(state) for state in states]
    scale_index = _first_index(states, lambda state: state["adapter_spec"] == 1)
    ready_index = _first_index(states, lambda state: state["ready_worker_pods"] >= 1)
    positive_index = _first_index(caps, lambda cap: cap > 0)
    terminal_zero_index = next(
        index for index in range(positive_index + 1, len(caps)) if caps[index] == 0
    )

    terminal = _read_json(run_dir / "terminal-batch.json")
    validation = _read_json(run_dir / "result-validation.json")
    adapter_before = _read_json(evidence_dir / "dgdsa.before.json")
    adapter_after = _read_json(evidence_dir / "dgdsa.after.json")
    watch_events = _read_jsonl(evidence_dir / "dgdsa.watch.jsonstream")
    planner_log = (evidence_dir / "planner.log").read_text(encoding="utf-8")

    redis_after = (
        (evidence_dir / "redis.after.txt").read_text(encoding="utf-8").splitlines()
    )
    redis_pttl_ms = int(
        (evidence_dir / "redis.after.pttl-ms.txt").read_text(encoding="utf-8").strip()
    )

    metrics_before = (evidence_dir / "async-metrics.before.txt").read_text(
        encoding="utf-8"
    )
    metrics_after = (evidence_dir / "async-metrics.after.txt").read_text(
        encoding="utf-8"
    )
    dispatched_before = _metric_value(metrics_before, DISPATCHED)
    dispatched_after = _metric_value(metrics_after, DISPATCHED)
    successful_before = _metric_value(metrics_before, SUCCESSFUL)
    successful_after = _metric_value(metrics_after, SUCCESSFUL)
    first_increase_at, first_increase_value = _first_metric_increase(
        run_dir, dispatched_before
    )
    positive_log_at = _first_log_time(planner_log, "max_admission_rps=5.0")

    log_needles = [
        "replica_floor=1 max_admission_rps=0.0",
        "Updating decode component VllmDecodeWorker from 0 to desired replica count 1",
        "Scaled DGDSA qwen3-0-6b-batch-vllmdecodeworker to 1 replicas",
        "replica_floor=1 max_admission_rps=5.0",
        "replica_floor=0 max_admission_rps=0.0",
    ]
    log_positions: list[int] = []
    log_cursor = 0
    for needle in log_needles:
        position = planner_log.find(needle, log_cursor)
        log_positions.append(position)
        if position >= 0:
            log_cursor = position + len(needle)

    assertions = {
        "harness_exit_zero": (run_dir / "exit_code.txt").read_text().strip() == "0",
        "gateway_terminal_100_100_0": terminal.get("status") == "completed"
        and terminal.get("request_counts")
        == {"completed": 100, "failed": 0, "total": 100},
        "downloaded_results_valid": validation.get("valid") is True
        and validation.get("downloaded_output_lines") == 100
        and validation.get("unique_custom_ids") == 100,
        "adapter_started_at_zero": adapter_before["spec"]["replicas"] == 0
        and adapter_before["status"]["replicas"] == 0
        and states[0]["adapter_spec"] == 0,
        "adapter_finished_at_one": adapter_after["spec"]["replicas"] == 1
        and adapter_after["status"]["replicas"] == 1
        and states[-1]["adapter_spec"] == 1,
        "watch_observed_zero_to_one": watch_events[0]["object"]["spec"]["replicas"] == 0
        and any(event["object"]["spec"]["replicas"] == 1 for event in watch_events),
        "planner_logged_authoritative_scale": all(
            position >= 0 for position in log_positions[:3]
        ),
        "planner_policy_log_order": all(position >= 0 for position in log_positions),
        "scale_preceded_worker_readiness": scale_index < ready_index,
        "lease_remained_closed_until_ready": positive_index > ready_index
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
        and _parse_rfc3339(first_increase_at) >= positive_log_at,
        "terminal_zero_observed_after_positive": terminal_zero_index > positive_index
        and caps[-1] == 0,
        "authoritative_terminal_lease_zero_and_fresh": len(redis_after) >= 5
        and redis_after[0] == "llm-d.ai/v1alpha1"
        and redis_after[1] == "dynamo-batch"
        and float(redis_after[2]) == 0
        and 0 < redis_pttl_ms <= 60_000,
        "exactly_100_dispatches_and_successes": dispatched_after - dispatched_before
        == 100
        and successful_after - successful_before == 100,
        "async_terminal_queues_empty": _metric_value(metrics_after, BACKLOG) == 0
        and _metric_value(metrics_after, INFLIGHT) == 0
        and _metric_value(metrics_after, QUEUE_DEPTH) == 0,
    }

    return {
        "schema_version": 1,
        "all_passed": all(assertions.values()),
        "assertions": assertions,
        "timeline": {
            "observer_first": _normalize_observer_time(states[0]["observed_at"]),
            "adapter_one": _normalize_observer_time(states[scale_index]["observed_at"]),
            "worker_ready": _normalize_observer_time(
                states[ready_index]["observed_at"]
            ),
            "positive_lease": _normalize_observer_time(
                states[positive_index]["observed_at"]
            ),
            "first_dispatch_counter_increase": first_increase_at,
            "terminal_zero_lease": _normalize_observer_time(
                states[terminal_zero_index]["observed_at"]
            ),
            "observer_last": _normalize_observer_time(states[-1]["observed_at"]),
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
                "first_increase_value": first_increase_value,
            },
            "successful_counter": {
                "before": successful_before,
                "after": successful_after,
            },
            "redis_terminal_cap_rps": float(redis_after[2]),
            "redis_terminal_pttl_ms": redis_pttl_ms,
            "async_idle_last_evaluated_drain_gauge_rps": _metric_value(
                metrics_after, DRAIN_GAUGE
            ),
        },
        "notes": [
            "The Redis lease is authoritative; Async's drain gauge reports the last gate evaluation and can remain at 5 while the queue is idle.",
            "A batch floor is a lower bound, so the worker may remain at one replica after the batch completes even though the floor and drain cap return to zero.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()

    result = verify(args.run_dir.resolve(), args.evidence_dir.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
