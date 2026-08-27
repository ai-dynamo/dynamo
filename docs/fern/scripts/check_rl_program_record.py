#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate cross-cutting RL routing, weight, observability, and replay evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "dynamo.rl.program-evidence.v1"
COMMIT = re.compile(r"[0-9a-f]{40}")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
NUMERIC = (int, float)
DIAGNOSES = (
    "request_or_engine_queueing",
    "kv_routing_miss_or_cache_loss",
    "blocked_or_failed_weight_refresh",
)

REQUIRED_SHAPE: dict[str, Any] = {
    "schema": str,
    "record_state": str,
    "record_id": str,
    "pins": {
        "dynamo_commit": str,
        "framework_name": str,
        "framework_commit": str,
        "backend_name": str,
        "backend_version": str,
        "container_image": str,
        "container_image_digest": str,
        "model_name": str,
        "model_revision": str,
    },
    "owners": {
        "routing": str,
        "weight_updates": str,
        "observability": str,
        "replay_simulation": str,
        "clean_room_reviewer": str,
    },
    "run_window": {"started_at": str, "completed_at": str, "artifact_root": str},
    "routing": {
        "status": str,
        "headline_metric": {
            "name": str,
            "numerator": str,
            "denominator": str,
            "freshness_rule": str,
        },
        "workload": {
            "name": str,
            "request_count": int,
            "unique_prompts": int,
            "samples_per_prompt": int,
            "schedule": str,
            "prompt_length_distribution": str,
            "output_length_distribution": str,
            "prefix_sharing_shape": str,
            "session_shape": str,
            "concurrency": int,
            "seed": int,
        },
        "fixed_controls": list,
        "variants": list,
        "mechanism_evidence": str,
        "claim_boundary": str,
        "limitations": list,
    },
    "weight_paths": {"status": str, "paths": list, "limitations": list},
    "observability": {
        "status": str,
        "clock_synchronization": str,
        "trace_overhead": {
            "baseline_repetitions": int,
            "traced_repetitions": int,
            "percent": (int, float, type(None)),
            "artifact": str,
        },
        "diagnoses": dict,
        "limitations": list,
    },
    "replay_simulation": {
        "status": str,
        "capture": {
            "framework_attempts": int,
            "expected_replay_requests": int,
            "trace_requests": int,
            "input_tokens": int,
            "output_tokens": int,
            "sessions": int,
            "trace_block_size": int,
            "artifact": str,
        },
        "live_replay": {"status": str, "repetitions": int, "artifact": str},
        "dynosim": {"status": str, "repetitions": int, "artifact": str},
        "calibration": {
            "metrics": list,
            "material_error_threshold_percent": (int, float, type(None)),
            "material_error_disclosure": str,
            "conclusion": str,
            "artifact": str,
        },
        "limitations": list,
    },
    "last_validated": str,
}


class RecordError(ValueError):
    """Raised when a program record cannot be loaded."""


def load_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecordError(f"cannot load {path}: {exc}") from exc
    if not isinstance(record, dict):
        raise RecordError(f"{path} must contain a JSON object")
    return record


def _matches_type(value: Any, expected: type | tuple[type, ...]) -> bool:
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(expected, tuple) and int in expected and isinstance(value, bool):
        return False
    return isinstance(value, expected)


def _check_shape(value: Any, expected: Any, location: str, findings: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            findings.append(f"{location} must be an object")
            return
        for key, child in expected.items():
            child_location = f"{location}.{key}" if location else key
            if key not in value:
                findings.append(f"{child_location} is required")
            else:
                _check_shape(value[key], child, child_location, findings)
        return
    if not _matches_type(value, expected):
        names = (
            ", ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        findings.append(f"{location} must be {names}")


def _parse_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _nonempty(value: str, location: str, findings: list[str]) -> None:
    if not value.strip():
        findings.append(f"{location} must be recorded for publication")


def _artifact_list(value: Any, location: str, findings: list[str]) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
    ):
        findings.append(f"{location} must contain at least one artifact URI")


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, NUMERIC)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _validate_variant_shape(variant: Any, index: int, findings: list[str]) -> None:
    location = f"routing.variants[{index}]"
    expected = {
        "name": str,
        "baseline": bool,
        "router_config": dict,
        "repetitions": int,
        "metrics": dict,
        "artifacts": list,
    }
    _check_shape(variant, expected, location, findings)


def _validate_weight_shape(path: Any, index: int, findings: list[str]) -> None:
    location = f"weight_paths.paths[{index}]"
    expected = {
        "name": str,
        "status": str,
        "placement": str,
        "serving_mode": str,
        "framework_name": str,
        "framework_commit": str,
        "backend": str,
        "backend_version": str,
        "container_image": str,
        "container_image_digest": str,
        "model_name": str,
        "model_revision": str,
        "transport": str,
        "model_class": str,
        "source_parallelism": {"tp": int, "pp": int, "dp": int, "ep": int},
        "target_parallelism": {"tp": int, "pp": int, "dp": int, "ep": int},
        "workers_targeted": int,
        "workers_verified": int,
        "cache_handling": str,
        "version_verified": bool,
        "output_mutation_or_numerical_validation": bool,
        "partial_failure_recovered": bool,
        "post_update_generation": bool,
        "artifacts": list,
    }
    _check_shape(path, expected, location, findings)


def _validate_metric_shape(metric: Any, index: int, findings: list[str]) -> None:
    location = f"replay_simulation.calibration.metrics[{index}]"
    expected = {
        "name": str,
        "unit": str,
        "live": (int, float, type(None)),
        "simulated": (int, float, type(None)),
        "absolute_error": (int, float, type(None)),
        "relative_error_percent": (int, float, type(None)),
    }
    _check_shape(metric, expected, location, findings)


def validate_structure(record: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    _check_shape(record, REQUIRED_SHAPE, "", findings)
    if findings:
        return findings
    if record["schema"] != SCHEMA:
        findings.append(f"schema must be {SCHEMA!r}")
    if record["record_state"] not in {"planned", "passed", "failed"}:
        findings.append("record_state must be planned, passed, or failed")
    for section in ("routing", "weight_paths", "observability", "replay_simulation"):
        if record[section]["status"] not in {"not_run", "passed", "failed"}:
            findings.append(f"{section}.status must be not_run, passed, or failed")
    for index, variant in enumerate(record["routing"]["variants"]):
        _validate_variant_shape(variant, index, findings)
    for index, weight_path in enumerate(record["weight_paths"]["paths"]):
        _validate_weight_shape(weight_path, index, findings)
    diagnoses = record["observability"]["diagnoses"]
    for name in DIAGNOSES:
        if name not in diagnoses:
            findings.append(f"observability.diagnoses.{name} is required")
            continue
        _check_shape(
            diagnoses[name],
            {"status": str, "conclusion": str, "artifacts": list},
            f"observability.diagnoses.{name}",
            findings,
        )
        if isinstance(diagnoses[name], dict) and diagnoses[name].get("status") not in {
            "not_run",
            "passed",
            "failed",
        }:
            findings.append(
                f"observability.diagnoses.{name}.status must be not_run, passed, or failed"
            )
    for index, metric in enumerate(
        record["replay_simulation"]["calibration"]["metrics"]
    ):
        _validate_metric_shape(metric, index, findings)
    for location in (
        "run_window.started_at",
        "run_window.completed_at",
        "last_validated",
    ):
        value: Any = record
        for part in location.split("."):
            value = value[part]
        if value and _parse_time(value) is None:
            findings.append(f"{location} must be ISO-8601 with a UTC offset when set")
    return findings


def _common_publication(record: dict[str, Any], findings: list[str]) -> None:
    if record["record_state"] != "passed":
        findings.append("record_state must be passed for publication")
    _nonempty(record["record_id"], "record_id", findings)
    pins = record["pins"]
    for field in ("dynamo_commit", "framework_commit"):
        if not COMMIT.fullmatch(pins[field]):
            findings.append(f"pins.{field} must be a full lowercase commit SHA")
    for field in (
        "framework_name",
        "backend_name",
        "backend_version",
        "container_image",
        "model_name",
        "model_revision",
    ):
        _nonempty(pins[field], f"pins.{field}", findings)
    if not IMAGE_DIGEST.fullmatch(pins["container_image_digest"]):
        findings.append(
            "pins.container_image_digest must be an immutable sha256 digest"
        )
    for field, value in record["owners"].items():
        _nonempty(value, f"owners.{field}", findings)
    reviewer = record["owners"]["clean_room_reviewer"]
    topic_owners = {
        value for key, value in record["owners"].items() if key != "clean_room_reviewer"
    }
    if reviewer and reviewer in topic_owners:
        findings.append(
            "owners.clean_room_reviewer must be independent of topic owners"
        )
    window = record["run_window"]
    started = _parse_time(window["started_at"])
    completed = _parse_time(window["completed_at"])
    if started is None:
        findings.append(
            "run_window.started_at must be a recorded timestamp with offset"
        )
    if completed is None:
        findings.append(
            "run_window.completed_at must be a recorded timestamp with offset"
        )
    if started is not None and completed is not None and completed < started:
        findings.append("run_window.completed_at must not precede started_at")
    _nonempty(window["artifact_root"], "run_window.artifact_root", findings)
    if _parse_time(record["last_validated"]) is None:
        findings.append("last_validated must be a recorded timestamp with offset")


def _routing_publication(record: dict[str, Any], findings: list[str]) -> None:
    routing = record["routing"]
    if routing["status"] != "passed":
        findings.append("routing.status must be passed")
    for field, value in routing["headline_metric"].items():
        _nonempty(value, f"routing.headline_metric.{field}", findings)
    workload = routing["workload"]
    for field in (
        "name",
        "schedule",
        "prompt_length_distribution",
        "output_length_distribution",
        "prefix_sharing_shape",
        "session_shape",
    ):
        _nonempty(workload[field], f"routing.workload.{field}", findings)
    for field in (
        "request_count",
        "unique_prompts",
        "samples_per_prompt",
        "concurrency",
    ):
        if workload[field] < 1:
            findings.append(f"routing.workload.{field} must be at least 1")
    if not routing["fixed_controls"] or any(
        not isinstance(item, str) or not item for item in routing["fixed_controls"]
    ):
        findings.append("routing.fixed_controls must list the matched conditions")
    variants = routing["variants"]
    if len(variants) < 2:
        findings.append(
            "routing.variants must contain a baseline and at least one variant"
        )
    if sum(bool(item.get("baseline")) for item in variants) != 1:
        findings.append("routing.variants must contain exactly one baseline")
    headline = routing["headline_metric"]["name"]
    for index, variant in enumerate(variants):
        location = f"routing.variants[{index}]"
        _nonempty(variant["name"], f"{location}.name", findings)
        if not variant["router_config"]:
            findings.append(f"{location}.router_config must record the complete config")
        if variant["repetitions"] < 3:
            findings.append(f"{location}.repetitions must be at least 3")
        if (
            not headline
            or headline not in variant["metrics"]
            or not _finite_number(variant["metrics"].get(headline))
        ):
            findings.append(
                f"{location}.metrics must contain numeric headline metric {headline!r}"
            )
        _artifact_list(variant["artifacts"], f"{location}.artifacts", findings)
    _nonempty(routing["mechanism_evidence"], "routing.mechanism_evidence", findings)
    if routing["claim_boundary"] != "live_measurement":
        findings.append(
            "routing.claim_boundary must be live_measurement for publication"
        )


def _weight_publication(record: dict[str, Any], findings: list[str]) -> None:
    weight = record["weight_paths"]
    if weight["status"] != "passed":
        findings.append("weight_paths.status must be passed")
    paths = weight["paths"]
    if len(paths) < 2:
        findings.append("weight_paths.paths must contain at least two validated paths")
    if not any(path.get("placement") == "colocated" for path in paths):
        findings.append("weight_paths.paths must include a colocated path")
    if not any(path.get("serving_mode") == "disaggregated" for path in paths):
        findings.append("weight_paths.paths must include a disaggregated serving path")
    colocated = [
        index
        for index, path in enumerate(paths)
        if path.get("placement") == "colocated"
    ]
    disaggregated = [
        index
        for index, path in enumerate(paths)
        if path.get("serving_mode") == "disaggregated"
    ]
    if (
        colocated
        and disaggregated
        and not any(
            colocated_index != disaggregated_index
            for colocated_index in colocated
            for disaggregated_index in disaggregated
        )
    ):
        findings.append(
            "weight_paths.paths must use distinct colocated and disaggregated paths"
        )
    for index, path in enumerate(paths):
        location = f"weight_paths.paths[{index}]"
        if path["status"] != "passed":
            findings.append(f"{location}.status must be passed")
        if path["placement"] not in {"colocated", "external"}:
            findings.append(f"{location}.placement must be colocated or external")
        if path["serving_mode"] not in {"aggregated", "disaggregated"}:
            findings.append(
                f"{location}.serving_mode must be aggregated or disaggregated"
            )
        for field in (
            "name",
            "framework_name",
            "backend",
            "backend_version",
            "container_image",
            "model_name",
            "model_revision",
            "transport",
            "model_class",
        ):
            _nonempty(path[field], f"{location}.{field}", findings)
        if not COMMIT.fullmatch(path["framework_commit"]):
            findings.append(
                f"{location}.framework_commit must be a full lowercase commit SHA"
            )
        if not IMAGE_DIGEST.fullmatch(path["container_image_digest"]):
            findings.append(
                f"{location}.container_image_digest must be an immutable sha256 digest"
            )
        for group in ("source_parallelism", "target_parallelism"):
            for field, value in path[group].items():
                if value < 1:
                    findings.append(f"{location}.{group}.{field} must be at least 1")
        if path["workers_targeted"] < 1:
            findings.append(f"{location}.workers_targeted must be at least 1")
        if path["workers_verified"] != path["workers_targeted"]:
            findings.append(f"{location}.workers_verified must equal workers_targeted")
        if path["cache_handling"] not in {
            "invalidated",
            "verified_preserved",
            "not_applicable",
        }:
            findings.append(f"{location}.cache_handling must be verified")
        for field in (
            "version_verified",
            "output_mutation_or_numerical_validation",
            "partial_failure_recovered",
            "post_update_generation",
        ):
            if not path[field]:
                findings.append(f"{location}.{field} must be true")
        _artifact_list(path["artifacts"], f"{location}.artifacts", findings)


def _observability_publication(record: dict[str, Any], findings: list[str]) -> None:
    observability = record["observability"]
    if observability["status"] != "passed":
        findings.append("observability.status must be passed")
    _nonempty(
        observability["clock_synchronization"],
        "observability.clock_synchronization",
        findings,
    )
    overhead = observability["trace_overhead"]
    for field in ("baseline_repetitions", "traced_repetitions"):
        if overhead[field] < 3:
            findings.append(f"observability.trace_overhead.{field} must be at least 3")
    if not _finite_number(overhead["percent"]) or overhead["percent"] < 0:
        findings.append(
            "observability.trace_overhead.percent must be measured and nonnegative"
        )
    _nonempty(overhead["artifact"], "observability.trace_overhead.artifact", findings)
    for name in DIAGNOSES:
        diagnosis = observability["diagnoses"][name]
        location = f"observability.diagnoses.{name}"
        if diagnosis["status"] != "passed":
            findings.append(f"{location}.status must be passed")
        _nonempty(diagnosis["conclusion"], f"{location}.conclusion", findings)
        _artifact_list(diagnosis["artifacts"], f"{location}.artifacts", findings)


def _replay_publication(record: dict[str, Any], findings: list[str]) -> None:
    replay = record["replay_simulation"]
    if replay["status"] != "passed":
        findings.append("replay_simulation.status must be passed")
    capture = replay["capture"]
    if capture["framework_attempts"] < 1:
        findings.append(
            "replay_simulation.capture.framework_attempts must be at least 1"
        )
    if capture["expected_replay_requests"] < 1:
        findings.append(
            "replay_simulation.capture.expected_replay_requests must be at least 1"
        )
    if capture["trace_requests"] != capture["expected_replay_requests"]:
        findings.append(
            "replay_simulation.capture.trace_requests must equal expected_replay_requests"
        )
    for field in ("input_tokens", "output_tokens", "trace_block_size"):
        if capture[field] < 1:
            findings.append(f"replay_simulation.capture.{field} must be at least 1")
    if capture["sessions"] < 0:
        findings.append("replay_simulation.capture.sessions must be nonnegative")
    _nonempty(capture["artifact"], "replay_simulation.capture.artifact", findings)
    for section in ("live_replay", "dynosim"):
        run = replay[section]
        if run["status"] != "passed":
            findings.append(f"replay_simulation.{section}.status must be passed")
        if run["repetitions"] < 3:
            findings.append(
                f"replay_simulation.{section}.repetitions must be at least 3"
            )
        _nonempty(run["artifact"], f"replay_simulation.{section}.artifact", findings)
    calibration = replay["calibration"]
    if not calibration["metrics"]:
        findings.append("replay_simulation.calibration.metrics must not be empty")
    for index, metric in enumerate(calibration["metrics"]):
        location = f"replay_simulation.calibration.metrics[{index}]"
        for field in ("name", "unit"):
            _nonempty(metric[field], f"{location}.{field}", findings)
        for field in ("live", "simulated", "absolute_error"):
            if not _finite_number(metric[field]):
                findings.append(f"{location}.{field} must be finite")
        if all(
            _finite_number(metric[field])
            for field in ("live", "simulated", "absolute_error")
        ):
            expected_absolute = abs(float(metric["live"]) - float(metric["simulated"]))
            if not math.isclose(
                float(metric["absolute_error"]),
                expected_absolute,
                rel_tol=1e-6,
                abs_tol=1e-9,
            ):
                findings.append(
                    f"{location}.absolute_error does not match live versus simulated"
                )
            expected_relative = (
                None
                if float(metric["live"]) == 0
                else expected_absolute / abs(float(metric["live"])) * 100
            )
            relative = metric["relative_error_percent"]
            if expected_relative is None:
                if relative is not None:
                    findings.append(
                        f"{location}.relative_error_percent must be null when live is zero"
                    )
            elif not _finite_number(relative) or not math.isclose(
                float(relative), expected_relative, rel_tol=1e-6, abs_tol=1e-9
            ):
                findings.append(
                    f"{location}.relative_error_percent does not match live versus simulated"
                )
    threshold = calibration["material_error_threshold_percent"]
    if not _finite_number(threshold) or threshold < 0:
        findings.append(
            "replay_simulation.calibration.material_error_threshold_percent must be nonnegative"
        )
    for field in ("material_error_disclosure", "conclusion", "artifact"):
        _nonempty(
            calibration[field], f"replay_simulation.calibration.{field}", findings
        )


def publication_findings(record: dict[str, Any]) -> list[str]:
    findings = validate_structure(record)
    if findings:
        return findings
    _common_publication(record, findings)
    _routing_publication(record, findings)
    _weight_publication(record, findings)
    _observability_publication(record, findings)
    _replay_publication(record, findings)
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument("--publication-gate", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        record = load_record(args.record)
    except RecordError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    findings = (
        publication_findings(record)
        if args.publication_gate
        else validate_structure(record)
    )
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}", file=sys.stderr)
        return 1
    mode = "publication" if args.publication_gate else "structure"
    print(f"RL program evidence passed ({mode}; schema={SCHEMA}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
