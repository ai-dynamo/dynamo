#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate an RL framework run record and its publication-gate evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCHEMA = "dynamo.rl.validation.v1"
COMMIT = re.compile(r"[0-9a-f]{40}")
IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
GATE_NAMES = (
    "generation_smoke",
    "token_logprob",
    "training_iteration",
    "policy_update",
    "retry_and_cancellation",
    "failure_recovery",
    "trace_correlation",
)

REQUIRED_SHAPE: dict[str, Any] = {
    "schema": str,
    "record_state": str,
    "record_id": str,
    "framework": {
        "name": str,
        "integration_artifact": str,
        "recipe_commit": str,
        "core_commit": str,
    },
    "environment": {
        "dynamo_commit": str,
        "backend": {"name": str, "version": str},
        "container_image": str,
        "container_image_digest": str,
        "cuda_version": str,
        "driver_version": str,
        "model": {
            "name": str,
            "revision": str,
            "tokenizer_revision": str,
        },
        "artifacts": list,
    },
    "hardware": {
        "nodes": int,
        "gpu_model": str,
        "gpus_per_node": int,
        "interconnect": str,
        "network": str,
        "artifacts": list,
    },
    "topology": {
        "placement": str,
        "serving_mode": str,
        "trainer_parallelism": {"tp": int, "pp": int, "dp": int, "ep": int},
        "rollout_parallelism": {"tp": int, "pp": int, "dp": int, "ep": int},
    },
    "owners": {"framework": str, "dynamo": str, "clean_room_reviewer": str},
    "run": {
        "started_at": str,
        "completed_at": str,
        "commands": list,
        "artifact_root": str,
    },
    "gates": {
        "generation_smoke": {"status": str, "artifacts": list},
        "token_logprob": {
            "status": str,
            "exact_completion_token_ids": bool,
            "completion_logprobs_aligned": bool,
            "prompt_logprobs": str,
            "terminal_reasons_verified": bool,
            "artifacts": list,
        },
        "training_iteration": {
            "status": str,
            "optimizer_steps": int,
            "rollout_phase_completed": bool,
            "reward_or_advantage_completed": bool,
            "actor_update_completed": bool,
            "weight_sync_completed": bool,
            "post_update_rollout_completed": bool,
            "artifacts": list,
        },
        "policy_update": {
            "status": str,
            "target_version": str,
            "workers_targeted": int,
            "workers_verified": int,
            "cache_handling": str,
            "post_update_generation": bool,
            "artifacts": list,
        },
        "retry_and_cancellation": {
            "status": str,
            "duplicate_suppression_verified": bool,
            "canceled_incomplete_sample_verified": bool,
            "artifacts": list,
        },
        "failure_recovery": {
            "status": str,
            "request_failure_recovered": bool,
            "worker_failure_recovered": bool,
            "weight_update_failure_recovered": bool,
            "artifacts": list,
        },
        "trace_correlation": {
            "status": str,
            "framework_attempts": int,
            "joined_payloads": int,
            "expected_terminals": int,
            "joined_terminals": int,
            "trace_overhead_percent": (int, float, type(None)),
            "artifacts": list,
        },
    },
    "limitations": list,
    "last_validated": str,
}


class RecordError(ValueError):
    """Raised when the validation record cannot be loaded."""


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
        for key, child_shape in expected.items():
            child_location = f"{location}.{key}" if location else key
            if key not in value:
                findings.append(f"{child_location} is required")
            else:
                _check_shape(value[key], child_shape, child_location, findings)
        return
    if not _matches_type(value, expected):
        names = (
            ", ".join(item.__name__ for item in expected)
            if isinstance(expected, tuple)
            else expected.__name__
        )
        findings.append(f"{location} must be {names}")


def _parse_iso_time(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _iso_time(value: str) -> bool:
    return _parse_iso_time(value) is not None


def validate_structure(record: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    _check_shape(record, REQUIRED_SHAPE, "", findings)
    if findings:
        return findings
    if record["schema"] != SCHEMA:
        findings.append(f"schema must be {SCHEMA!r}")
    if record["record_state"] not in {"planned", "passed", "failed"}:
        findings.append("record_state must be planned, passed, or failed")
    if record["topology"]["placement"] not in {"colocated", "external"}:
        findings.append("topology.placement must be colocated or external")
    if record["topology"]["serving_mode"] not in {
        "aggregated",
        "disaggregated",
    }:
        findings.append("topology.serving_mode must be aggregated or disaggregated")
    for gate_name in GATE_NAMES:
        status = record["gates"][gate_name]["status"]
        if status not in {"not_run", "passed", "failed"}:
            findings.append(
                f"gates.{gate_name}.status must be not_run, passed, or failed"
            )
    prompt_status = record["gates"]["token_logprob"]["prompt_logprobs"]
    if prompt_status not in {"not_run", "verified", "not_applicable"}:
        findings.append(
            "gates.token_logprob.prompt_logprobs must be not_run, verified, or not_applicable"
        )
    cache_status = record["gates"]["policy_update"]["cache_handling"]
    if cache_status not in {
        "unverified",
        "invalidated",
        "verified_preserved",
        "not_applicable",
    }:
        findings.append(
            "gates.policy_update.cache_handling must be unverified, invalidated, verified_preserved, or not_applicable"
        )
    for location in ("run.started_at", "run.completed_at", "last_validated"):
        container: Any = record
        for part in location.split("."):
            container = container[part]
        if container and not _iso_time(container):
            findings.append(f"{location} must be ISO-8601 with a UTC offset when set")
    return findings


def _require_nonempty(value: str, location: str, findings: list[str]) -> None:
    if not value.strip():
        findings.append(f"{location} must be recorded for publication")


def _require_artifacts(
    gate: dict[str, Any], location: str, findings: list[str]
) -> None:
    artifacts = gate["artifacts"]
    if not artifacts or any(
        not isinstance(item, str) or not item for item in artifacts
    ):
        findings.append(f"{location}.artifacts must contain at least one artifact URI")


def publication_findings(record: dict[str, Any]) -> list[str]:
    findings = validate_structure(record)
    if findings:
        return findings
    if record["record_state"] != "passed":
        findings.append("record_state must be passed for publication")
    _require_nonempty(record["record_id"], "record_id", findings)

    framework = record["framework"]
    for field in ("name", "integration_artifact"):
        _require_nonempty(framework[field], f"framework.{field}", findings)
    for field in ("recipe_commit", "core_commit"):
        if not COMMIT.fullmatch(framework[field]):
            findings.append(f"framework.{field} must be a full lowercase commit SHA")

    environment = record["environment"]
    if not COMMIT.fullmatch(environment["dynamo_commit"]):
        findings.append("environment.dynamo_commit must be a full lowercase commit SHA")
    for field in (
        "container_image",
        "cuda_version",
        "driver_version",
    ):
        _require_nonempty(environment[field], f"environment.{field}", findings)
    if not IMAGE_DIGEST.fullmatch(environment["container_image_digest"]):
        findings.append(
            "environment.container_image_digest must be an immutable sha256 digest"
        )
    for field in ("name", "version"):
        _require_nonempty(
            environment["backend"][field], f"environment.backend.{field}", findings
        )
    for field in ("name", "revision", "tokenizer_revision"):
        _require_nonempty(
            environment["model"][field], f"environment.model.{field}", findings
        )
    _require_artifacts(environment, "environment", findings)

    hardware = record["hardware"]
    if hardware["nodes"] < 1:
        findings.append("hardware.nodes must be at least 1")
    if hardware["gpus_per_node"] < 1:
        findings.append("hardware.gpus_per_node must be at least 1")
    for field in ("gpu_model", "interconnect", "network"):
        _require_nonempty(hardware[field], f"hardware.{field}", findings)
    _require_artifacts(hardware, "hardware", findings)
    for group in ("trainer_parallelism", "rollout_parallelism"):
        for field, value in record["topology"][group].items():
            if value < 1:
                findings.append(f"topology.{group}.{field} must be at least 1")

    for field, value in record["owners"].items():
        _require_nonempty(value, f"owners.{field}", findings)
    reviewer = record["owners"]["clean_room_reviewer"]
    if reviewer and reviewer in {
        record["owners"]["framework"],
        record["owners"]["dynamo"],
    }:
        findings.append(
            "owners.clean_room_reviewer must be independent of the framework and Dynamo owners"
        )
    run = record["run"]
    for field in ("started_at", "completed_at"):
        if not _iso_time(run[field]):
            findings.append(f"run.{field} must be a recorded ISO-8601 timestamp")
    started = _parse_iso_time(run["started_at"])
    completed = _parse_iso_time(run["completed_at"])
    if started is not None and completed is not None:
        if completed < started:
            findings.append("run.completed_at must not precede run.started_at")
    if not run["commands"] or any(
        not isinstance(command, str) or not command for command in run["commands"]
    ):
        findings.append("run.commands must contain the exact executed commands")
    _require_nonempty(run["artifact_root"], "run.artifact_root", findings)
    if not _iso_time(record["last_validated"]):
        findings.append("last_validated must be a recorded ISO-8601 timestamp")

    gates = record["gates"]
    for gate_name in GATE_NAMES:
        gate = gates[gate_name]
        if gate["status"] != "passed":
            findings.append(f"gates.{gate_name}.status must be passed")
        _require_artifacts(gate, f"gates.{gate_name}", findings)

    token_gate = gates["token_logprob"]
    for field in (
        "exact_completion_token_ids",
        "completion_logprobs_aligned",
        "terminal_reasons_verified",
    ):
        if not token_gate[field]:
            findings.append(f"gates.token_logprob.{field} must be true")
    if token_gate["prompt_logprobs"] not in {"verified", "not_applicable"}:
        findings.append(
            "gates.token_logprob.prompt_logprobs must be verified or not_applicable"
        )

    training_gate = gates["training_iteration"]
    if training_gate["optimizer_steps"] < 1:
        findings.append("gates.training_iteration.optimizer_steps must be at least 1")
    for field in (
        "rollout_phase_completed",
        "reward_or_advantage_completed",
        "actor_update_completed",
        "weight_sync_completed",
        "post_update_rollout_completed",
    ):
        if not training_gate[field]:
            findings.append(f"gates.training_iteration.{field} must be true")

    update_gate = gates["policy_update"]
    _require_nonempty(
        update_gate["target_version"],
        "gates.policy_update.target_version",
        findings,
    )
    if update_gate["workers_targeted"] < 1:
        findings.append("gates.policy_update.workers_targeted must be at least 1")
    if update_gate["workers_verified"] != update_gate["workers_targeted"]:
        findings.append(
            "gates.policy_update.workers_verified must equal workers_targeted"
        )
    if update_gate["cache_handling"] == "unverified":
        findings.append("gates.policy_update.cache_handling must be verified")
    if not update_gate["post_update_generation"]:
        findings.append("gates.policy_update.post_update_generation must be true")

    retry_gate = gates["retry_and_cancellation"]
    for field in (
        "duplicate_suppression_verified",
        "canceled_incomplete_sample_verified",
    ):
        if not retry_gate[field]:
            findings.append(f"gates.retry_and_cancellation.{field} must be true")

    failure_gate = gates["failure_recovery"]
    for field in (
        "request_failure_recovered",
        "worker_failure_recovered",
        "weight_update_failure_recovered",
    ):
        if not failure_gate[field]:
            findings.append(f"gates.failure_recovery.{field} must be true")

    trace_gate = gates["trace_correlation"]
    if trace_gate["framework_attempts"] < 1:
        findings.append("gates.trace_correlation.framework_attempts must be at least 1")
    if trace_gate["joined_payloads"] != trace_gate["framework_attempts"]:
        findings.append(
            "gates.trace_correlation.joined_payloads must equal framework_attempts"
        )
    if trace_gate["joined_terminals"] != trace_gate["expected_terminals"]:
        findings.append(
            "gates.trace_correlation.joined_terminals must equal expected_terminals"
        )
    overhead = trace_gate["trace_overhead_percent"]
    if overhead is None or overhead < 0:
        findings.append(
            "gates.trace_correlation.trace_overhead_percent must be measured and nonnegative"
        )
    return findings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    parser.add_argument(
        "--publication-gate",
        action="store_true",
        help="require all framework graduation evidence, not only valid structure",
    )
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
    print(f"RL validation record passed ({mode}; schema={SCHEMA}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
