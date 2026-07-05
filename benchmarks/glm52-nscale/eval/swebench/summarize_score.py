#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize evaluator results and fail closed on incomplete evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from manage_scope import load_scope


def _id_set(raw: dict[str, Any], field: str) -> set[str]:
    values = raw.get(field)
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(f"{field} must be a list of non-empty instance IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"{field} contains duplicate instance IDs")
    return set(values)


def _count(raw: dict[str, Any], field: str, ids: set[str]) -> int:
    value = raw.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if value != len(ids):
        raise ValueError(f"{field}={value} but its ID list contains {len(ids)} IDs")
    return value


def _normalize_swebench(
    raw: Any, target_ids: set[str]
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(raw, dict):
        raise TypeError("SWE-bench raw score must be a JSON object")
    if raw.get("schema_version") != 2:
        raise ValueError(
            f"unsupported SWE-bench report schema: {raw.get('schema_version')}"
        )
    total_instances = raw.get("total_instances")
    if (
        not isinstance(total_instances, int)
        or isinstance(total_instances, bool)
        or total_instances != len(target_ids)
    ):
        raise ValueError(
            f"evaluator report covers {total_instances} instances, "
            f"but run scope targets {len(target_ids)}"
        )

    submitted_ids = _id_set(raw, "submitted_ids")
    completed_ids = _id_set(raw, "completed_ids")
    incomplete_ids = _id_set(raw, "incomplete_ids")
    passed_ids = _id_set(raw, "resolved_ids")
    unresolved_ids = _id_set(raw, "unresolved_ids")
    empty_patch_ids = _id_set(raw, "empty_patch_ids")
    error_ids = _id_set(raw, "error_ids")

    _count(raw, "submitted_instances", submitted_ids)
    _count(raw, "completed_instances", completed_ids)
    _count(raw, "resolved_instances", passed_ids)
    _count(raw, "unresolved_instances", unresolved_ids)
    _count(raw, "empty_patch_instances", empty_patch_ids)
    _count(raw, "error_instances", error_ids)

    outcome_sets = {
        "resolved": passed_ids,
        "unresolved": unresolved_ids,
        "empty_patch": empty_patch_ids,
        "error": error_ids,
    }
    outcome_names = list(outcome_sets)
    for index, left_name in enumerate(outcome_names):
        for right_name in outcome_names[index + 1 :]:
            overlap = outcome_sets[left_name] & outcome_sets[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name} and {right_name} outcomes overlap: {sorted(overlap)}"
                )

    outcome_ids = set().union(*outcome_sets.values())
    missing_submission_ids = target_ids - submitted_ids
    unexpected_submission_ids = submitted_ids - target_ids
    unexpected_outcome_ids = outcome_ids - target_ids
    missing_evaluation_ids = target_ids - outcome_ids

    gate_failures = []
    if missing_submission_ids:
        gate_failures.append(
            f"missing {len(missing_submission_ids)} submitted instance IDs"
        )
    if unexpected_submission_ids:
        gate_failures.append(
            f"found {len(unexpected_submission_ids)} unexpected submitted instance IDs"
        )
    if incomplete_ids:
        gate_failures.append(
            f"evaluator reports {len(incomplete_ids)} incomplete instances"
        )
    if error_ids:
        gate_failures.append(f"evaluator reports {len(error_ids)} error instances")
    if unexpected_outcome_ids:
        gate_failures.append(
            f"evaluator reports {len(unexpected_outcome_ids)} unexpected outcome IDs"
        )
    if missing_evaluation_ids:
        gate_failures.append(
            f"evaluator has no outcome for {len(missing_evaluation_ids)} target IDs"
        )

    expected_incomplete = target_ids - submitted_ids
    if incomplete_ids != expected_incomplete:
        raise ValueError(
            "incomplete_ids do not match the target IDs absent from submitted_ids"
        )
    successful_ids = passed_ids | unresolved_ids
    if not successful_ids <= completed_ids:
        raise ValueError("resolved/unresolved IDs must be present in completed_ids")
    if not completed_ids <= successful_ids | error_ids:
        raise ValueError(
            "completed_ids contain IDs without a resolved, unresolved, or error outcome"
        )

    normalized = {
        "submitted_ids": submitted_ids,
        "completed_ids": completed_ids,
        "passed_ids": passed_ids,
        "unresolved_ids": unresolved_ids,
        "empty_patch_ids": empty_patch_ids,
        "error_ids": error_ids,
        "incomplete_ids": incomplete_ids,
        "missing_submission_ids": missing_submission_ids,
        "unexpected_submission_ids": unexpected_submission_ids,
        "missing_evaluation_ids": missing_evaluation_ids,
        "unexpected_evaluation_ids": unexpected_outcome_ids,
    }
    return normalized, gate_failures


def _load_pro_statuses(status_dir: Path) -> tuple[set[str], set[str]]:
    completed_ids: set[str] = set()
    error_ids: set[str] = set()
    for path in sorted(status_dir.glob("*.json")):
        status = json.loads(path.read_text())
        if not isinstance(status, dict):
            raise TypeError(f"{path}: status must be a JSON object")
        instance_id = status.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"{path}: invalid instance_id")
        if instance_id in completed_ids | error_ids:
            raise ValueError(f"duplicate Pro evaluation status for {instance_id}")
        if status.get("status") == "completed":
            completed_ids.add(instance_id)
        elif status.get("status") == "error":
            error_ids.add(instance_id)
        else:
            raise ValueError(f"{path}: invalid status {status.get('status')!r}")
    return completed_ids, error_ids


def _normalize_pro(
    raw: Any, target_ids: set[str], status_dir: Path
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(raw, dict):
        raise TypeError("SWE-bench Pro raw score must be a JSON object")
    if not all(
        isinstance(instance_id, str) and instance_id and isinstance(passed, bool)
        for instance_id, passed in raw.items()
    ):
        raise ValueError("Pro raw score must map non-empty instance IDs to booleans")

    submitted_ids = set(raw)
    passed_ids = {instance_id for instance_id, passed in raw.items() if passed}
    unresolved_ids = submitted_ids - passed_ids
    completed_ids, error_ids = _load_pro_statuses(status_dir)
    status_ids = completed_ids | error_ids
    missing_submission_ids = target_ids - submitted_ids
    unexpected_submission_ids = submitted_ids - target_ids
    missing_evaluation_ids = target_ids - status_ids
    unexpected_status_ids = status_ids - target_ids

    gate_failures = []
    if missing_submission_ids:
        gate_failures.append(
            f"missing {len(missing_submission_ids)} Pro result instance IDs"
        )
    if unexpected_submission_ids:
        gate_failures.append(
            f"found {len(unexpected_submission_ids)} unexpected Pro result instance IDs"
        )
    if missing_evaluation_ids:
        gate_failures.append(
            f"missing evaluator status for {len(missing_evaluation_ids)} Pro target IDs"
        )
    if unexpected_status_ids:
        gate_failures.append(
            f"found {len(unexpected_status_ids)} unexpected Pro evaluator status IDs"
        )
    if error_ids:
        gate_failures.append(f"Pro evaluator reports {len(error_ids)} error instances")

    normalized = {
        "submitted_ids": submitted_ids,
        "completed_ids": completed_ids,
        "passed_ids": passed_ids,
        "unresolved_ids": unresolved_ids - error_ids,
        "empty_patch_ids": set(),
        "error_ids": error_ids,
        "incomplete_ids": set(),
        "missing_submission_ids": missing_submission_ids,
        "unexpected_submission_ids": unexpected_submission_ids,
        "missing_evaluation_ids": missing_evaluation_ids,
        "unexpected_evaluation_ids": unexpected_status_ids,
    }
    return normalized, gate_failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("swebench", "pro"), required=True)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--expected", required=True, type=int)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--scope", required=True, type=Path)
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--status-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    scope = load_scope(args.scope, args.dataset, args.expected)
    target_ids = set(scope["target_ids"])
    raw = json.loads(args.raw.read_text())
    if args.kind == "swebench":
        normalized, gate_failures = _normalize_swebench(raw, target_ids)
    else:
        if args.status_dir is None:
            parser.error("--status-dir is required for Pro results")
        normalized, gate_failures = _normalize_pro(raw, target_ids, args.status_dir)

    passed_ids = normalized["passed_ids"]
    failed_ids = target_ids - passed_ids
    summary = {
        "suite": args.suite,
        "evaluator": (
            "SWE-bench Pro public" if args.kind == "pro" else "SWE-bench 4.1.0"
        ),
        "scope": scope["scope"],
        "full_run": scope["full_run"],
        "dataset_instances": args.expected,
        "expected_instances": scope["target_instances"],
        "target_instances": scope["target_instances"],
        "excluded_dataset_instances": args.expected - scope["target_instances"],
        "submitted_instances": len(normalized["submitted_ids"]),
        "completed_instances": len(normalized["completed_ids"]),
        "passed_instances": len(passed_ids),
        "failed_instances": len(failed_ids),
        "missing_instances": len(normalized["missing_submission_ids"]),
        "missing_ids": sorted(normalized["missing_submission_ids"]),
        "unexpected_ids": sorted(normalized["unexpected_submission_ids"]),
        "incomplete_evaluation_ids": sorted(normalized["incomplete_ids"]),
        "missing_evaluation_ids": sorted(normalized["missing_evaluation_ids"]),
        "unexpected_evaluation_ids": sorted(normalized["unexpected_evaluation_ids"]),
        "evaluation_error_ids": sorted(normalized["error_ids"]),
        "unresolved_ids": sorted(normalized["unresolved_ids"]),
        "empty_patch_ids": sorted(normalized["empty_patch_ids"]),
        "score_on_submitted": (
            len(passed_ids) / len(normalized["submitted_ids"])
            if normalized["submitted_ids"]
            else 0.0
        ),
        "score_on_scope": len(passed_ids) / len(target_ids),
        "benchmark_score": (
            len(passed_ids) / args.expected
            if scope["full_run"] and not gate_failures
            else None
        ),
        "complete": not gate_failures,
        "gate_failures": gate_failures,
        "passed_ids": sorted(passed_ids),
        "failed_ids": sorted(failed_ids),
        "raw_result": str(args.raw),
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    if args.require_complete and gate_failures:
        raise SystemExit(
            "evaluation completeness gate failed: " + "; ".join(gate_failures)
        )


if __name__ == "__main__":
    main()
