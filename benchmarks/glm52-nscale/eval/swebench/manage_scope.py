#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create and validate the exact instance set owned by one SWE-bench run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def load_dataset_ids(dataset: Path, expected: int) -> list[str]:
    ids: list[str] = []
    for line_number, line in enumerate(dataset.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"{dataset}:{line_number}: invalid instance_id")
        ids.append(instance_id)
    if len(ids) != expected:
        raise ValueError(f"expected {expected} dataset rows, got {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("dataset instance_id values are not unique")
    return ids


def select_ids(
    dataset_ids: list[str], filter_spec: str = "", slice_spec: str = ""
) -> list[str]:
    try:
        pattern = re.compile(filter_spec)
    except re.error as error:
        raise ValueError(f"invalid INSTANCE_FILTER: {error}") from error
    selected = [
        instance_id for instance_id in dataset_ids if pattern.match(instance_id)
    ]
    if slice_spec:
        parts = slice_spec.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("INSTANCE_SLICE must have start:stop[:step] syntax")
        try:
            values = [int(part) if part else None for part in parts]
        except ValueError as error:
            raise ValueError("INSTANCE_SLICE values must be integers") from error
        selected = selected[slice(*values)]
    if not selected:
        raise ValueError("instance selection is empty")
    return selected


def build_scope(
    dataset_ids: list[str], filter_spec: str = "", slice_spec: str = ""
) -> dict[str, Any]:
    target_ids = select_ids(dataset_ids, filter_spec, slice_spec)
    full_run = not filter_spec and not slice_spec
    return {
        "schema_version": 1,
        "scope": "full" if full_run else "smoke",
        "full_run": full_run,
        "dataset_instances": len(dataset_ids),
        "target_instances": len(target_ids),
        "instance_filter": filter_spec or None,
        "instance_slice": slice_spec or None,
        "target_ids": target_ids,
    }


def validate_scope(
    scope: dict[str, Any], dataset_ids: list[str], expected: int
) -> dict[str, Any]:
    required = {
        "schema_version",
        "scope",
        "full_run",
        "dataset_instances",
        "target_instances",
        "instance_filter",
        "instance_slice",
        "target_ids",
    }
    missing_fields = sorted(required - scope.keys())
    if missing_fields:
        raise ValueError(f"scope is missing fields: {missing_fields}")
    if scope["schema_version"] != 1:
        raise ValueError(f"unsupported scope schema: {scope['schema_version']}")
    if scope["dataset_instances"] != expected:
        raise ValueError(
            f"scope dataset count is {scope['dataset_instances']}, expected {expected}"
        )
    target_ids = scope["target_ids"]
    if not isinstance(target_ids, list) or not all(
        isinstance(instance_id, str) and instance_id for instance_id in target_ids
    ):
        raise ValueError("scope target_ids must be a list of non-empty strings")
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("scope target_ids are not unique")
    if scope["target_instances"] != len(target_ids):
        raise ValueError("scope target_instances does not match target_ids")
    expected_scope = build_scope(
        dataset_ids,
        scope["instance_filter"] or "",
        scope["instance_slice"] or "",
    )
    if scope != expected_scope:
        raise ValueError("scope does not match its dataset/filter/slice inputs")
    return scope


def load_scope(scope_path: Path, dataset: Path, expected: int) -> dict[str, Any]:
    dataset_ids = load_dataset_ids(dataset, expected)
    scope = json.loads(scope_path.read_text())
    if not isinstance(scope, dict):
        raise TypeError("scope must be a JSON object")
    return validate_scope(scope, dataset_ids, expected)


def prepare_scope(args: argparse.Namespace) -> None:
    dataset_ids = load_dataset_ids(args.dataset, args.expected)
    requested = build_scope(dataset_ids, args.instance_filter, args.instance_slice)
    if args.output.exists():
        current = load_scope(args.output, args.dataset, args.expected)
        if args.reuse_existing_if_unselected and not (
            args.instance_filter or args.instance_slice
        ):
            return
        if current != requested:
            raise SystemExit(
                "run scope differs from the existing scope; use a new run-name"
            )
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(requested, indent=2) + "\n")


def print_target_ids(args: argparse.Namespace) -> None:
    scope = load_scope(args.scope, args.dataset, args.expected)
    for instance_id in scope["target_ids"]:
        print(instance_id)


def print_selector(args: argparse.Namespace) -> None:
    scope = load_scope(args.scope, args.dataset, args.expected)
    print(scope[args.field] or "")


def print_batch_filters(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    scope = load_scope(args.scope, args.dataset, args.expected)
    target_ids = scope["target_ids"]
    for offset in range(0, len(target_ids), args.batch_size):
        batch = target_ids[offset : offset + args.batch_size]
        print("^(?:" + "|".join(re.escape(instance_id) for instance_id in batch) + ")$")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--dataset", required=True, type=Path)
    prepare.add_argument("--expected", required=True, type=int)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--filter", dest="instance_filter", default="")
    prepare.add_argument("--slice", dest="instance_slice", default="")
    prepare.add_argument("--reuse-existing-if-unselected", action="store_true")
    prepare.set_defaults(func=prepare_scope)

    list_ids = subparsers.add_parser("list")
    list_ids.add_argument("--scope", required=True, type=Path)
    list_ids.add_argument("--dataset", required=True, type=Path)
    list_ids.add_argument("--expected", required=True, type=int)
    list_ids.set_defaults(func=print_target_ids)

    selector = subparsers.add_parser("selector")
    selector.add_argument("--scope", required=True, type=Path)
    selector.add_argument("--dataset", required=True, type=Path)
    selector.add_argument("--expected", required=True, type=int)
    selector.add_argument(
        "--field", choices=("instance_filter", "instance_slice"), required=True
    )
    selector.set_defaults(func=print_selector)

    batches = subparsers.add_parser("batch-filters")
    batches.add_argument("--scope", required=True, type=Path)
    batches.add_argument("--dataset", required=True, type=Path)
    batches.add_argument("--expected", required=True, type=int)
    batches.add_argument("--batch-size", required=True, type=int)
    batches.set_defaults(func=print_batch_filters)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
