#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve and verify the pinned Terminal-Bench package through Harbor Hub."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harbor.registry.client.package import PackageDatasetClient


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


async def resolve(args: argparse.Namespace) -> dict[str, Any]:
    metadata = await PackageDatasetClient().get_dataset_metadata(args.dataset)
    task_refs = [task.model_dump(mode="json") for task in metadata.task_ids]
    resolved = {
        "resolved_at": utc_now(),
        "requested_ref": args.dataset,
        "name": metadata.name,
        "version": metadata.version,
        "dataset_version_id": metadata.dataset_version_id,
        "content_hash": metadata.dataset_version_content_hash,
        "task_count": len(metadata.task_ids),
        "task_refs": task_refs,
    }

    mismatches: list[str] = []
    if metadata.dataset_version_content_hash != args.expected_content_hash:
        mismatches.append(
            "content hash "
            f"{metadata.dataset_version_content_hash!r} != {args.expected_content_hash!r}"
        )
    if metadata.dataset_version_id != args.expected_version_id:
        mismatches.append(
            f"version id {metadata.dataset_version_id!r} != {args.expected_version_id!r}"
        )
    if len(metadata.task_ids) != args.expected_tasks:
        mismatches.append(
            f"task count {len(metadata.task_ids)} != {args.expected_tasks}"
        )
    if mismatches:
        raise RuntimeError("; ".join(mismatches))
    return resolved


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True)
    result.add_argument("--expected-content-hash", required=True)
    result.add_argument("--expected-version-id", required=True)
    result.add_argument("--expected-tasks", type=int, required=True)
    result.add_argument("--output", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        resolved = asyncio.run(resolve(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"dataset verification error: {error}", file=sys.stderr)
        return 1
    if args.output:
        atomic_write_json(args.output.resolve(), resolved)
    print(
        json.dumps(
            {
                "content_hash": resolved["content_hash"],
                "dataset_version_id": resolved["dataset_version_id"],
                "task_count": resolved["task_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
