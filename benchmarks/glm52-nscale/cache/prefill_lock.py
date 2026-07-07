#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acquire and release the evaluator's global campaign lock for cache prefill."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_owner(lock_dir: Path) -> dict[str, Any]:
    payload = json.loads((lock_dir / "owner.json").read_text())
    if not isinstance(payload, dict):
        raise RuntimeError("campaign lock owner is not an object")
    return payload


def acquire(lock_dir: Path, invocation_id: str, suite: str, state_dir: Path) -> None:
    owner = {
        "schema_version": 1,
        "operation": "swebench-cache-prefill",
        "suite": suite,
        "state_dir": str(state_dir),
        "invocation_id": invocation_id,
        "acquired_at": utc_now(),
    }
    staging = lock_dir.with_name(f".{lock_dir.name}.{invocation_id}.preparing")
    if lock_dir.exists():
        raise RuntimeError(
            f"campaign lock is already held: {json.dumps(read_owner(lock_dir), sort_keys=True)}"
        )
    try:
        staging.mkdir(mode=0o700)
        owner_path = staging / "owner.json"
        owner_path.write_text(json.dumps(owner, indent=2, sort_keys=True) + "\n")
        owner_path.chmod(0o400)
        try:
            os.rename(staging, lock_dir)
        except OSError as error:
            if lock_dir.exists():
                raise RuntimeError(
                    "campaign lock was acquired by another invocation: "
                    f"{json.dumps(read_owner(lock_dir), sort_keys=True)}"
                ) from error
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def release(lock_dir: Path, invocation_id: str) -> None:
    owner = read_owner(lock_dir)
    if owner.get("invocation_id") != invocation_id:
        raise RuntimeError("refusing to release a campaign lock owned by another invocation")
    shutil.rmtree(lock_dir)


def record_exit(
    lock_dir: Path, invocation_id: str, status_path: Path, exit_code: int
) -> None:
    owner = read_owner(lock_dir)
    if owner.get("invocation_id") != invocation_id:
        raise RuntimeError("refusing to record exit for another lock owner")
    payload = json.loads(status_path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError("prefill status is not an object")
    terminal_success = payload.get("state") == "complete" and exit_code == 0
    if payload.get("state") != "failed" and not terminal_success:
        payload.update(
            {
                "state": "failed",
                "updated_at": utc_now(),
                "process_exit_code": exit_code,
                "error": (
                    f"prefill process exited with status {exit_code} "
                    "before publishing terminal state"
                ),
            }
        )
        temporary = status_path.with_name(
            f".{status_path.name}.{invocation_id}.tmp"
        )
        try:
            temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            temporary.chmod(0o644)
            temporary.replace(status_path)
        finally:
            temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="operation", required=True)
    acquire_parser = subparsers.add_parser("acquire")
    acquire_parser.add_argument("--lock-dir", required=True, type=Path)
    acquire_parser.add_argument("--invocation-id", required=True)
    acquire_parser.add_argument("--suite", required=True)
    acquire_parser.add_argument("--state-dir", required=True, type=Path)
    release_parser = subparsers.add_parser("release")
    release_parser.add_argument("--lock-dir", required=True, type=Path)
    release_parser.add_argument("--invocation-id", required=True)
    exit_parser = subparsers.add_parser("record-exit")
    exit_parser.add_argument("--lock-dir", required=True, type=Path)
    exit_parser.add_argument("--invocation-id", required=True)
    exit_parser.add_argument("--status", required=True, type=Path)
    exit_parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()
    if args.operation == "acquire":
        acquire(args.lock_dir, args.invocation_id, args.suite, args.state_dir)
    elif args.operation == "release":
        release(args.lock_dir, args.invocation_id)
    else:
        record_exit(
            args.lock_dir,
            args.invocation_id,
            args.status,
            args.exit_code,
        )


if __name__ == "__main__":
    main()
