# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import LOGICAL_MOCKER_WORKERS, PROTECTED_HARNESS_ARGUMENTS, REPO

def validate_args(args: argparse.Namespace) -> None:
    args.harness = args.harness.expanduser().resolve()
    # Do not call Path.resolve() here: .venv/bin/python is usually a symlink
    # into uv's managed interpreter. Invoking its resolved target loses the
    # venv prefix and therefore the Dynamo/aiohttp dependencies. abspath
    # normalizes a relative user value without dereferencing that launcher.
    args.python = Path(os.path.abspath(args.python.expanduser()))
    if args.output_dir is None:
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        args.output_dir = REPO / "bench" / "results" / f"valkey-frontend-scale-{stamp}"
    else:
        args.output_dir = args.output_dir.expanduser().resolve()

    if not args.harness.is_file():
        raise FileNotFoundError(f"child harness was not found: {args.harness}")
    if not args.python.is_file() or not os.access(args.python, os.X_OK):
        raise FileNotFoundError(f"Python interpreter is not executable: {args.python}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory must be empty to avoid mixed artifacts: {args.output_dir}"
        )
    if args.requests < args.concurrency:
        raise ValueError("--requests must be at least --concurrency")
    if args.mocker_processes > LOGICAL_MOCKER_WORKERS:
        raise ValueError(
            f"--mocker-processes must be in 1..={LOGICAL_MOCKER_WORKERS}; "
            f"got {args.mocker_processes}"
        )
    if not 10_000 <= args.valkey_admission_lease_ms <= 600_000:
        raise ValueError("--valkey-admission-lease-ms must be in 10000..=600000")
    if args.valkey_gc_interval_ms != 0 and not (
        1_000 <= args.valkey_gc_interval_ms <= 86_400_000
    ):
        raise ValueError(
            "--valkey-gc-interval-ms must be 0 (disabled) or in 1000..=86400000"
        )
    if args.valkey_gc_inspection_budget > 1_048_576:
        raise ValueError("--valkey-gc-inspection-budget must be in 1..=1048576")
    if args.replica_ready_timeout <= 0:
        raise ValueError("--replica-ready-timeout must be greater than zero")
    for extra_argument in args.harness_extra_arg:
        if not extra_argument.startswith("-"):
            raise ValueError(
                "--harness-extra-arg must be one option beginning with '-', "
                f"got {extra_argument!r}"
            )
        option = extra_argument.split("=", 1)[0]
        if option in PROTECTED_HARNESS_ARGUMENTS:
            raise ValueError(
                f"{option} is controlled by the scale driver and cannot be forwarded"
            )


def prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "children").mkdir()
    (output_dir / "commands").mkdir()
    (output_dir / "logs").mkdir()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=REPO,
            capture_output=True,
            check=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout)


def python_runtime_record() -> dict[str, Any]:
    return {
        "executable": sys.executable,
        "resolved_executable": str(Path(sys.executable).resolve(strict=True)),
        "version": sys.version,
        "implementation": sys.implementation.name,
    }
