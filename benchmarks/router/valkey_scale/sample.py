# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .child import build_child_command, read_child_result, stop_process_group
from .common import REPO

def run_sample(
    args: argparse.Namespace,
    *,
    sample_index: int,
    repetition: int,
    frontend_count: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one fresh child topology and return a normalized validity record."""

    child_output_dir = (
        output_dir
        / "children"
        / (f"sample-{sample_index:02d}-rep-{repetition:02d}-frontends-{frontend_count}")
    )
    child_output_dir.mkdir(parents=True, exist_ok=False)
    command = build_child_command(args, frontend_count, child_output_dir)
    command_path = output_dir / "commands" / f"sample-{sample_index:02d}.sh"
    command_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(command) + "\n"
    )
    log_path = output_dir / "logs" / f"sample-{sample_index:02d}.log"
    started_at = datetime.now().astimezone().isoformat()
    started = time.monotonic()
    sample: dict[str, Any] = {
        "sample_index": sample_index,
        "repetition": repetition,
        "frontend_count": frontend_count,
        "started_at": started_at,
        "child_output_dir": str(child_output_dir),
        "command_path": str(command_path),
        "log_path": str(log_path),
        "command": command,
        "valid": False,
        "validation_errors": [],
    }

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=os.environ.copy(),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            try:
                returncode = process.wait()
            except BaseException:
                stop_process_group(process)
                raise
    except KeyboardInterrupt:
        raise
    except Exception as error:
        sample["validation_errors"].append(
            f"could not start or wait for child harness: {type(error).__name__}: {error}"
        )
        sample["child_returncode"] = None
    else:
        sample["child_returncode"] = returncode
        if returncode != 0:
            sample["validation_errors"].append(
                f"child harness exited with return code {returncode}"
            )
        normalized, errors = read_child_result(
            child_output_dir, frontend_count=frontend_count, args=args
        )
        sample["validation_errors"].extend(errors)
        if normalized is not None:
            sample.update(normalized)

    sample["elapsed_seconds"] = time.monotonic() - started
    sample["finished_at"] = datetime.now().astimezone().isoformat()
    sample["valid"] = not sample["validation_errors"]
    return sample


def require_consistent_input_dataset(
    sample: dict[str, Any], prior_samples: Sequence[Mapping[str, Any]]
) -> None:
    """Invalidate a sample whose generated aiperf input differs from the campaign."""

    prior_hashes = {
        prior.get("aiperf_input_sha256")
        for prior in prior_samples
        if isinstance(prior.get("aiperf_input_sha256"), str)
    }
    current_hash = sample.get("aiperf_input_sha256")
    if (
        sample.get("valid") is True
        and prior_hashes
        and current_hash not in prior_hashes
    ):
        sample["validation_errors"].append(
            "aiperf input dataset SHA-256 differs from earlier scale samples"
        )
        sample["valid"] = False


def require_consistent_child_provenance(
    sample: dict[str, Any], prior_samples: Sequence[Mapping[str, Any]]
) -> None:
    """Invalidate a sample if its executable/harness identity changed."""

    prior_records = [
        prior.get("child_provenance")
        for prior in prior_samples
        if isinstance(prior.get("child_provenance"), Mapping)
    ]
    current_record = sample.get("child_provenance")
    if (
        sample.get("valid") is True
        and prior_records
        and current_record != prior_records[0]
    ):
        sample["validation_errors"].append(
            "child benchmark provenance differs from earlier scale samples"
        )
        sample["valid"] = False
