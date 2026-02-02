#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profiler status file management.

Provides utilities for writing and checking profiler status files.
"""

import argparse
import os
import sys
import time
from enum import Enum
from pathlib import Path

import yaml


class ProfilerStatus(str, Enum):
    """Profiler execution status."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


STATUS_FILE_NAME = "profiler_status.yaml"


def write_profiler_status(
    output_dir: str,
    status: ProfilerStatus,
    message: str = "",
    error: str = "",
    outputs: dict | None = None,
) -> None:
    """
    Write profiler status file.

    Args:
        output_dir: Output directory path
        status: Status enum value
        message: Optional status message
        error: Optional error message (for failed status)
        outputs: Optional dict of output files (for success status)
    """
    status_file = os.path.join(output_dir, STATUS_FILE_NAME)
    status_data = {
        "status": status.value,
        "timestamp": str(
            os.path.getmtime(status_file) if os.path.exists(status_file) else ""
        ),
    }
    if message:
        status_data["message"] = message
    if error:
        status_data["error"] = error
    if outputs:
        status_data["outputs"] = outputs

    try:
        with open(status_file, "w") as f:
            yaml.safe_dump(status_data, f, sort_keys=False)
    except Exception:
        pass


def check_profiler_status(output_dir: str, timeout_seconds: int = 120) -> int:
    """
    Check profiler status file and return exit code.

    Args:
        output_dir: Path to profiler output directory
        timeout_seconds: How long to wait for status file (default: 120s)

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    status_file = Path(output_dir) / STATUS_FILE_NAME
    start_time = time.time()

    # Wait for status file
    while not status_file.exists():
        if time.time() - start_time > timeout_seconds:
            print(f"ERROR: Status file not found after {timeout_seconds}s")
            return 1
        time.sleep(2)

    try:
        with open(status_file, "r") as f:
            status_data = yaml.safe_load(f)

        if not status_data or "status" not in status_data:
            print("ERROR: Invalid status file format")
            return 1

        status_str = status_data["status"]
        message = status_data.get("message", "")
        error = status_data.get("error", "")

        try:
            status = ProfilerStatus(status_str)
        except ValueError:
            print(f"ERROR: Unknown status: {status_str}")
            return 1

        if status == ProfilerStatus.SUCCESS:
            print(f"Profiler succeeded: {message}")
            return 0
        elif status == ProfilerStatus.FAILED:
            print(f"Profiler failed: {error or message}")
            return 1
        else:  # RUNNING
            print("ERROR: Profiler still running (unexpected)")
            return 1

    except Exception as e:
        print(f"ERROR: Failed to read status file: {e}")
        return 1


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Check profiler status file")
    parser.add_argument("--output-dir", required=True, help="Profiler output directory")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout in seconds")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print(f"ERROR: Directory not found: {args.output_dir}")
        sys.exit(2)

    exit_code = check_profiler_status(args.output_dir, args.timeout)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
