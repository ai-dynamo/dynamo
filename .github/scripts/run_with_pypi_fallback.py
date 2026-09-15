#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ARTIFACTORY_PREFIX = "https://artifactory.nvidia.com/"
CLOUDFRONT_HOST = "d1j32scj9xxftt.cloudfront.net"
PUBLIC_PYPI = "https://pypi.org/simple/"


def run_command(command: list[str], log_file: Path, env=None) -> int:
    """Run a command while streaming its combined output to stdout and a log."""
    with log_file.open("w") as output:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            output.write(line)
            output.flush()
        return process.wait()


def is_retryable_artifactory_failure(log_file: Path) -> bool:
    if not os.environ.get("PIP_INDEX_URL", "").startswith(ARTIFACTORY_PREFIX):
        return False

    output = log_file.read_text(errors="replace").lower()
    return CLOUDFRONT_HOST in output and (
        "403" in output or "request blocked" in output
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry an Artifactory CDN failure against public PyPI"
    )
    parser.add_argument("log_file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")
    return args


def main() -> int:
    args = parse_args()
    primary_exit_code = run_command(args.command, args.log_file)
    if primary_exit_code == 0 or not is_retryable_artifactory_failure(args.log_file):
        return primary_exit_code

    # Keep the retry log clean for BuildKit metrics parsing.
    failed_log = args.log_file.with_name(f"{args.log_file.name}.artifactory-failed")
    shutil.move(args.log_file, failed_log)
    print(
        "::warning title=PyPI fallback::"
        "Artifactory CDN returned 403; retrying with public PyPI",
        flush=True,
    )

    fallback_env = os.environ.copy()
    fallback_env["PIP_INDEX_URL"] = PUBLIC_PYPI
    fallback_env["UV_DEFAULT_INDEX"] = PUBLIC_PYPI
    return run_command(args.command, args.log_file, env=fallback_env)


if __name__ == "__main__":
    sys.exit(main())
