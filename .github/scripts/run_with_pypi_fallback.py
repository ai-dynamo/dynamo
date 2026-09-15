#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import subprocess
import sys

ARTIFACTORY_PREFIX = "https://artifactory.nvidia.com/"
CLOUDFRONT_HOST = "d1j32scj9xxftt.cloudfront.net"
PUBLIC_PYPI = "https://pypi.org/simple/"


def run_command(command: list[str], env=None) -> tuple[int, bool]:
    """Stream command output and detect the known Artifactory CDN failure."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
    )
    assert process.stdout is not None

    saw_cloudfront_host = False
    saw_access_denied = False
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        normalized = line.lower()
        saw_cloudfront_host |= CLOUDFRONT_HOST in normalized
        saw_access_denied |= "403" in normalized or "request blocked" in normalized

    return process.wait(), saw_cloudfront_host and saw_access_denied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry an Artifactory CDN failure against public PyPI"
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command:
        parser.error("a command is required")
    return args


def main() -> int:
    args = parse_args()
    primary_exit_code, retryable_failure = run_command(args.command)
    uses_artifactory = os.environ.get("PIP_INDEX_URL", "").startswith(
        ARTIFACTORY_PREFIX
    )
    if primary_exit_code == 0 or not uses_artifactory or not retryable_failure:
        return primary_exit_code

    print(
        "::warning title=PyPI fallback::"
        "Artifactory CDN returned 403; retrying with public PyPI",
        flush=True,
    )

    fallback_env = os.environ.copy()
    fallback_env["PIP_INDEX_URL"] = PUBLIC_PYPI
    fallback_env["UV_DEFAULT_INDEX"] = PUBLIC_PYPI
    fallback_exit_code, _ = run_command(args.command, env=fallback_env)
    return fallback_exit_code


if __name__ == "__main__":
    sys.exit(main())
