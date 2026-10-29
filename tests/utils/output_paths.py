# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility for resolving test output paths.

This module provides centralized logic for determining where test output
(logs, temporary files, etc.) should be written. The goal is to keep test
output out of the git working tree by default.
"""

import glob
import os
import shutil
import tempfile
from pathlib import Path
from typing import Union


def resolve_test_output_path(path: Union[str, Path]) -> str:
    """Resolve a test output path to an absolute path.

    This function ensures test output is written to a dedicated location
    rather than cluttering the git working tree. The behavior matches
    the existing ManagedProcess implementation.

    Args:
        path: A relative or absolute path for test output.

    Returns:
        An absolute path. If the input is already absolute, it's returned
        unchanged. If relative, it's resolved under the test output root
        directory. Characters in relative paths that GitHub artifact
        uploads reject (e.g. `*`) are percent-encoded.

    Environment Variables:
        DYN_TEST_OUTPUT_PATH: Override the default test output root.
                              Defaults to /tmp/dynamo_tests/.

    Examples:
        >>> resolve_test_output_path("/absolute/path")
        '/absolute/path'
        >>> resolve_test_output_path("test_foo")  # doctest: +SKIP
        '/tmp/dynamo_tests/test_foo'
        >>> resolve_test_output_path("test_foo[*]")  # doctest: +SKIP
        '/tmp/dynamo_tests/test_foo[%2A]'
    """
    path_str = str(path)

    if os.path.isabs(path_str):
        return path_str

    # encode special characters so GitHub accepts test log uploads
    path_str = "".join(
        # escape `%` too so encoded names cannot collide with literal escapes
        f"%{ord(char):02X}" if char in '"<>:|*?\r\n%' else char
        for char in path_str
    )

    log_root = os.environ.get(
        "DYN_TEST_OUTPUT_PATH",
        os.path.join(tempfile.gettempdir(), "dynamo_tests"),
    )
    return os.path.join(log_root, path_str)


# Executions seen per test in this process. Retries of a test (Datadog Auto Test
# Retries and Early Flake Detection, pytest-rerunfailures) run again in the same
# process, including under xdist.
_output_dir_attempts: dict[str, int] = {}


def prepare_test_output_dir(log_dir: str, test_id: str) -> None:
    """Create an empty output directory for one execution of a test.

    The first execution of ``test_id`` in this process removes stale output
    left by earlier sessions. A repeat execution moves the previous attempt's
    output to ``<log_dir>.attempt<N>`` instead of deleting it, so a failed
    attempt's logs survive a passing retry and are uploaded with it.
    """
    attempt = _output_dir_attempts.get(test_id, 0) + 1
    _output_dir_attempts[test_id] = attempt
    if attempt == 1:
        shutil.rmtree(log_dir, ignore_errors=True)
        for stale in glob.glob(f"{glob.escape(log_dir)}.attempt*"):
            shutil.rmtree(stale, ignore_errors=True)
    elif os.path.exists(log_dir):
        previous = f"{log_dir}.attempt{attempt - 1}"
        shutil.rmtree(previous, ignore_errors=True)
        os.rename(log_dir, previous)
    os.makedirs(log_dir, exist_ok=True)
