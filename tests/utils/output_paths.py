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


# File timestamps use a coarse kernel clock that can trail time.time() by a
# few milliseconds, so output just created this session must not look stale.
_STALE_OUTPUT_MARGIN_S = 1.0


def prepare_test_output_dir(log_dir: str, session_start: float) -> None:
    """Create an empty output directory for one execution of a test.

    Output last modified before ``session_start`` is left from an earlier
    session and is removed. Output written during this session belongs to an
    earlier attempt of this test, retried by Datadog or pytest-rerunfailures,
    or to a same-named test in another file or xdist worker; it is moved to the
    first free ``<log_dir>.attempt<N>`` so it is uploaded too. Checking the
    time instead of counting attempts gives the same answer in every xdist
    worker, because the workers share the output directory.
    """
    for path in [log_dir, *glob.glob(f"{glob.escape(log_dir)}.attempt*")]:
        try:
            stale = os.path.getmtime(path) < session_start - _STALE_OUTPUT_MARGIN_S
        except OSError:
            continue
        if stale:
            shutil.rmtree(path, ignore_errors=True)
    if os.path.exists(log_dir):
        attempt = 1
        while os.path.exists(f"{log_dir}.attempt{attempt}"):
            attempt += 1
        os.rename(log_dir, f"{log_dir}.attempt{attempt}")
    os.makedirs(log_dir, exist_ok=True)
