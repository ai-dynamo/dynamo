# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Utility for resolving test output paths.

This module provides centralized logic for determining where test output
(logs, temporary files, etc.) should be written. The goal is to keep test
output out of the git working tree by default.
"""

import os
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
        directory.

    Environment Variables:
        DYN_TEST_OUTPUT_PATH: Override the default test output root.
                              Defaults to /tmp/dynamo_tests/.

    Examples:
        >>> resolve_test_output_path("/absolute/path")
        '/absolute/path'
        >>> resolve_test_output_path("test_foo")  # doctest: +SKIP
        '/tmp/dynamo_tests/test_foo'
    """
    path_str = str(path)

    if os.path.isabs(path_str):
        return path_str

    log_root = os.environ.get(
        "DYN_TEST_OUTPUT_PATH",
        os.path.join(tempfile.gettempdir(), "dynamo_tests"),
    )
    return os.path.join(log_root, path_str)
