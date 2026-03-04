#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Profiler status file management.

Provides utilities for writing profiler status files.
"""

import logging
import os
import time
from enum import Enum

import yaml

logger = logging.getLogger(__name__)


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
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
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
    except Exception as e:
        logger.warning("Failed to write profiler status file: %s", e)
