# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import logging
import platform
import sys
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information.

    Returns:
        Dictionary containing platform, architecture, processor, hostname,
        and operating system details.

    Note:
        Gracefully handles errors by returning partial information.
    """
    info: Dict[str, Any] = {}

    try:
        info["platform"] = platform.platform()
    except Exception as e:
        logger.warning(f"Failed to get platform: {e}")
        info["platform"] = "unknown"

    try:
        info["architecture"] = platform.architecture()
    except Exception as e:
        logger.warning(f"Failed to get architecture: {e}")
        info["architecture"] = ("unknown", "unknown")

    try:
        info["processor"] = platform.processor() or "unknown"
    except Exception as e:
        logger.warning(f"Failed to get processor: {e}")
        info["processor"] = "unknown"

    try:
        info["hostname"] = platform.node()
    except Exception as e:
        logger.warning(f"Failed to get hostname: {e}")
        info["hostname"] = "unknown"

    try:
        info["os_name"] = platform.system()
        info["os_release"] = platform.release()
        info["os_version"] = platform.version()
    except Exception as e:
        logger.warning(f"Failed to get OS details: {e}")

    return info


def get_runtime_info() -> Dict[str, Any]:
    """
    Get Python runtime information.

    Returns:
        Dictionary containing Python version, executable path, and command-line arguments.

    Note:
        Gracefully handles errors by returning partial information.
    """
    info: Dict[str, Any] = {}

    try:
        info["python_version"] = sys.version
        info["python_version_info"] = {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        }
    except Exception as e:
        logger.warning(f"Failed to get Python version: {e}")
        info["python_version"] = "unknown"

    try:
        info["python_executable"] = sys.executable
    except Exception as e:
        logger.warning(f"Failed to get Python executable: {e}")
        info["python_executable"] = "unknown"

    try:
        info["command_line_args"] = sys.argv
    except Exception as e:
        logger.warning(f"Failed to get command-line args: {e}")
        info["command_line_args"] = []

    return info


def get_gpu_info() -> Optional[Dict[str, Any]]:
    """
    Get GPU information if available.

    Returns:
        Dictionary containing GPU details if available, None otherwise.
        Attempts to use nvidia-smi via subprocess.

    Note:
        This is a best-effort function and returns None if GPU info cannot be obtained.
    """
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().split("\n")
            gpus = []
            for line in gpu_lines:
                if line:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        gpus.append(
                            {
                                "name": parts[0],
                                "driver_version": parts[1],
                                "memory_total": parts[2],
                            }
                        )
            return {"gpus": gpus, "count": len(gpus)} if gpus else None
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Failed to get GPU info: {e}")

    return None


def get_package_info() -> Optional[Dict[str, Any]]:
    """
    Get package information.

    Returns:
        Dictionary containing installed packages and their versions.
    """

    packages = {}
    for package in importlib.metadata.distributions():
        packages[package.name] = package.version

    return packages
