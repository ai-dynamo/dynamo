# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract dpkg package information from a container image."""

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_HELPER_SCRIPT_PATH = Path(__file__).resolve().parent / "helpers" / "dpkg_helper.py"


def extract_dpkg(
    image: str,
    docker_cmd: str = "docker",
    verbose: bool = False,
) -> list[dict[str, str]]:
    """Extract dpkg package attributions from a container image.

    Returns a list of dicts with keys: package_name, version, type, spdx_license
    """
    cmd = [
        docker_cmd,
        "run",
        "--rm",
        "--entrypoint", "python3",
        "-v",
        f"{_HELPER_SCRIPT_PATH}:/tmp/dpkg_helper.py:ro",
        image,
        "/tmp/dpkg_helper.py",
    ]
    if verbose:
        log.info("Running: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        # Exit 127 means python3 not found — fall back to shell-based dpkg-query
        if result.returncode == 127:
            log.warning("python3 not found in %s, falling back to shell-based dpkg extraction (no license info)", image)
            return _extract_dpkg_shell(image, docker_cmd, verbose)
        log.error(
            "dpkg extraction failed (exit %d): %s", result.returncode, result.stderr
        )
        raise RuntimeError(f"dpkg extraction failed: {result.stderr}")

    packages = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 2)
        if len(parts) != 3:
            if verbose:
                log.warning("Skipping malformed line: %r", line)
            continue
        pkg_name, version, spdx_license = parts
        packages.append(
            {
                "package_name": pkg_name,
                "version": version,
                "type": "dpkg",
                "spdx_license": spdx_license,
            }
        )

    if verbose:
        log.info("Extracted %d dpkg packages", len(packages))

    return packages


def _extract_dpkg_shell(
    image: str,
    docker_cmd: str = "docker",
    verbose: bool = False,
) -> list[dict[str, str]]:
    """Fallback: extract dpkg packages via shell when python3 is unavailable.

    License info will be UNKNOWN for all packages.
    """
    cmd = [
        docker_cmd,
        "run",
        "--rm",
        "--entrypoint", "sh",
        image,
        "-c", "dpkg-query -W -f='${Package}\\t${Version}\\n'",
    ]
    if verbose:
        log.info("Running (shell fallback): %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        log.error(
            "dpkg shell extraction failed (exit %d): %s", result.returncode, result.stderr
        )
        raise RuntimeError(f"dpkg shell extraction failed: {result.stderr}")

    packages = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        pkg_name, version = parts
        packages.append(
            {
                "package_name": pkg_name,
                "version": version,
                "type": "dpkg",
                "spdx_license": "UNKNOWN",
            }
        )

    if verbose:
        log.info("Extracted %d dpkg packages (shell fallback)", len(packages))

    return packages
