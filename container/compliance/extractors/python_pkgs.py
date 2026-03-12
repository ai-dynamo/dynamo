# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract Python package information from a container image."""

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)

_HELPER_SCRIPT_PATH = Path(__file__).resolve().parent / "helpers" / "python_helper.py"


def extract_python(
    image: str,
    docker_cmd: str = "docker",
    verbose: bool = False,
) -> list[dict[str, str]]:
    """Extract Python package attributions from a container image.

    Returns a list of dicts with keys: package_name, version, type, spdx_license
    """
    cmd = [
        docker_cmd,
        "run",
        "--rm",
        "--entrypoint", "python3",
        "-v",
        f"{_HELPER_SCRIPT_PATH}:/tmp/python_helper.py:ro",
        image,
        "/tmp/python_helper.py",
    ]
    if verbose:
        log.info("Running: %s", " ".join(cmd))

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        # Exit 127 means python3 not found — no Python packages in this image
        if result.returncode == 127:
            log.warning("python3 not found in %s, skipping Python package extraction", image)
            return []
        log.error(
            "Python extraction failed (exit %d): %s",
            result.returncode,
            result.stderr,
        )
        raise RuntimeError(f"Python extraction failed: {result.stderr}")

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
                "type": "python",
                "spdx_license": spdx_license,
            }
        )

    if verbose:
        log.info("Extracted %d Python packages", len(packages))

    return packages
