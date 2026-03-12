# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extract dpkg package information from a container image."""

import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# This script runs INSIDE the container. It must be fully self-contained
# with zero external dependencies (only Python stdlib).
_HELPER_SCRIPT = r'''
import os
import re
import subprocess
import sys

# Conservative DEP-5 license field -> SPDX mapping
_DEP5_MAP = {
    "Apache-2.0": "Apache-2.0",
    "Apache-2": "Apache-2.0",
    "Artistic-2.0": "Artistic-2.0",
    "BSD-2-clause": "BSD-2-Clause",
    "BSD-3-clause": "BSD-3-Clause",
    "BSL-1.0": "BSL-1.0",
    "CC0-1.0": "CC0-1.0",
    "Expat": "MIT",
    "GPL-2": "GPL-2.0-only",
    "GPL-2+": "GPL-2.0-or-later",
    "GPL-2.0": "GPL-2.0-only",
    "GPL-2.0+": "GPL-2.0-or-later",
    "GPL-3": "GPL-3.0-only",
    "GPL-3+": "GPL-3.0-or-later",
    "GPL-3.0": "GPL-3.0-only",
    "GPL-3.0+": "GPL-3.0-or-later",
    "ISC": "ISC",
    "LGPL-2": "LGPL-2.0-only",
    "LGPL-2+": "LGPL-2.0-or-later",
    "LGPL-2.0": "LGPL-2.0-only",
    "LGPL-2.0+": "LGPL-2.0-or-later",
    "LGPL-2.1": "LGPL-2.1-only",
    "LGPL-2.1+": "LGPL-2.1-or-later",
    "LGPL-3": "LGPL-3.0-only",
    "LGPL-3+": "LGPL-3.0-or-later",
    "LGPL-3.0": "LGPL-3.0-only",
    "LGPL-3.0+": "LGPL-3.0-or-later",
    "MIT": "MIT",
    "MPL-2.0": "MPL-2.0",
    "PSF-2": "PSF-2.0",
    "public-domain": "CC0-1.0",
    "Zlib": "Zlib",
    "OpenSSL": "OpenSSL",
    "WTFPL": "WTFPL",
}

_DEP5_MAP_LOWER = {k.lower(): v for k, v in _DEP5_MAP.items()}


def is_dep5(content):
    for line in content.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        return s.startswith("Format:")
    return False


def extract_dep5_license(content):
    """Extract the primary license from a DEP-5 copyright file."""
    licenses = set()
    for line in content.splitlines():
        s = line.strip()
        if s.startswith("License:"):
            val = s[len("License:"):].strip()
            if val:
                mapped = _DEP5_MAP.get(val) or _DEP5_MAP_LOWER.get(val.lower())
                if mapped:
                    licenses.add(mapped)
    if len(licenses) == 1:
        return licenses.pop()
    elif len(licenses) > 1:
        return " AND ".join(sorted(licenses))
    return "UNKNOWN"


def get_license_for_package(pkg_name):
    """Read /usr/share/doc/<pkg>/copyright and extract license info."""
    copyright_path = f"/usr/share/doc/{pkg_name}/copyright"
    if not os.path.isfile(copyright_path):
        return "UNKNOWN"
    try:
        with open(copyright_path, "r", errors="replace") as f:
            content = f.read()
    except (OSError, IOError):
        return "UNKNOWN"

    if not content.strip():
        return "UNKNOWN"

    if is_dep5(content):
        return extract_dep5_license(content)

    return "UNKNOWN"


def main():
    result = subprocess.run(
        ["dpkg-query", "-W", "-f=${Package}\t${Version}\n"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"ERROR: dpkg-query failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    for line in result.stdout.strip().splitlines():
        parts = line.split("\t", 1)
        if len(parts) != 2:
            continue
        pkg, version = parts
        license_id = get_license_for_package(pkg)
        print(f"{pkg}\t{version}\t{license_id}")


if __name__ == "__main__":
    main()
'''


def extract_dpkg(
    image: str,
    docker_cmd: str = "docker",
    verbose: bool = False,
) -> list[dict[str, str]]:
    """Extract dpkg package attributions from a container image.

    Returns a list of dicts with keys: package_name, version, type, spdx_license
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="dpkg_helper_", delete=False
    ) as f:
        f.write(_HELPER_SCRIPT)
        helper_path = f.name

    try:
        cmd = [
            docker_cmd, "run", "--rm",
            "-v", f"{helper_path}:/tmp/dpkg_helper.py:ro",
            image,
            "python3", "/tmp/dpkg_helper.py",
        ]
        if verbose:
            log.info("Running: %s", " ".join(cmd))

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            log.error("dpkg extraction failed (exit %d): %s", result.returncode, result.stderr)
            raise RuntimeError(f"dpkg extraction failed: {result.stderr}")

        packages = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("\t", 2)
            if len(parts) != 3:
                if verbose:
                    log.warning("Skipping malformed line: %r", line)
                continue
            pkg_name, version, spdx_license = parts
            packages.append({
                "package_name": pkg_name,
                "version": version,
                "type": "dpkg",
                "spdx_license": spdx_license,
            })

        if verbose:
            log.info("Extracted %d dpkg packages", len(packages))

        return packages
    finally:
        Path(helper_path).unlink(missing_ok=True)
