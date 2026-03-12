# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract Python package information from a container image."""

import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)

# This script runs INSIDE the container. It must be fully self-contained
# with zero external dependencies (only Python stdlib).
_HELPER_SCRIPT = r'''
import importlib.metadata
import sys

# Conservative classifier -> SPDX mapping
_CLASSIFIER_MAP = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: BSD License": "BSD-3-Clause",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: GNU General Public License v2 (GPLv2)": "GPL-2.0-only",
    "License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)": "GPL-2.0-or-later",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)": "GPL-3.0-only",
    "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)": "GPL-3.0-or-later",
    "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)": "LGPL-2.0-only",
    "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)": "LGPL-2.0-or-later",
    "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)": "LGPL-3.0-only",
    "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)": "LGPL-3.0-or-later",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: Boost Software License 1.0 (BSL-1.0)": "BSL-1.0",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
    "License :: OSI Approved :: Artistic License": "Artistic-2.0",
    "License :: OSI Approved :: zlib/libpng License": "Zlib",
    "License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication": "CC0-1.0",
    "License :: Public Domain": "CC0-1.0",
}

# Conservative free-text license -> SPDX mapping
_LICENSE_MAP = {
    "MIT": "MIT",
    "MIT License": "MIT",
    "The MIT License": "MIT",
    "The MIT License (MIT)": "MIT",
    "Apache License 2.0": "Apache-2.0",
    "Apache License, Version 2.0": "Apache-2.0",
    "Apache Software License": "Apache-2.0",
    "Apache 2.0": "Apache-2.0",
    "Apache-2.0": "Apache-2.0",
    "BSD License": "BSD-3-Clause",
    "BSD": "BSD-3-Clause",
    "BSD-2-Clause": "BSD-2-Clause",
    "BSD-3-Clause": "BSD-3-Clause",
    "3-Clause BSD License": "BSD-3-Clause",
    "2-Clause BSD License": "BSD-2-Clause",
    "Simplified BSD License": "BSD-2-Clause",
    "New BSD License": "BSD-3-Clause",
    "ISC": "ISC",
    "ISC License": "ISC",
    "ISC License (ISCL)": "ISC",
    "MPL-2.0": "MPL-2.0",
    "Mozilla Public License 2.0": "MPL-2.0",
    "Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "PSF-2.0": "PSF-2.0",
    "Python Software Foundation License": "PSF-2.0",
    "Unlicense": "Unlicense",
    "The Unlicense": "Unlicense",
    "CC0-1.0": "CC0-1.0",
    "Public Domain": "CC0-1.0",
    "WTFPL": "WTFPL",
    "Zlib": "Zlib",
}

_LICENSE_MAP_LOWER = {k.lower(): v for k, v in _LICENSE_MAP.items()}


def get_license(dist):
    """Extract SPDX license for a distribution, conservative approach."""
    meta = dist.metadata

    # 1. PEP 639 License-Expression (already SPDX)
    license_expr = meta.get("License-Expression")
    if license_expr and license_expr.strip():
        return license_expr.strip()

    # 2. Free-text License field
    license_field = meta.get("License")
    if license_field and license_field.strip():
        val = license_field.strip()
        mapped = _LICENSE_MAP.get(val) or _LICENSE_MAP_LOWER.get(val.lower())
        if mapped:
            return mapped

    # 3. Trove classifiers
    classifiers = meta.get_all("Classifier") or []
    license_classifiers = [c for c in classifiers if c.startswith("License ::")]
    for clf in license_classifiers:
        if clf in _CLASSIFIER_MAP:
            return _CLASSIFIER_MAP[clf]

    return "UNKNOWN"


def main():
    seen = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        if not name:
            continue
        # Deduplicate (importlib.metadata can return duplicates)
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)

        version = dist.metadata["Version"] or "UNKNOWN"
        spdx = get_license(dist)
        print(f"{name}\t{version}\t{spdx}")


if __name__ == "__main__":
    main()
'''


def extract_python(
    image: str,
    docker_cmd: str = "docker",
    verbose: bool = False,
) -> list[dict[str, str]]:
    """Extract Python package attributions from a container image.

    Returns a list of dicts with keys: package_name, version, type, spdx_license
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", prefix="python_helper_", delete=False
    ) as f:
        f.write(_HELPER_SCRIPT)
        helper_path = f.name

    try:
        cmd = [
            docker_cmd,
            "run",
            "--rm",
            "-v",
            f"{helper_path}:/tmp/python_helper.py:ro",
            image,
            "python3",
            "/tmp/python_helper.py",
        ]
        if verbose:
            log.info("Running: %s", " ".join(cmd))

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        if result.returncode != 0:
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
    finally:
        Path(helper_path).unlink(missing_ok=True)
