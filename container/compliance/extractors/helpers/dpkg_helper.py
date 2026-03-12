# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script runs INSIDE the container. It must be fully self-contained
# with zero external dependencies (only Python stdlib).

import os
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
            val = s[len("License:") :].strip()
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
        capture_output=True,
        text=True,
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
