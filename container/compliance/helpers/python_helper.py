# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script runs INSIDE the container (local mode) or against a mounted
# filesystem root (--root /target mode for BuildKit extraction).
# It must be fully self-contained with zero external dependencies (only Python stdlib).

import argparse
import glob
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
    parser = argparse.ArgumentParser(
        description="Extract Python package info (stdlib only)"
    )
    parser.add_argument(
        "--root",
        default="/",
        help="Filesystem root to inspect (default: /, i.e. running system)",
    )
    args = parser.parse_args()
    root = args.root.rstrip("/") or "/"

    if root != "/":
        # BuildKit mode: scan site-packages directories in the mounted target filesystem
        _patterns = [
            "/usr/lib/python*/dist-packages",
            "/usr/lib/python*/site-packages",
            "/usr/local/lib/python*/dist-packages",
            "/usr/local/lib/python*/site-packages",
            # conda / virtualenv layouts common in ML containers (e.g. /opt/conda)
            "/opt/*/lib/python*/site-packages",
            "/opt/*/lib/python*/dist-packages",
            # virtualenv one level deeper (e.g. /opt/dynamo/venv/lib/python*/site-packages)
            "/opt/*/*/lib/python*/site-packages",
            "/opt/*/*/lib/python*/dist-packages",
        ]
        search_paths = []
        print(f"[python] search paths ({len(_patterns)} patterns):", file=sys.stderr)
        for pattern in _patterns:
            matches = glob.glob(f"{root}{pattern}")
            marker = "+" if matches else "-"
            label = f"({len(matches)} match)" if matches else "(no match)"
            print(f"[python]   {marker} {root}{pattern}  {label}", file=sys.stderr)
            search_paths.extend(matches)
        dists = importlib.metadata.distributions(path=search_paths)
    else:
        # Local mode: enumerate distributions in the running Python environment
        dists = importlib.metadata.distributions()

    seen = set()
    for dist in dists:
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

    count = len(seen)
    icon = "✅" if count > 0 else "⚠️"
    print(f"{icon} [python] extracted {count} package(s)", file=sys.stderr)


if __name__ == "__main__":
    main()
