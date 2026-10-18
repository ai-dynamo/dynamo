# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ship the source/relink materials required by the reviewed soxr exception."""

from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

from .generators.python import installed_search_paths

SOXR_SHA256 = "9f228ae21c78fa9359ca98d8a5e8e91f30639e438e574133dace62c5b5309e44"


def collect(root: Path, legal: Path) -> None:
    versions = set()
    for site in installed_search_paths(root):
        for metadata in site.glob("soxr-*.dist-info/METADATA"):
            package = email.parser.Parser().parsestr(metadata.read_text())
            if package.get("Name") == "soxr":
                versions.add(package["Version"])
    if not versions:
        return
    if versions != {"1.1.0"}:
        raise ValueError(f"Review source/relink materials for soxr versions {versions}")
    with urllib.request.urlopen(
        "https://pypi.org/pypi/soxr/1.1.0/json", timeout=60
    ) as response:
        release = json.load(response)
    archives = [entry for entry in release["urls"] if entry["packagetype"] == "sdist"]
    if len(archives) != 1 or archives[0]["digests"]["sha256"] != SOXR_SHA256:
        raise ValueError("Unexpected soxr source archive identity")
    with urllib.request.urlopen(archives[0]["url"], timeout=120) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != SOXR_SHA256:
        raise ValueError("soxr source checksum mismatch")
    destination = legal / "sources/soxr-1.1.0"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "soxr-1.1.0.tar.gz").write_bytes(data)
    shutil.copyfile(
        Path(__file__).parent / "sources/soxr-1.1.0/README.md",
        destination / "README.md",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/"))
    parser.add_argument("--legal", type=Path, required=True)
    arguments = parser.parse_args()
    collect(arguments.root, arguments.legal)
