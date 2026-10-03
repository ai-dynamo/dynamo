# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Align installed SGLang requirement metadata with the image's wheel pins.

The runtime image installs av==18.1.0 and opencv-python-headless==4.14.0.94.
SGLang 0.5.17 still declares av==16.1.0 for aarch64 and
opencv-python-headless==4.10.0.84 for the diffusion extra. pip and
importlib.metadata read those declarations from sglang-*.dist-info/METADATA.
An editable install also keeps a copy under sglang.egg-info.
"""

from __future__ import annotations

import importlib.metadata as md
import re
import sys
from pathlib import Path

REQUIREMENT_NAMES = {"METADATA", "PKG-INFO", "requires.txt", "pyproject.toml", "setup.cfg"}
REPLACEMENTS = (
    (re.compile(r"av\s*==\s*16\.1\.0"), "av==18.1.0"),
    (
        re.compile(r"opencv-python-headless\s*==\s*4\.10\.0\.84"),
        "opencv-python-headless==4.14.0.94",
    ),
)


def _add(paths: list[Path], seen: set[Path], path: Path) -> None:
    if not path.is_file() or path.name not in REQUIREMENT_NAMES:
        return
    resolved = path.resolve()
    if resolved in seen:
        return
    seen.add(resolved)
    paths.append(resolved)


def requirement_files(dist: md.Distribution, package_file: Path | None) -> list[Path]:
    """Return installed and source requirement files for this distribution.

    ``Distribution.files`` plus ``locate()`` is the metadata pip reads, including
    ``sglang-*.dist-info/METADATA`` for a regular wheel install. ``locate_file("")``
    is the site-packages parent and is not a requirement file.
    """
    paths: list[Path] = []
    seen: set[Path] = set()
    if dist.files:
        for entry in dist.files:
            if Path(str(entry)).name not in REQUIREMENT_NAMES:
                continue
            locate = getattr(entry, "locate", None)
            if locate is None:
                continue
            _add(paths, seen, Path(locate()))
    info = getattr(dist, "_path", None)
    if info is not None:
        root = Path(info)
        for name in REQUIREMENT_NAMES:
            _add(paths, seen, root / name)
    if package_file is not None:
        for root in package_file.resolve().parents[1:3]:
            egg = root / "sglang.egg-info"
            for name in ("PKG-INFO", "requires.txt", "METADATA"):
                _add(paths, seen, egg / name)
            _add(paths, seen, root / "pyproject.toml")
            _add(paths, seen, root / "setup.cfg")
            _add(paths, seen, root / "python" / "pyproject.toml")
    return paths


def active_metadata(dist: md.Distribution) -> Path:
    """Return the METADATA file importlib uses for this distribution."""
    if dist.files:
        for entry in dist.files:
            if Path(str(entry)).name != "METADATA":
                continue
            locate = getattr(entry, "locate", None)
            if locate is None:
                continue
            path = Path(locate())
            if path.is_file():
                return path.resolve()
    info = getattr(dist, "_path", None)
    if info is not None:
        path = Path(info) / "METADATA"
        if path.is_file():
            return path.resolve()
    raise SystemExit("could not locate installed sglang METADATA")


def rewrite(text: str) -> str:
    revised = text
    for pattern, replacement in REPLACEMENTS:
        revised = pattern.sub(replacement, revised)
    return revised


def align(dist: md.Distribution, package_file: Path | None) -> tuple[list[Path], str]:
    metadata_path = active_metadata(dist)
    before = metadata_path.read_text(encoding="utf-8")
    updated: list[Path] = []
    for path in requirement_files(dist, package_file):
        text = path.read_text(encoding="utf-8")
        revised = rewrite(text)
        if revised == text:
            continue
        path.write_text(revised, encoding="utf-8")
        updated.append(path)
    after = metadata_path.read_text(encoding="utf-8")
    if re.search(r"av\s*==\s*16\.1\.0", after):
        raise SystemExit(f"sglang METADATA still declares av==16.1.0: {metadata_path}")
    if re.search(r"opencv-python-headless\s*==\s*4\.10\.0\.84", after):
        raise SystemExit(
            f"sglang METADATA still declares opencv-python-headless==4.10.0.84: {metadata_path}"
        )
    if re.search(r"av\s*==\s*16\.1\.0", before) and "av==18.1.0" not in after:
        raise SystemExit(f"sglang METADATA was not updated: {metadata_path}")
    if "4.10.0.84" in before and "opencv-python-headless==4.14.0.94" not in after:
        raise SystemExit(f"sglang OpenCV requirement was not updated: {metadata_path}")
    return updated, before


def main() -> None:
    dist = md.distribution("sglang")
    package_file = None
    try:
        import sglang
    except ImportError:
        sglang = None
    if sglang is not None and getattr(sglang, "__file__", None):
        package_file = Path(sglang.__file__)
    updated, before = align(dist, package_file)
    reported = "\n".join(md.requires("sglang") or [])
    if re.search(r"av\s*==\s*16\.1\.0", reported):
        raise SystemExit("importlib.metadata.requires('sglang') still reports av==16.1.0")
    if re.search(r"opencv-python-headless\s*==\s*4\.10\.0\.84", reported):
        raise SystemExit(
            "importlib.metadata.requires('sglang') still reports opencv-python-headless==4.10.0.84"
        )
    if re.search(r"av\s*==\s*16\.1\.0", before) and "av==18.1.0" not in reported:
        raise SystemExit(
            "importlib.metadata.requires('sglang') does not report av==18.1.0\n" + reported
        )
    if "4.10.0.84" in before and "opencv-python-headless==4.14.0.94" not in reported:
        raise SystemExit(
            "importlib.metadata.requires('sglang') does not report opencv-python-headless==4.14.0.94\n"
            + reported
        )
    if updated:
        print("\n".join(str(path) for path in updated))
    else:
        print(
            "installed SGLang metadata does not declare "
            "av==16.1.0 or opencv-python-headless==4.10.0.84"
        )


if __name__ == "__main__":
    sys.exit(main())
