# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify the published metadata and archive boundaries of Dynamo Python wheels."""

from __future__ import annotations

import argparse
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile


def find_one(wheelhouse: Path, pattern: str) -> Path:
    matches = sorted(wheelhouse.glob(pattern))
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one wheel matching {pattern!r}, found {matches}"
        )
    return matches[0]


def wheel_metadata(wheel: Path):
    with ZipFile(wheel) as archive:
        metadata_files = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise AssertionError(
                f"expected one METADATA file in {wheel.name}, found {metadata_files}"
            )
        return BytesParser().parsebytes(archive.read(metadata_files[0]))


def verify(wheelhouse: Path) -> None:
    dynamo_wheel = find_one(wheelhouse, "ai_dynamo-*-py3-none-any.whl")
    profiler_wheel = find_one(wheelhouse, "ai_dynamo_profiler-*-py3-none-any.whl")

    dynamo_metadata = wheel_metadata(dynamo_wheel)
    profiler_metadata = wheel_metadata(profiler_wheel)
    dynamo_version = dynamo_metadata["Version"]
    profiler_version = profiler_metadata["Version"]
    if profiler_version != dynamo_version:
        raise AssertionError(
            "ai-dynamo-profiler and ai-dynamo versions differ: "
            f"{profiler_version} != {dynamo_version}"
        )

    direct_references = [
        requirement
        for requirement in dynamo_metadata.get_all("Requires-Dist", [])
        if " @ " in requirement or "git+" in requirement
    ]
    if direct_references:
        raise AssertionError(
            f"{dynamo_wheel.name} contains direct dependency references: "
            f"{direct_references}"
        )

    profiler_requirements = {
        requirement.replace(" ", "")
        for requirement in profiler_metadata.get_all("Requires-Dist", [])
    }
    expected_requirements = {
        f"ai-dynamo=={dynamo_version}",
        "aiconfigurator==0.10.0",
        "aiconfigurator-core==0.10.0",
    }
    missing_requirements = expected_requirements - profiler_requirements
    if missing_requirements:
        raise AssertionError(
            f"{profiler_wheel.name} is missing exact dependencies: "
            f"{sorted(missing_requirements)}"
        )

    expected_license = Path(__file__).with_name("LICENSE").read_bytes()
    with ZipFile(profiler_wheel) as archive:
        names = archive.namelist()
        if "dynamo/profiler/__init__.py" not in names:
            raise AssertionError(
                f"{profiler_wheel.name} does not contain dynamo/profiler"
            )
        license_files = [
            name for name in names if name.endswith(".dist-info/licenses/LICENSE")
        ]
        if len(license_files) != 1:
            raise AssertionError(
                f"expected one packaged LICENSE in {profiler_wheel.name}, "
                f"found {license_files}"
            )
        if archive.read(license_files[0]) != expected_license:
            raise AssertionError(
                f"{profiler_wheel.name} does not contain the repository LICENSE"
            )

    with ZipFile(dynamo_wheel) as archive:
        profiler_payload = [name for name in archive.namelist() if "/profiler/" in name]
    if profiler_payload:
        raise AssertionError(
            f"{dynamo_wheel.name} still contains profiler payload: {profiler_payload[:5]}"
        )

    print(
        f"verified {dynamo_wheel.name} and {profiler_wheel.name}: "
        "version-only metadata, disjoint payloads, and packaged license"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheelhouse", type=Path)
    args = parser.parse_args()
    verify(args.wheelhouse)


if __name__ == "__main__":
    main()
