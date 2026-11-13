#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remove only reviewed packaging changelogs from the actual runtime filesystem.

Version, copyright bytes, and changelog content/target must all match before
any deletion. Copyright and upstream license/notice files are never removed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import overrides
from .generators.dpkg import baseline_keys

_DEFAULT_EVIDENCE = Path(__file__).parent / "evidence/dpkg-license-corrections.json"


def trim_reviewed_docs(
    root: Path,
    installed: dict[str, str],
    catalog: Path,
    evidence_path: Path,
    *,
    baseline: set[tuple[str, str]] | None = None,
) -> list[str]:
    root = root.resolve()
    records = {
        (r.name, r.version): r
        for r in overrides.load_records(catalog)
        if r.ecosystem == "dpkg" and r.absent_paths
    }
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence["schema_version"] != 1:
        raise ValueError("Unsupported packaging-document evidence schema")
    planned = []
    for review in evidence["packages"]:
        name, version = review["name"], review["version"]
        if installed.get(name) != version:
            continue  # Unsupported/new versions retain all files and get no correction.
        if baseline and (name, version) in baseline:
            print(
                f"Retained baseline-owned package without correction: {name}@{version}"
            )
            continue
        record = records.get((name, version))
        if record is None:
            raise ValueError(
                f"Missing scoped override for reviewed cleanup {name}@{version}"
            )
        if (
            record.license != review["license"]
            or record.copyright_sha256 != review["copyright_sha256"]
            or set(record.absent_paths) != set(review["absent_paths"])
        ):
            raise ValueError(f"Override/evidence scope mismatch for {name}@{version}")
        copyright_path = overrides.checked_artifact_path(
            root, f"/usr/share/doc/{name}/copyright"
        )
        if copyright_path.is_symlink() or not copyright_path.is_file():
            raise ValueError(f"Expected a regular copyright file for {name}@{version}")
        if (
            hashlib.sha256(copyright_path.read_bytes()).hexdigest()
            != record.copyright_sha256
        ):
            raise ValueError(
                f"Copyright hash mismatch before cleanup for {name}@{version}"
            )
        for relative in record.absent_paths:
            path = overrides.checked_artifact_path(
                root, relative, allow_leaf_symlink=True
            )
            if not (path.exists() or path.is_symlink()):
                continue  # Idempotent after a previous runtime/EFA stage.
            reviews = [
                binary["removed_document"]
                for binary in review["binary_packages"]
                if binary["removed_document"]["path"] == relative
            ]
            if path.is_symlink():
                allowed = {
                    item["symlink_target"]
                    for item in reviews
                    if "symlink_target" in item
                }
                if os.readlink(path) not in allowed:
                    raise ValueError(f"Unreviewed changelog symlink: {relative}")
            elif path.is_file():
                allowed = {item["sha256"] for item in reviews if "sha256" in item}
                if hashlib.sha256(path.read_bytes()).hexdigest() not in allowed:
                    raise ValueError(f"Unreviewed changelog content: {relative}")
            else:
                raise ValueError(f"Refusing non-file changelog: {relative}")
            planned.append(path)
    # Validate the complete plan before mutating any package's documentation.
    for path in planned:
        path.unlink()  # Unlink a reviewed symlink itself, never its target.
        print(f"Removed reviewed packaging changelog: /{path.relative_to(root)}")
    return ["/" + str(path.relative_to(root)) for path in planned]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/"))
    parser.add_argument(
        "--overrides",
        type=Path,
        default=Path(__file__).parent / "license_overrides.yaml",
    )
    parser.add_argument("--evidence", type=Path, default=_DEFAULT_EVIDENCE)
    parser.add_argument("--baseline-sbom", type=Path)
    args = parser.parse_args()
    command = ["dpkg-query", "-W", "-f=${Package}\t${Version}\t${db:Status-Status}\n"]
    if args.root != Path("/"):
        command.insert(1, f"--admindir={args.root / 'var/lib/dpkg'}")
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    installed = {}
    for line in result.stdout.splitlines():
        name, version, state = line.split("\t")
        if state == "installed":
            if name in installed and installed[name] != version:
                raise ValueError(f"Ambiguous multiarch package version: {name}")
            installed[name] = version
    trim_reviewed_docs(
        args.root,
        installed,
        args.overrides,
        args.evidence,
        baseline=baseline_keys(args.baseline_sbom)
        if args.baseline_sbom is not None
        else None,
    )


if __name__ == "__main__":
    main()
