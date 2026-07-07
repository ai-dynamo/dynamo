#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the campaign artifact tree to a dedicated volume and verify it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MARKER_NAME = ".glm52-artifact-migration-v1.json"
DESTINATION_NAME = "glm52-nscale"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_json(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_identity(root: Path, *, ignore_marker: bool = False) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"tree root is not a real directory: {root}")
    digest = hashlib.sha256()
    regular_files = 0
    symlinks = 0
    directories = 0
    byte_count = 0

    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if ignore_marker and relative == MARKER_NAME:
            continue
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
            content = b""
            directories += 1
        elif stat.S_ISREG(metadata.st_mode):
            kind = "file"
            content = bytes.fromhex(file_sha256(path))
            regular_files += 1
            byte_count += metadata.st_size
        elif stat.S_ISLNK(metadata.st_mode):
            kind = "symlink"
            content = os.readlink(path).encode()
            symlinks += 1
        else:
            raise ValueError(f"unsupported artifact type: {path}")
        digest.update(kind.encode())
        digest.update(b"\0")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(f"{mode:o}".encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")

    return {
        "tree_sha256": digest.hexdigest(),
        "regular_files": regular_files,
        "symlinks": symlinks,
        "directories": directories,
        "bytes": byte_count,
    }


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe relative artifact path: {value!r}")
    return path


def requirement_path(root: Path, value: str) -> Path:
    relative = safe_relative_path(value)
    root = root.resolve()
    candidate = root
    for part in relative.parts:
        candidate /= part
        if candidate.is_symlink():
            raise ValueError(f"required artifact path traverses a symlink: {value}")
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"required artifact path escapes the tree: {value}")
    return candidate


def validate_requirements(
    root: Path,
    required_absent: tuple[str, ...],
    required_files: dict[str, str],
) -> None:
    for value in required_absent:
        path = requirement_path(root, value)
        if path.exists() or path.is_symlink():
            raise ValueError(f"required clean replay path exists: {value}")
    for value, expected_sha256 in required_files.items():
        path = requirement_path(root, value)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"required artifact file is absent or unsafe: {value}")
        actual_sha256 = file_sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"required artifact hash differs for {value}: "
                f"expected={expected_sha256} actual={actual_sha256}"
            )


def copy_and_verify(
    source: Path,
    destination_parent: Path,
    source_pvc_uid: str,
    *,
    required_absent: tuple[str, ...] = (),
    required_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    required_files = required_files or {}
    source = source.resolve()
    destination_parent = destination_parent.resolve()
    destination = destination_parent / DESTINATION_NAME
    marker_path = destination / MARKER_NAME
    validate_requirements(source, required_absent, required_files)
    source_before = tree_identity(source)

    if destination.exists():
        if not marker_path.is_file():
            raise ValueError(f"unmarked destination already exists: {destination}")
        marker = json.loads(marker_path.read_text())
        if not isinstance(marker, dict) or marker.get("schema_version") != 1:
            raise ValueError("destination marker is invalid")
        if marker.get("source_pvc_uid") != source_pvc_uid:
            raise ValueError("destination marker source PVC differs")
        expected_requirements = {
            "required_absent": sorted(required_absent),
            "required_files": dict(sorted(required_files.items())),
        }
        if marker.get("requirements") != expected_requirements:
            raise ValueError("destination marker requirements differ")
        validate_requirements(destination, required_absent, required_files)
        destination_identity = tree_identity(destination, ignore_marker=True)
        if marker.get("source_identity") != source_before:
            raise ValueError("source tree differs from completed migration marker")
        if marker.get("destination_identity") != destination_identity:
            raise ValueError("destination tree differs from completed migration marker")
        return {
            "state": "already_complete",
            "marker": marker,
            "marker_sha256": hashlib.sha256(marker_path.read_bytes()).hexdigest(),
        }

    destination_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{DESTINATION_NAME}.", suffix=".staging", dir=destination_parent)
    )
    try:
        shutil.rmtree(staging)
        shutil.copytree(source, staging, symlinks=True, copy_function=shutil.copy2)
        source_after = tree_identity(source)
        if source_after != source_before:
            raise RuntimeError("source tree changed during migration")
        destination_identity = tree_identity(staging)
        if destination_identity != source_before:
            raise RuntimeError(
                "destination verification failed: "
                f"source={json.dumps(source_before, sort_keys=True)} "
                f"destination={json.dumps(destination_identity, sort_keys=True)}"
            )
        validate_requirements(staging, required_absent, required_files)
        marker = {
            "schema_version": 1,
            "completed_at": utc_now(),
            "source": "/source/glm52-nscale",
            "destination": "/destination/glm52-nscale",
            "source_pvc_uid": source_pvc_uid,
            "source_identity": source_before,
            "destination_identity": destination_identity,
            "requirements": {
                "required_absent": sorted(required_absent),
                "required_files": dict(sorted(required_files.items())),
            },
        }
        marker_bytes = canonical_json(marker)
        marker_file = staging / MARKER_NAME
        marker_file.write_bytes(marker_bytes)
        marker_file.chmod(0o444)
        os.sync()
        staging.rename(destination)
        return {
            "state": "complete",
            "marker": marker,
            "marker_sha256": hashlib.sha256(marker_bytes).hexdigest(),
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination-parent", required=True, type=Path)
    parser.add_argument("--source-pvc-uid", required=True)
    parser.add_argument("--require-absent", action="append", default=[])
    parser.add_argument(
        "--require-file-sha256",
        action="append",
        default=[],
        metavar="RELATIVE_PATH=SHA256",
    )
    args = parser.parse_args()
    if not args.source_pvc_uid.strip():
        parser.error("source PVC UID must not be empty")
    required_files: dict[str, str] = {}
    for value in args.require_file_sha256:
        path, separator, digest = value.rpartition("=")
        if not separator or not path or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            parser.error(f"invalid required file hash: {value!r}")
        if path in required_files:
            parser.error(f"duplicate required file path: {path}")
        required_files[path] = digest
    print(
        json.dumps(
            copy_and_verify(
                args.source,
                args.destination_parent,
                args.source_pvc_uid,
                required_absent=tuple(args.require_absent),
                required_files=required_files,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
