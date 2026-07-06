#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_manifest(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise RuntimeError(f"cache tree is not a directory: {root}")

    records: list[list[Any]] = []
    file_count = 0
    total_bytes = 0

    def add(path: Path, relative: str) -> None:
        nonlocal file_count, total_bytes
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if stat.S_ISDIR(metadata.st_mode):
            records.append([relative, "directory", mode])
        elif stat.S_ISREG(metadata.st_mode):
            records.append(
                [relative, "file", mode, metadata.st_size, file_sha256(path)]
            )
            file_count += 1
            total_bytes += metadata.st_size
        elif stat.S_ISLNK(metadata.st_mode):
            records.append([relative, "symlink", mode, os.readlink(path)])
        else:
            raise RuntimeError(f"unsupported cache entry: {path}")

    add(root, ".")
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        parent = Path(directory)
        for name in list(directory_names):
            path = parent / name
            add(path, path.relative_to(root).as_posix())
            if path.is_symlink():
                directory_names.remove(name)
        for name in file_names:
            path = parent / name
            add(path, path.relative_to(root).as_posix())

    digest = hashlib.sha256()
    for record in records:
        digest.update(
            json.dumps(record, separators=(",", ":"), ensure_ascii=False).encode()
        )
        digest.update(b"\n")
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": file_count,
        "bytes": total_bytes,
    }


def validate_existing(destination: Path, final: Path, marker_path: Path) -> dict[str, Any]:
    marker = json.loads(marker_path.read_text())
    if marker.get("schema_version") != 1:
        raise RuntimeError("unsupported migration marker schema")
    actual = tree_manifest(final)
    for key in ("tree_sha256", "file_count", "bytes"):
        if marker.get(key) != actual[key]:
            raise RuntimeError(f"existing migration marker mismatch: {key}")
    if marker.get("destination") != str(final):
        raise RuntimeError("existing migration marker destination mismatch")
    return marker


def emit_marker(marker: dict[str, Any], marker_path: Path) -> None:
    payload = marker_path.read_bytes()
    print(
        json.dumps(
            {
                "marker": marker,
                "marker_sha256": hashlib.sha256(payload).hexdigest(),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source", type=Path)
    source_group.add_argument("--initialize-empty", action="store_true")
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()

    temporary_source: tempfile.TemporaryDirectory[str] | None = None
    if args.initialize_empty:
        temporary_source = tempfile.TemporaryDirectory()
        source = Path(temporary_source.name) / "empty-cache"
        (source / "docker/registry/v2/repositories").mkdir(parents=True)
        source_label = "empty-initialization"
    else:
        source = args.source.resolve()
        source_label = str(source)
    destination = args.destination.resolve()
    final = destination / "dockerhub-registry"
    staging = destination / ".dockerhub-registry.migrating-v1"
    marker_path = destination / ".glm52-migration-v1.json"

    if final.exists() or marker_path.exists():
        if not final.is_dir() or not marker_path.is_file():
            raise RuntimeError("destination has an incomplete migration")
        emit_marker(validate_existing(destination, final, marker_path), marker_path)
        return

    unexpected = {
        path.name
        for path in destination.iterdir()
        if path.name not in {"lost+found", staging.name}
    }
    if unexpected:
        raise RuntimeError(f"destination is not empty: {sorted(unexpected)}")
    if staging.exists():
        shutil.rmtree(staging)

    source_manifest = tree_manifest(source)
    shutil.copytree(source, staging, symlinks=True, copy_function=shutil.copy2)
    destination_manifest = tree_manifest(staging)
    if destination_manifest != source_manifest:
        raise RuntimeError(
            f"copied cache differs: source={source_manifest} "
            f"destination={destination_manifest}"
        )

    staging.rename(final)
    marker = {
        "schema_version": 1,
        "captured_at": dt.datetime.now(dt.timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z"),
        "source": source_label,
        "destination": str(final),
        **source_manifest,
    }
    temporary = marker_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker_path)
    os.sync()
    emit_marker(marker, marker_path)
    if temporary_source is not None:
        temporary_source.cleanup()


if __name__ == "__main__":
    main()
