#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute the content identity used for runner-local prefill checkpoints."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path


def file_identity(path: Path) -> tuple[int, bytes]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.digest()


def tree_identity(root: Path) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"checkpoint root is not a real directory: {root}")
    digest = hashlib.sha256()
    count = 0
    directories = 0
    byte_count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"checkpoint contains a symlink: {path}")
        if stat.S_ISDIR(metadata.st_mode):
            digest.update(b"directory\0")
            digest.update(relative.encode())
            digest.update(b"\0")
            directories += 1
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"checkpoint contains an unsupported entry: {path}")
        size, content_digest = file_identity(path)
        digest.update(b"file\0")
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(content_digest)
        digest.update(b"\0")
        count += 1
        byte_count += size
    return {
        "files": count,
        "directories": directories,
        "bytes": byte_count,
        "tree_sha256": digest.hexdigest(),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} ROOT")
    print(json.dumps(tree_identity(Path(sys.argv[1]).resolve()), sort_keys=True))
