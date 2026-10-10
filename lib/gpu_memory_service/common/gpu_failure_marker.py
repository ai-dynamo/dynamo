# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-latency, non-authoritative notification of a GPU writer crash."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

GPU_FAILURE_MARKER_SUFFIX = ".gpu-failed-v1"


def gpu_failure_marker_path(cohort: str | Path) -> Path:
    """Return the marker uniquely associated with one writer generation."""

    path = Path(cohort)
    if not path.is_absolute():
        raise ValueError("GPU writer cohort path must be absolute")
    return path.with_name(path.name + GPU_FAILURE_MARKER_SUFFIX)


def publish_gpu_failure_marker(
    cohort: str | Path, *, rank: int, pid: int, source: str
) -> Path:
    """Publish the first crash report for ``cohort`` without following links.

    This is a latency hint only. It cannot prove host-writer retirement, CUDA
    quiescence, or lease ownership and must never be used as such.
    """

    path = gpu_failure_marker_path(cohort)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=path.parent)
    try:
        payload = f"{int(rank)} {int(pid)} {source}\n".encode()
        while payload:
            payload = payload[os.write(fd, payload) :]
        os.close(fd)
        fd = -1
        try:
            # link(2) makes the complete file visible atomically and preserves
            # the first report when multiple ranks detect the same cohort loss.
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            pass
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return path


def read_gpu_failure_marker(path: str | Path) -> tuple[int, int, str] | None:
    """Read a complete marker, returning ``None`` while publication is partial."""

    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        data = os.read(fd, 256)
    finally:
        os.close(fd)
    try:
        fields = data.decode().strip().split(maxsplit=2)
    except UnicodeDecodeError:
        return None
    if len(fields) != 3:
        return None
    try:
        rank, pid = int(fields[0]), int(fields[1])
    except ValueError:
        return None
    if rank < 0 or pid <= 0 or not fields[2]:
        return None
    return rank, pid, fields[2]
