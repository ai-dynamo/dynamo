# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Confine a process to one GPU's device node with Landlock.

libcuda initializes every GPU whose /dev/nvidiaN it can open, and
CUDA_VISIBLE_DEVICES only filters afterwards. A process that can open just
its own GPU's node brings up that GPU alone, which is much faster and does not
slow other CUDA work starting on the node. This is for start-up time, not
isolation. Landlock is an unprivileged Linux security module (5.13+): a
process restricts which files it and its children may open, irrevocably.
"""

from __future__ import annotations

import ctypes
import os
import re
from pathlib import Path

# x86_64 syscall numbers and Landlock uapi constants.
_LANDLOCK_CREATE_RULESET, _LANDLOCK_ADD_RULE, _LANDLOCK_RESTRICT_SELF = 444, 445, 446
_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_ACCESS_FS_WRITE_FILE, _ACCESS_FS_READ_FILE = 1 << 1, 1 << 2
_ACCESS_FS_IOCTL_DEV = 1 << 15  # Landlock ABI 5
_PR_SET_NO_NEW_PRIVS = 38
_GPU_NODE = re.compile(r"nvidia(\d+)$")


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _libc():
    return ctypes.CDLL(None, use_errno=True)


def _check(result: int, operation: str) -> int:
    if result < 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"{operation}: {os.strerror(errno)}")
    return result


def abi_version() -> int:
    """The kernel's Landlock ABI version; raises OSError without Landlock."""
    return _check(
        _libc().syscall(
            _LANDLOCK_CREATE_RULESET,
            None,
            ctypes.c_size_t(0),
            ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
        ),
        "Landlock",
    )


def confine_to_gpu(minor: int, dev: Path = Path("/dev")) -> None:
    """Let this process, and all it starts, open no GPU node in ``dev`` but its own.

    Every other path stays open for reading and writing. Confines the calling
    thread, so call it before any other thread starts.
    """
    libc = _libc()
    handled = _ACCESS_FS_WRITE_FILE | _ACCESS_FS_READ_FILE
    if abi_version() >= 5:
        handled |= _ACCESS_FS_IOCTL_DEV
    attr = ctypes.c_uint64(handled)
    ruleset = _check(
        libc.syscall(
            _LANDLOCK_CREATE_RULESET,
            ctypes.byref(attr),
            ctypes.c_size_t(ctypes.sizeof(attr)),
            ctypes.c_uint32(0),
        ),
        "landlock_create_ruleset",
    )

    def allow(path: Path) -> None:
        if path.is_symlink():  # covered through its target, e.g. /dev/stderr
            return
        fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        try:
            rule = _PathBeneathAttr(handled, fd)
            _check(
                libc.syscall(
                    _LANDLOCK_ADD_RULE,
                    ruleset,
                    _LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(rule),
                    ctypes.c_uint32(0),
                ),
                f"landlock_add_rule {path}",
            )
        finally:
            os.close(fd)

    try:
        # Allow everything beside the path from / down to dev, then dev's
        # entries but the other GPUs' nodes.
        dev = dev.resolve()
        for child in (*reversed(dev.parents[:-1]), dev):
            for entry in child.parent.iterdir():
                if entry != child:
                    allow(entry)
        for entry in dev.iterdir():
            match = _GPU_NODE.match(entry.name)
            other_gpu = match is not None and int(match.group(1)) != minor
            if entry.name != "nvidia-caps" and not other_gpu:
                allow(entry)
        # Landlock requires it of an unprivileged caller.
        _check(libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0), "PR_SET_NO_NEW_PRIVS")
        _check(
            libc.syscall(_LANDLOCK_RESTRICT_SELF, ruleset, ctypes.c_uint32(0)),
            "landlock_restrict_self",
        )
    finally:
        os.close(ruleset)
