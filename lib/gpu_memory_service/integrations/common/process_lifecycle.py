# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Linux process-lifetime fencing for shared GPU-memory writers."""

from __future__ import annotations

import asyncio
import ctypes
import fcntl
import os
import signal
import sys
from pathlib import Path

_PR_SET_PDEATHSIG = 1


class WriterCohortRetired(RuntimeError):
    """A late child attempted to enter a fenced writer generation."""


def acquire_writer_guard(path: Path) -> int:
    """Join an open cohort, returning a descriptor held until process exit.

    Check retirement *under* the shared lock. The successor writes the tombstone
    under an exclusive lock, so a child either joins before retirement and is
    included in the fence, or fails before it can initialize CUDA/shared KV.
    """
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        if os.pread(fd, 1, 0):
            raise WriterCohortRetired("GMS writer cohort is retired")
        return fd
    except BaseException:
        os.close(fd)
        raise


async def retire_writer_cohort(path: Path) -> None:
    """Exclude current and future CPU submitters; NOT a CUDA completion fence.

    Never unlink/recreate the inode: a delayed opener must see its tombstone.
    Cancellation while waiting leaves admission and ownership unchanged.
    """
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                await asyncio.sleep(0.01)
        if os.pwrite(fd, b"R", 0) != 1:
            raise OSError("could not retire GMS writer cohort")
    finally:
        os.close(fd)


def arm_parent_death_signal(
    signum: int = signal.SIGKILL, *, expected_parent_pid: int | None = None
) -> None:
    """Terminate this process if the process that created it exits.

    A GMS failover flock lives in the Dynamo leader, while EngineCore and CUDA
    writers are descendants. Without this fence, killing only the leader can
    release ownership while an orphaned child still writes shared HBM.
    """
    if sys.platform != "linux":
        raise RuntimeError("GMS shared-KV failover requires Linux PDEATHSIG support")

    # A spawned child can first execute after its parent has already died.
    # getppid() alone would incorrectly arm against the reaper in that case.
    parent_pid = os.getppid() if expected_parent_pid is None else expected_parent_pid
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signum)
        raise RuntimeError("GMS writer's expected parent has already exited")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, int(signum), 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))

    # Close the race where the parent dies between getppid() and prctl().
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signum)
        raise RuntimeError("GMS writer's parent exited while arming PDEATHSIG")
