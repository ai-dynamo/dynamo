# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fence the complete SGLang CUDA-writer cohort before ownership changes.

The active flock belongs to the Dynamo process, not its CUDA children. Each
boot therefore has a separate shared flock held through process exit by both
the parent and every scheduler. A successor takes the predecessor's exclusive
flock before publishing its own boot identity or touching surviving KV.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import uuid
from functools import partial
from pathlib import Path

from gpu_memory_service.integrations.common.process_lifecycle import (
    arm_parent_death_signal,
)

_boot: tuple[int, Path] | None = None
# Keep these open through scheduler return and CUDA/process teardown. The kernel
# releases them on exit. Never unlink a cohort file: delayed children must open
# the same inode as the successor fencing it, not a replacement.
_writer_fds: list[int] = []


def _hold_writer_guard(path: Path) -> None:
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
    except BaseException:
        os.close(fd)
        raise
    _writer_fds.append(fd)


def prepare_writer_cohort() -> Path:
    """Create one boot identity before acquiring ownership or spawning children."""
    global _boot
    if _boot is not None and _boot[0] == os.getpid():
        return _boot[1]
    directory = Path(os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock"))
    directory = directory.with_name(directory.name + ".writers")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / uuid.uuid4().hex
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
    except BaseException:
        os.close(fd)
        raise
    _writer_fds.append(fd)
    _boot = (os.getpid(), path)
    return path


async def fence_predecessor_writers() -> None:
    """Called with active flock held; cancellation never advances ownership.

    There is deliberately no timeout: a delayed or unkillable writer is not proof
    of safe KV reuse. The old cohort must also exit on cooperative handoff; merely
    releasing its active lock while its writers remain alive is insufficient.
    """
    current = prepare_writer_cohort()
    marker = current.parent / "active"
    try:
        previous = marker.read_text().strip()
    except FileNotFoundError:
        previous = None
    if previous is not None and previous != current.name:
        if uuid.UUID(hex=previous).hex != previous:
            raise RuntimeError("Invalid GMS writer-cohort identity")
        fd = os.open(
            current.parent / previous, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        )
        try:
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(0.01)
        finally:
            os.close(fd)
    # A crash before atomic replace leaves the old identity; a crash after it
    # causes the next owner to fence this boot. The main flock serializes this.
    pending = current.parent / (current.name + ".active")
    pending.write_text(current.name)
    os.replace(pending, marker)


def run_guarded_scheduler(
    expected_parent_pid: int, guard_path: str, scheduler, *args, **kwargs
):
    """Picklable multiprocessing entry; arm before any SGLang/CUDA work."""
    arm_parent_death_signal(expected_parent_pid=expected_parent_pid)
    _hold_writer_guard(Path(guard_path))
    # The parent may have exited while waiting for the shared flock.
    arm_parent_death_signal(expected_parent_pid=expected_parent_pid)
    return scheduler(*args, **kwargs)


def create_guarded_engine(engine_type, *, server_args):
    """Use SGLang's scheduler-entry seam, without replacing engine methods."""
    guard = prepare_writer_cohort()

    class GuardedEngine(engine_type):
        run_scheduler_process_func = staticmethod(
            partial(
                run_guarded_scheduler,
                os.getpid(),
                str(guard),
                engine_type.run_scheduler_process_func,
            )
        )

    return GuardedEngine(server_args=server_args)
