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
import time
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


def retired_writer_cohort_has_no_processes(path: Path) -> bool:
    """Check the immutable tombstone while excluding all cohort guard holders.

    This proves that the registered CPU/CUDA worker processes have released
    their lifetime guards. It is deliberately not a substitute for MPS client
    termination: an MPS server can retain outstanding GPU work after a client
    process exits.
    """
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return os.pread(fd, 1, 0) == b"R"
    finally:
        os.close(fd)


_PF_EXITING = 0x00000004
_SIGKILL_BIT = 1 << (signal.SIGKILL - 1)
_NETWORK_FILESYSTEMS = ("nfs", "nfs4", "cifs", "smb3", "fuse", "ceph", "9p", "lustre")


def _pod_scoped_fence() -> bool:
    return os.environ.get("DYN_GMS_WRITER_COHORT_SCOPE", "").strip().lower() == "pod"


def _on_local_filesystem(path: Path) -> bool:
    """Whether every guard holder must run in this kernel (not a network FS)."""
    target = os.path.realpath(path)
    best, fstype = "", None
    try:
        with open("/proc/self/mountinfo") as mounts:
            for line in mounts:
                left, _, right = line.partition(" - ")
                mount_point = left.split()[4]
                inside = target == mount_point or target.startswith(
                    mount_point.rstrip("/") + "/"
                )
                if inside and len(mount_point) >= len(best):
                    best, fstype = mount_point, right.split()[0]
    except OSError:
        return False
    return fstype is not None and not fstype.startswith(_NETWORK_FILESYSTEMS)


def _guard_lock_holders(path: Path) -> set[int]:
    """PIDs with a descriptor that holds a lock on ``path``'s inode.

    ``fdinfo`` lists the locks of each open file description, so a child that
    inherited a locked descriptor across fork is reported too. Descriptors
    without a lock (for example this fence's own) are not holders.
    """
    target = os.stat(path)
    holders = set()
    for entry in os.listdir("/proc"):
        if entry.isdigit() and _holds_lock_on(f"/proc/{entry}", target):
            holders.add(int(entry))
    return holders


def _holds_lock_on(proc: str, target: os.stat_result) -> bool:
    for base in _descriptor_tables(proc):
        try:
            fds = os.listdir(f"{base}/fd")
        except OSError:
            continue
        for fd in fds:
            try:
                st = os.stat(f"{base}/fd/{fd}")
                if (st.st_dev, st.st_ino) != (target.st_dev, target.st_ino):
                    continue
                with open(f"{base}/fdinfo/{fd}") as info:
                    if any(line.startswith("lock:") for line in info):
                        return True
            except OSError:
                continue
    return False


def _descriptor_tables(proc: str):
    """Yield ``proc``, then its threads if its main thread has no fd table.

    ``/proc/<pid>/fd`` shows the main thread's table. A main thread that
    exited first (a zombie leader) shows none, while threads still blocked
    in driver teardown keep the shared table, and its flocks, alive.
    """
    yield proc
    try:
        if os.listdir(f"{proc}/fd"):
            return
        tasks = os.listdir(f"{proc}/task")
    except OSError:
        return
    pid = proc.rsplit("/", 1)[1]
    for tid in tasks:
        task = f"{proc}/task/{tid}"
        try:
            if tid != pid and os.listdir(f"{task}/fd"):
                # Threads share one table; the first live one shows it.
                yield task
                return
        except OSError:
            continue


def _sigkill_pending(task: str) -> bool:
    try:
        with open(f"{task}/status") as status:
            for line in status:
                if line.startswith(("SigPnd:", "ShdPnd:")):
                    if int(line.split()[1], 16) & _SIGKILL_BIT:
                        return True
    except OSError:
        return True
    return False


def _user_space_dead(pid: int) -> bool:
    """True once no thread of ``pid`` can return to user space.

    The kernel sets PF_EXITING on a thread as it enters do_exit; it never runs
    user code again. A killed thread still blocked in a driver call enters
    do_exit only when that call returns, but its pending SIGKILL is taken
    before any return to user space and cannot be blocked or handled. Driver
    teardown of GPU mappings can take seconds, and the process's descriptors
    (and flocks) are only closed at the very end.
    """
    try:
        tasks = os.listdir(f"/proc/{pid}/task")
    except OSError:
        return True
    for tid in tasks:
        task = f"/proc/{pid}/task/{tid}"
        try:
            with open(f"{task}/stat") as stat:
                fields = stat.read().rsplit(")", 1)[1].split()
        except OSError:
            continue
        if not int(fields[6]) & _PF_EXITING and not _sigkill_pending(task):
            return False
    return True


def _lock_owner_pids(target: os.stat_result) -> list[int]:
    """Owners of granted locks on ``target`` listed in /proc/locks.

    An exiting process closes its descriptors before the final close of each
    file, which runs with the (slow) GPU device release at the end of exit.
    In between, the lock is held but in no descriptor table; its owner still
    appears here. Locks of processes outside this PID namespace are hidden.
    """
    owners = []
    try:
        with open("/proc/locks") as locks:
            for line in locks:
                fields = line.split()
                if len(fields) < 6 or fields[1] == "->":
                    continue  # a blocked waiter holds nothing
                major, minor, ino = fields[5].split(":")
                device = (int(major, 16), int(minor, 16))
                if int(ino) == target.st_ino and device == (
                    os.major(target.st_dev),
                    os.minor(target.st_dev),
                ):
                    owners.append(int(fields[4]))
    except (OSError, ValueError):
        return []
    return owners


def _holders_all_exiting(path: Path) -> bool:
    owners = _lock_owner_pids(os.stat(path))
    if not owners or any(pid <= 0 for pid in owners):
        # A busy lock with no visible owner is held outside this namespace.
        return False
    holders = _guard_lock_holders(path) | set(owners)
    holders.discard(os.getpid())
    return bool(holders) and all(_user_space_dead(pid) for pid in holders)


async def retire_writer_cohort(path: Path) -> None:
    """Exclude current and future CPU submitters; NOT a CUDA completion fence.

    Never unlink/recreate the inode: a delayed opener must see its tombstone.
    Cancellation while waiting leaves admission and ownership unchanged.

    With a pod-scoped cohort on a local filesystem every guard holder runs in
    this pod, whose containers must share one PID namespace
    (``shareProcessNamespace``). The fence can then also complete once every
    holder has stopped running user code, instead of waiting seconds for
    driver teardown to close its descriptors. The tombstone is written first
    and the holders re-checked, so a member that joined in between keeps the
    fence waiting.
    """
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        fast = _pod_scoped_fence() and _on_local_filesystem(path)
        next_scan = 0.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                now = time.monotonic()
                if fast and now >= next_scan:
                    next_scan = now + 0.02
                    if _holders_all_exiting(path):
                        if os.pwrite(fd, b"R", 0) != 1:
                            raise OSError("could not retire GMS writer cohort")
                        if _holders_all_exiting(path):
                            return
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
