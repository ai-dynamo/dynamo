# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fence every vLLM process that can enqueue work against shared KV HBM."""

from __future__ import annotations

import fcntl
import hashlib
import multiprocessing
import os
import time
import uuid
from pathlib import Path

from gpu_memory_service.integrations.common.process_lifecycle import (
    WriterCohortRetired,
    acquire_writer_guard,
    arm_parent_death_signal,
    retire_writer_cohort,
)

_COHORT_ENV = "GMS_VLLM_WRITER_COHORT_PATH"
_boot: tuple[int, Path] | None = None
_writer_fds: list[int] = []
_held_in_pid: tuple[int, Path] | None = None
_leader_fd: int | None = None


def _close_leader_guard() -> None:
    global _leader_fd
    if _leader_fd is not None:
        os.close(_leader_fd)
        _leader_fd = None


# A forked child must not keep the leader-admission lease alive. Spawn/exec
# already closes it via O_CLOEXEC. The separate writer guard remains inherited.
os.register_at_fork(after_in_child=_close_leader_guard)


def _leader_is_live(path: Path) -> bool:
    if _boot == (os.getpid(), path) and _leader_fd is not None:
        return True
    try:
        fd = os.open(
            path.with_suffix(".leader"), os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW
        )
    except FileNotFoundError:
        return False
    try:
        try:
            # Probes must be shared: two concurrent probes must not mistake
            # one another for the leader's exclusive lifetime lock.
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False
    finally:
        os.close(fd)


def writer_cohort_required() -> bool:
    """Return whether either supported vLLM failover switch is enabled."""
    return any(
        os.environ.get(name, "").strip().lower() not in ("", "0", "false", "no", "off")
        for name in (
            "DYN_GMS_FAILOVER_SHADOW_MODE",
            "DYN_VLLM_GMS_SHADOW_MODE",
        )
    )


def _directory() -> Path:
    lock = Path(os.environ.get("FAILOVER_LOCK_PATH", "/shared/failover.lock"))
    return lock.with_name(lock.name + ".vllm-writers")


def _joining_marker() -> Path:
    engine_id = os.environ.get("ENGINE_ID", "0")
    digest = hashlib.sha256(engine_id.encode("utf-8")).hexdigest()[:20]
    return _directory() / ("joining-" + digest)


def _hold(path: Path) -> None:
    global _held_in_pid
    identity = (os.getpid(), path)
    if _held_in_pid == identity:
        return
    fd = acquire_writer_guard(path)
    _writer_fds.append(fd)
    _held_in_pid = identity


def prepare_writer_cohort() -> Path:
    """Create a boot identity before vLLM starts EngineCore or CUDA workers."""
    global _boot, _held_in_pid, _leader_fd
    if _boot is not None and _boot[0] == os.getpid():
        return _boot[1]
    directory = _directory()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / uuid.uuid4().hex
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        leader_fd = os.open(
            path.with_suffix(".leader"),
            os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC,
            0o600,
        )
        try:
            fcntl.flock(leader_fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(leader_fd)
            raise
    except BaseException:
        os.close(fd)
        raise
    _leader_fd = leader_fd
    _writer_fds.append(fd)
    _held_in_pid = (os.getpid(), path)
    _boot = (os.getpid(), path)
    os.environ[_COHORT_ENV] = str(path)
    marker = _joining_marker()
    pending = marker.with_name(marker.name + "." + path.name)
    pending.write_text(path.name)
    os.replace(pending, marker)
    return path


def join_prepared_writer_cohort(timeout: float = 120.0) -> Path:
    """Join a live leader, never a marker left by an earlier launch.

    The leader-only lock is checked while holding this generation's writer
    guard. A leader dying immediately afterward cannot escape the cohort fence.
    This uses the existing shared lock directory, including across TP nodes.
    """
    global _held_in_pid
    stop = time.monotonic() + max(0.0, timeout)
    marker = _joining_marker()
    while True:
        try:
            identity = marker.read_text().strip()
            if uuid.UUID(hex=identity).hex != identity:
                raise RuntimeError("invalid vLLM joining writer-cohort identity")
            path = _directory() / identity
            fd = acquire_writer_guard(path)
            try:
                if _leader_is_live(path) and marker.read_text().strip() == identity:
                    _writer_fds.append(fd)
                    _held_in_pid = (os.getpid(), path)
                    os.environ[_COHORT_ENV] = str(path)
                    return path
            except BaseException:
                os.close(fd)
                raise
            os.close(fd)
        except (FileNotFoundError, WriterCohortRetired):
            # Missing/stale/retired markers are normal when ranks start first.
            pass
        if time.monotonic() >= stop:
            raise RuntimeError("timed out waiting for a live vLLM writer-cohort leader")
        time.sleep(0.01)


def hold_writer_guard() -> None:
    """Join the boot cohort before this process can initialize shared KV."""
    raw = os.environ.get(_COHORT_ENV)
    if not raw:
        raise RuntimeError("vLLM shared KV started without a writer-cohort identity")
    path = Path(raw)
    if path.parent != _directory() or uuid.UUID(hex=path.name).hex != path.name:
        raise RuntimeError("invalid vLLM writer-cohort identity")
    _hold(path)


def join_writer_cohort_process() -> None:
    """Arm parent death around a potentially blocking cohort join."""
    # The launcher may itself run a worker. It already holds this boot's guard
    # and is not a multiprocessing child requiring a parent-death signal.
    raw = os.environ.get(_COHORT_ENV)
    if raw and _held_in_pid == (os.getpid(), Path(raw)):
        return
    parent = multiprocessing.parent_process()
    if parent is None:
        raise RuntimeError("GMS writer lacks a launch-time multiprocessing parent")
    # multiprocessing retains the PID captured by the launcher, even if this
    # child's first execution happens after reparenting. Never sample getppid()
    # as the expected identity. Non-direct launch methods fail closed below.
    expected_parent = parent.pid
    if expected_parent is None:
        raise RuntimeError("GMS writer lacks a launch-time parent PID")
    arm_parent_death_signal(expected_parent_pid=expected_parent)
    hold_writer_guard()
    # The successor may have held the old cohort exclusively while this late
    # child waited. Recheck the original parent before any CUDA initialization.
    arm_parent_death_signal(expected_parent_pid=expected_parent)


async def fence_predecessor_writers() -> Path | None:
    """Retire predecessor CPU submitters; GPU completion is a separate contract."""
    current = prepare_writer_cohort()
    marker = current.parent / "active"
    try:
        previous = marker.read_text().strip()
    except FileNotFoundError:
        previous = None
    predecessor = None
    if previous is not None and previous != current.name:
        if uuid.UUID(hex=previous).hex != previous:
            raise RuntimeError("invalid vLLM predecessor writer-cohort identity")
        predecessor = current.parent / previous
        await retire_writer_cohort(predecessor)
    pending = current.parent / (current.name + ".active")
    pending.write_text(current.name)
    os.replace(pending, marker)
    return predecessor
