# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fence every vLLM process that can enqueue work against shared KV HBM."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import os
import uuid
from pathlib import Path

from gpu_memory_service.integrations.common.process_lifecycle import (
    arm_parent_death_signal,
)

_COHORT_ENV = "GMS_VLLM_WRITER_COHORT_PATH"
_boot: tuple[int, Path] | None = None
_writer_fds: list[int] = []
_held_in_pid: tuple[int, Path] | None = None


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
    fd = os.open(path, os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
    except BaseException:
        os.close(fd)
        raise
    _writer_fds.append(fd)
    _held_in_pid = identity


def prepare_writer_cohort() -> Path:
    """Create a boot identity before vLLM starts EngineCore or CUDA workers."""
    global _boot, _held_in_pid
    if _boot is not None and _boot[0] == os.getpid():
        return _boot[1]
    directory = _directory()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / uuid.uuid4().hex
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
    except BaseException:
        os.close(fd)
        raise
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
    """Join the leader's boot identity from a headless TP node."""
    import time

    stop = time.monotonic() + max(0.0, timeout)
    marker = _joining_marker()
    while True:
        try:
            identity = marker.read_text().strip()
            if uuid.UUID(hex=identity).hex != identity:
                raise RuntimeError("invalid vLLM joining writer-cohort identity")
            path = _directory() / identity
            _hold(path)
            os.environ[_COHORT_ENV] = str(path)
            return path
        except FileNotFoundError:
            if time.monotonic() >= stop:
                raise RuntimeError("timed out waiting for vLLM writer-cohort identity")
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
    expected_parent = os.getppid()
    arm_parent_death_signal(expected_parent_pid=expected_parent)
    hold_writer_guard()
    # The successor may have held the old cohort exclusively while this late
    # child waited. Recheck the original parent before any CUDA initialization.
    arm_parent_death_signal(expected_parent_pid=expected_parent)


async def fence_predecessor_writers() -> None:
    """Wait for kernel release of every predecessor cohort descriptor."""
    current = prepare_writer_cohort()
    marker = current.parent / "active"
    try:
        previous = marker.read_text().strip()
    except FileNotFoundError:
        previous = None
    if previous is not None and previous != current.name:
        if uuid.UUID(hex=previous).hex != previous:
            raise RuntimeError("invalid vLLM predecessor writer-cohort identity")
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
    pending = current.parent / (current.name + ".active")
    pending.write_text(current.name)
    os.replace(pending, marker)
