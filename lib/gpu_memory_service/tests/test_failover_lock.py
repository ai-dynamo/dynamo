# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flock-based failover lock.

These are pure Python/OS tests exercising flock semantics across asyncio
tasks and child processes, so they stay on the generic cpu-style pre-merge
lane instead of the dedicated GPU job.
"""

import asyncio
import multiprocessing
import os
import signal
import time

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.failover_lock.flock import FlockFailoverLock

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@pytest.fixture
def lock_path(tmp_path):
    return str(tmp_path / "failover.lock")


# ── Test 1: basic acquire / release ──────────────────────────────────


@pytest.mark.asyncio
async def test_acquire_release(lock_path):
    lock = FlockFailoverLock(lock_path)

    await lock.acquire("engine-0")

    # Lock file should contain the engine id
    with open(lock_path) as f:
        assert f.read().strip() == "engine-0"

    # Internal fd is open
    assert lock._fd is not None

    await lock.release()

    # fd is closed
    assert lock._fd is None


# ── Test 2: two-engine contention ────────────────────────────────────


@pytest.mark.asyncio
async def test_two_engines_contention(lock_path):
    """Engine A holds lock. Engine B blocks. A releases. B acquires."""
    lock_a = FlockFailoverLock(lock_path)
    lock_b = FlockFailoverLock(lock_path)

    await lock_a.acquire("engine-a")

    b_acquired = asyncio.Event()

    async def acquire_b():
        await lock_b.acquire("engine-b", poll_interval=0.01)
        b_acquired.set()

    task_b = asyncio.create_task(acquire_b())

    # Give B a few poll cycles — it should NOT acquire
    await asyncio.sleep(0.1)
    assert not b_acquired.is_set()

    # Release A — B should acquire
    await lock_a.release()
    await asyncio.wait_for(b_acquired.wait(), timeout=2.0)

    assert b_acquired.is_set()

    # Lock file should now show engine-b
    with open(lock_path) as f:
        assert f.read().strip() == "engine-b"

    await lock_b.release()
    task_b.cancel()


# ── Test 3: process death releases lock ──────────────────────────────


def _child_acquire_and_hang(lock_path: str, ready_fd: int):
    """Child process: acquire flock, signal parent, then block forever."""
    import fcntl

    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)
    os.write(fd, b"child")

    # Signal parent that we hold the lock
    os.write(ready_fd, b"1")
    os.close(ready_fd)

    # Block forever (parent will SIGKILL us)
    time.sleep(3600)


@pytest.mark.asyncio
async def test_process_death_releases(lock_path):
    """SIGKILL a child holding the lock. Parent should acquire."""
    read_fd, write_fd = os.pipe()

    child = multiprocessing.Process(
        target=_child_acquire_and_hang, args=(lock_path, write_fd)
    )
    child.start()
    os.close(write_fd)

    # Wait for child to signal it holds the lock
    os.read(read_fd, 1)
    os.close(read_fd)

    # Child holds the lock — verify we can't acquire immediately
    lock = FlockFailoverLock(lock_path)
    fd_check = os.open(lock_path, os.O_RDWR)
    try:
        import fcntl

        fcntl.flock(fd_check, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pytest.fail("Should not have acquired — child holds the lock")
    except BlockingIOError:
        pass  # expected
    finally:
        os.close(fd_check)

    # Destroy the child process — kernel releases the flock
    os.kill(child.pid, signal.SIGKILL)
    child.join(timeout=5)

    # Now parent should acquire
    await lock.acquire("parent", poll_interval=0.01)

    with open(lock_path) as f:
        assert f.read().strip() == "parent"

    await lock.release()


# ── Test 4: owner() ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_owner(lock_path):
    lock = FlockFailoverLock(lock_path)

    # No lock file yet
    assert await lock.owner() is None

    await lock.acquire("engine-x")
    assert await lock.owner() == "engine-x"

    await lock.release()

    # File still exists with stale content (flock is the authority, not file content)
    assert await lock.owner() == "engine-x"


@pytest.mark.asyncio
async def test_owner_separate_instance(lock_path):
    """owner() works from a different FlockFailoverLock instance."""
    lock_holder = FlockFailoverLock(lock_path)
    lock_observer = FlockFailoverLock(lock_path)

    await lock_holder.acquire("holder")
    assert await lock_observer.owner() == "holder"

    await lock_holder.release()


# ── Test 5: cross-process race ───────────────────────────────────────


# How long each racer keeps the lock; the 0.1 s wait floor asserted below is
# half of it, a 2x margin on an exact bound.
HOLD_S = 0.2

# Start p2 late enough that removing the parent gate recreates the original
# no-contention failure: p1 can finish its hold before p2 starts its probe.
START_STAGGER_S = 0.3


def _racer(
    lock_path: str,
    engine_id: str,
    ready_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
):
    """Confirm contention, acquire the lock, report timing, hold, and release."""
    import fcntl

    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)

    t0 = time.monotonic()
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        # A failed non-blocking flock proves this racer reached kernel
        # contention before the parent releases the gate.
        ready_queue.put(engine_id)
        fcntl.flock(fd, fcntl.LOCK_EX)

    t1 = time.monotonic()

    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, engine_id.encode())

    time.sleep(HOLD_S)

    # Stamped before the close, so the reported hold is a subset of the real
    # one; stamping after would let the other child acquire inside the gap.
    released_at = time.monotonic()
    os.close(fd)

    result_queue.put(
        {
            "engine_id": engine_id,
            "wait_s": t1 - t0,
            "acquired_at": t1,
            "released_at": released_at,
        }
    )


@pytest.mark.asyncio
# Backstop, not the primary bound: every wait below carries its own 10 s
# timeout, summing to 80 s, so those report a precise failure first. This
# catches a child or queue shutdown path that blocks outside those waits.
@pytest.mark.timeout(90)
async def test_cross_process_race(lock_path):
    """Two processes contend for the lock; the kernel serializes their holds."""
    import fcntl

    ready_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    p1 = multiprocessing.Process(
        target=_racer, args=(lock_path, "p1", ready_queue, result_queue)
    )
    p2 = multiprocessing.Process(
        target=_racer, args=(lock_path, "p2", ready_queue, result_queue)
    )

    try:
        gate_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
        try:
            fcntl.flock(gate_fd, fcntl.LOCK_EX)

            p1.start()
            assert ready_queue.get(timeout=10) == "p1"
            time.sleep(START_STAGGER_S)
            p2.start()
            assert ready_queue.get(timeout=10) == "p2"
        finally:
            # LOCK_UN, not os.close(gate_fd): children inherit the parent's
            # open file description, so closing only the parent copy would
            # leave the gate locked.
            fcntl.flock(gate_fd, fcntl.LOCK_UN)
            os.close(gate_fd)

        # Blocking gets rather than Queue.empty(): empty() is not a
        # synchronization primitive, and joining a child before draining its
        # queue can deadlock.
        results = [result_queue.get(timeout=10), result_queue.get(timeout=10)]

        p1.join(timeout=10)
        p2.join(timeout=10)
        assert p1.exitcode == 0
        assert p2.exitcode == 0
    finally:
        # Any get above can time out and skip the joins. A child left parked in
        # flock() would hold the lock file past the end of the test, and an
        # unclosed queue leaves its feeder thread running.
        for p in (p1, p2):
            if p.is_alive():
                p.terminate()
            if p.pid is not None:  # None when start() was never reached
                p.join(timeout=10)
        ready_queue.close()
        result_queue.close()

    # CLOCK_MONOTONIC is system-wide on Linux, so the two children's stamps are
    # comparable; fcntl.flock already makes this module Linux-only.
    results.sort(key=lambda r: r["acquired_at"])
    first, second = results

    # Mutual exclusion: the second acquirer did not get in before the first
    # one let go.
    assert second["acquired_at"] >= first["released_at"]

    assert second["wait_s"] >= 0.1

    # Both finished — both eventually acquired
    assert {r["engine_id"] for r in results} == {"p1", "p2"}
