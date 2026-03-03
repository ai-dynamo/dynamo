# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the flock-based failover lock.

No GPU required — these are pure Python/OS tests exercising flock
semantics across asyncio tasks and child processes.
"""

import asyncio
import multiprocessing
import os
import signal
import time

import pytest
import pytest_asyncio  # noqa: F401 — ensures the plugin is available

from gpu_memory_service.failover_lock.flock import FlockFailoverLock


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


def _racer(lock_path: str, engine_id: str, result_queue: multiprocessing.Queue):
    """Acquire the lock, report timing, hold briefly, release."""
    import fcntl

    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR)
    t0 = time.monotonic()
    fcntl.flock(fd, fcntl.LOCK_EX)
    t1 = time.monotonic()

    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, engine_id.encode())

    result_queue.put({"engine_id": engine_id, "wait_s": t1 - t0})

    # Hold the lock briefly
    time.sleep(0.2)
    os.close(fd)


@pytest.mark.asyncio
async def test_cross_process_race(lock_path):
    """Two processes race. Exactly one wins first, the other acquires after."""
    result_queue = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=_racer, args=(lock_path, "p1", result_queue))
    p2 = multiprocessing.Process(target=_racer, args=(lock_path, "p2", result_queue))

    p1.start()
    p2.start()

    p1.join(timeout=10)
    p2.join(timeout=10)

    results = []
    while not result_queue.empty():
        results.append(result_queue.get_nowait())

    assert len(results) == 2

    # Sort by wait time — the one with shorter wait won the race
    results.sort(key=lambda r: r["wait_s"])
    winner = results[0]
    loser = results[1]

    # Winner acquired almost immediately
    assert winner["wait_s"] < 0.1

    # Loser had to wait (winner held for 0.2s)
    assert loser["wait_s"] >= 0.1

    # Both finished — both eventually acquired
    assert {r["engine_id"] for r in results} == {"p1", "p2"}
