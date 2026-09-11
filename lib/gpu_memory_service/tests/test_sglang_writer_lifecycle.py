# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-level proof that parent flock release is not a writer-exit fence."""

import asyncio
import os
import pickle
import select
import signal
import sys
import time
from pathlib import Path

import pytest
from gpu_memory_service.integrations.common.process_lifecycle import (
    arm_parent_death_signal,
)
from gpu_memory_service.integrations.sglang import writer_lifecycle as lifecycle

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux writer fencing")


@pytest.fixture(autouse=True)
def isolated_cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILOVER_LOCK_PATH", str(tmp_path / "failover.lock"))
    monkeypatch.setattr(lifecycle, "_boot", None)
    descriptors = []
    monkeypatch.setattr(lifecycle, "_writer_fds", descriptors)
    yield
    for fd in descriptors:
        os.close(fd)


def _read(fd, count=1):
    readable, _, _ = select.select([fd], [], [], 3.0)
    assert readable, "child process did not report progress"
    return os.read(fd, count)


def test_parent_only_death_waits_for_delayed_writer_exit():
    read_fd, write_fd = os.pipe()
    supervisor = os.fork()
    if supervisor == 0:
        os.close(read_fd)
        guard = lifecycle.prepare_writer_cohort()
        asyncio.run(lifecycle.fence_predecessor_writers())
        expected_parent = os.getpid()
        writer = os.fork()
        if writer == 0:
            # A catchable death signal simulates delayed child termination.
            # Production uses SIGKILL; the flock must not rely on its speed.
            def delayed_exit(*_):
                time.sleep(0.65)
                os._exit(0)

            signal.signal(signal.SIGUSR1, delayed_exit)
            arm_parent_death_signal(signal.SIGUSR1, expected_parent_pid=expected_parent)
            lifecycle._hold_writer_guard(guard)
            os.write(write_fd, b"R")
            signal.pause()
            os._exit(0)
        os.close(write_fd)
        signal.pause()
        os._exit(1)
    os.close(write_fd)
    try:
        assert _read(read_fd) == b"R"
        marker = Path(os.environ["FAILOVER_LOCK_PATH"] + ".writers") / "active"
        previous = marker.read_text()
        os.kill(supervisor, signal.SIGKILL)
        os.waitpid(supervisor, 0)
        supervisor = 0

        async def verify():
            fence = asyncio.create_task(lifecycle.fence_predecessor_writers())
            try:
                await asyncio.sleep(0.30)
                assert not fence.done(), "250ms elapsed is not proof of writer exit"
                assert marker.read_text() == previous
                await asyncio.wait_for(fence, 3.0)
                assert marker.read_text() != previous
            finally:
                if not fence.done():
                    fence.cancel()
                    await asyncio.gather(fence, return_exceptions=True)

        asyncio.run(verify())
        assert _read(read_fd) == b""
    finally:
        os.close(read_fd)
        if supervisor:
            os.kill(supervisor, signal.SIGKILL)
            os.waitpid(supervisor, 0)


def test_expected_parent_died_before_child_entry():
    read_fd, write_fd = os.pipe()
    trigger_read, trigger_write = os.pipe()
    supervisor = os.fork()
    if supervisor == 0:
        os.close(read_fd)
        os.close(trigger_write)
        expected_parent = os.getpid()
        child = os.fork()
        if child == 0:
            os.write(write_fd, b"R")
            os.read(trigger_read, 1)
            arm_parent_death_signal(expected_parent_pid=expected_parent)
            os.write(write_fd, b"BAD")
            os._exit(1)
        os.close(write_fd)
        signal.pause()
        os._exit(1)
    os.close(write_fd)
    os.close(trigger_read)
    try:
        assert _read(read_fd) == b"R"
        os.kill(supervisor, signal.SIGKILL)
        os.waitpid(supervisor, 0)
        supervisor = 0
        os.write(trigger_write, b"G")
        assert _read(read_fd, 3) == b""
    finally:
        os.close(read_fd)
        os.close(trigger_write)
        if supervisor:
            os.kill(supervisor, signal.SIGKILL)
            os.waitpid(supervisor, 0)


def _fake_scheduler():
    return "scheduler"


def test_engine_uses_picklable_guarded_scheduler_entry():
    class Engine:
        run_scheduler_process_func = staticmethod(_fake_scheduler)

        def __init__(self, *, server_args):
            self.server_args = server_args

    engine = lifecycle.create_guarded_engine(Engine, server_args="args")
    entry = pickle.loads(pickle.dumps(engine.run_scheduler_process_func))
    assert entry.func is lifecycle.run_guarded_scheduler
    assert entry.args[0] == os.getpid()
    assert entry.args[2] is _fake_scheduler
    assert engine.server_args == "args"


def test_cancelled_fence_does_not_publish_successor():
    directory = Path(os.environ["FAILOVER_LOCK_PATH"] + ".writers")
    directory.mkdir()
    import fcntl
    import uuid

    previous = uuid.uuid4().hex
    predecessor = directory / previous
    predecessor.touch()
    (directory / "active").write_text(previous)
    with predecessor.open("r+") as held:
        fcntl.flock(held, fcntl.LOCK_SH)

        async def verify():
            task = asyncio.create_task(lifecycle.fence_predecessor_writers())
            await asyncio.sleep(0.02)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(verify())
        assert (directory / "active").read_text() == previous
