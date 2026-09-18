# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import os
import select
import signal
import sys
import time
from pathlib import Path

import pytest
from gpu_memory_service.integrations.common.process_lifecycle import (
    arm_parent_death_signal,
)
from gpu_memory_service.integrations.vllm import writer_lifecycle as lifecycle

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux writer fencing")


@pytest.fixture(autouse=True)
def isolated_cohort(tmp_path, monkeypatch):
    monkeypatch.setenv("FAILOVER_LOCK_PATH", str(tmp_path / "failover.lock"))
    monkeypatch.setenv("ENGINE_ID", "primary")
    monkeypatch.delenv("GMS_VLLM_WRITER_COHORT_PATH", raising=False)
    monkeypatch.setattr(lifecycle, "_boot", None)
    monkeypatch.setattr(lifecycle, "_held_in_pid", None)
    descriptors = []
    monkeypatch.setattr(lifecycle, "_writer_fds", descriptors)
    yield
    for fd in descriptors:
        os.close(fd)


def _read(fd, count=1):
    readable, _, _ = select.select([fd], [], [], 3.0)
    assert readable, "child process did not report progress"
    return os.read(fd, count)


@pytest.mark.parametrize(
    "name", ["DYN_GMS_FAILOVER_SHADOW_MODE", "DYN_VLLM_GMS_SHADOW_MODE"]
)
def test_writer_cohort_requirement_accepts_both_failover_switches(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    assert lifecycle.writer_cohort_required() is True


def test_writer_cohort_requirement_treats_false_strings_as_disabled(monkeypatch):
    monkeypatch.setenv("DYN_GMS_FAILOVER_SHADOW_MODE", "0")
    monkeypatch.setenv("DYN_VLLM_GMS_SHADOW_MODE", "false")
    assert lifecycle.writer_cohort_required() is False


def test_headless_rank_joins_leader_boot_identity():
    leader = lifecycle.prepare_writer_cohort()
    os.environ.pop("GMS_VLLM_WRITER_COHORT_PATH")
    joined = lifecycle.join_prepared_writer_cohort(timeout=0.1)
    assert joined == leader
    assert os.environ["GMS_VLLM_WRITER_COHORT_PATH"] == str(leader)


def test_process_join_rechecks_same_parent_after_guard(monkeypatch):
    lifecycle.prepare_writer_cohort()
    events = []
    monkeypatch.setattr(lifecycle.os, "getppid", lambda: 1234)
    monkeypatch.setattr(
        lifecycle,
        "arm_parent_death_signal",
        lambda *, expected_parent_pid: events.append(("arm", expected_parent_pid)),
    )
    monkeypatch.setattr(
        lifecycle, "hold_writer_guard", lambda: events.append(("hold", None))
    )

    lifecycle.join_writer_cohort_process()

    assert events == [("arm", 1234), ("hold", None), ("arm", 1234)]


def test_takeover_waits_for_orphaned_engine_core_writer():
    read_fd, write_fd = os.pipe()
    supervisor = os.fork()
    if supervisor == 0:
        os.close(read_fd)
        guard = lifecycle.prepare_writer_cohort()
        asyncio.run(lifecycle.fence_predecessor_writers())
        expected_parent = os.getpid()
        writer = os.fork()
        if writer == 0:

            def delayed_exit(*_):
                time.sleep(0.65)
                os._exit(0)

            signal.signal(signal.SIGUSR1, delayed_exit)
            arm_parent_death_signal(signal.SIGUSR1, expected_parent_pid=expected_parent)
            lifecycle._hold(guard)
            os.write(write_fd, b"R")
            signal.pause()
            os._exit(0)
        os.close(write_fd)
        signal.pause()
        os._exit(1)
    os.close(write_fd)
    try:
        assert _read(read_fd) == b"R"
        marker = Path(os.environ["FAILOVER_LOCK_PATH"] + ".vllm-writers") / "active"
        previous = marker.read_text()
        os.kill(supervisor, signal.SIGKILL)
        os.waitpid(supervisor, 0)
        supervisor = 0

        async def verify():
            fence = asyncio.create_task(lifecycle.fence_predecessor_writers())
            await asyncio.sleep(0.30)
            assert not fence.done(), "elapsed delay is not proof of CUDA-writer exit"
            assert marker.read_text() == previous
            await asyncio.wait_for(fence, 3.0)
            assert marker.read_text() != previous

        asyncio.run(verify())
        assert _read(read_fd) == b""
    finally:
        os.close(read_fd)
        if supervisor:
            os.kill(supervisor, signal.SIGKILL)
            os.waitpid(supervisor, 0)
