# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import fcntl
import multiprocessing
import os
import select
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from gpu_memory_service.integrations.common.process_lifecycle import (
    WriterCohortRetired,
    acquire_writer_guard,
    arm_parent_death_signal,
    retire_writer_cohort,
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
    monkeypatch.setattr(lifecycle, "_leader_fd", None)
    descriptors = []
    monkeypatch.setattr(lifecycle, "_writer_fds", descriptors)
    yield
    lifecycle._close_leader_guard()
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
    monkeypatch.setattr(lifecycle, "_held_in_pid", None)
    events = []
    monkeypatch.setattr(
        lifecycle.multiprocessing, "parent_process", lambda: SimpleNamespace(pid=1234)
    )
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


def test_main_process_without_launch_identity_fails_closed(monkeypatch):
    lifecycle.prepare_writer_cohort()
    monkeypatch.setattr(lifecycle, "_held_in_pid", None)
    monkeypatch.setattr(lifecycle.multiprocessing, "parent_process", lambda: None)
    with pytest.raises(RuntimeError, match="launch-time multiprocessing parent"):
        lifecycle.join_writer_cohort_process()


def test_launcher_already_holding_guard_needs_no_parent_signal(monkeypatch):
    lifecycle.prepare_writer_cohort()

    def unexpected_call(**kwargs):
        pytest.fail("launcher must not arm against an unrelated parent")

    monkeypatch.setattr(lifecycle, "arm_parent_death_signal", unexpected_call)
    lifecycle.join_writer_cohort_process()


def test_stale_joining_marker_rejected_even_with_surviving_writers(monkeypatch):
    old = lifecycle.prepare_writer_cohort()
    lifecycle._close_leader_guard()
    monkeypatch.setattr(lifecycle, "_boot", None)
    # Old writer guards still exist, but they must not impersonate a live leader.
    with pytest.raises(RuntimeError, match="live vLLM writer-cohort leader"):
        lifecycle.join_prepared_writer_cohort(timeout=0)
    new = lifecycle.prepare_writer_cohort()
    assert new != old
    assert lifecycle.join_prepared_writer_cohort(timeout=0) == new


def test_concurrent_liveness_probe_cannot_impersonate_leader(monkeypatch):
    path = lifecycle.prepare_writer_cohort()
    lifecycle._close_leader_guard()
    monkeypatch.setattr(lifecycle, "_boot", None)
    with path.with_suffix(".leader").open("r+") as another_probe:
        fcntl.flock(another_probe, fcntl.LOCK_SH)
        assert not lifecycle._leader_is_live(path)


def test_rank_started_first_waits_for_fresh_leader(monkeypatch):
    old = lifecycle.prepare_writer_cohort()
    lifecycle._close_leader_guard()
    monkeypatch.setattr(lifecycle, "_boot", None)
    created = []

    def start_new_leader(_delay):
        created.append(lifecycle.prepare_writer_cohort())

    monkeypatch.setattr(lifecycle.time, "sleep", start_new_leader)
    joined = lifecycle.join_prepared_writer_cohort(timeout=1)
    assert created == [joined]
    assert joined != old


def test_forked_child_does_not_keep_leader_admission_alive():
    lifecycle.prepare_writer_cohort()
    read_fd, write_fd = os.pipe()
    trigger_read, trigger_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(read_fd)
        os.close(trigger_write)
        os.write(write_fd, b"R")
        os.read(trigger_read, 1)
        try:
            lifecycle.join_prepared_writer_cohort(timeout=0)
        except RuntimeError:
            os.write(write_fd, b"X")
            os._exit(0)
        os.write(write_fd, b"BAD")
        os._exit(1)
    os.close(write_fd)
    os.close(trigger_read)
    try:
        assert _read(read_fd) == b"R"
        lifecycle._close_leader_guard()
        os.write(trigger_write, b"G")
        assert _read(read_fd, 3) == b"X"
        _, status = os.waitpid(child, 0)
        child = 0
        assert os.waitstatus_to_exitcode(status) == 0
    finally:
        os.close(read_fd)
        os.close(trigger_write)
        if child:
            os.kill(child, signal.SIGKILL)
            os.waitpid(child, 0)


def _delayed_mp_writer(write_fd, trigger_read):
    os.write(write_fd, b"R")
    os.read(trigger_read, 1)
    lifecycle.join_writer_cohort_process()
    os.write(write_fd, b"BAD")


def _mp_writer(connection):
    lifecycle.join_writer_cohort_process()
    connection.send((os.getpid(), lifecycle._held_in_pid))
    connection.close()


@pytest.mark.parametrize("method", ["fork", "spawn"])
def test_normal_multiprocessing_writer_joins(method):
    guard = lifecycle.prepare_writer_cohort()
    context = multiprocessing.get_context(method)
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_mp_writer, args=(send,))
    child.start()
    send.close()
    try:
        assert receive.poll(15), "writer did not join"
        pid, held = receive.recv()
        assert held == (pid, guard)
        child.join(timeout=5)
        assert child.exitcode == 0
    finally:
        receive.close()
        if child.is_alive():
            child.kill()
        child.join(timeout=5)
        child.close()


def test_original_parent_exits_before_first_multiprocessing_child_entry():
    read_fd, write_fd = os.pipe()
    trigger_read, trigger_write = os.pipe()
    supervisor = os.fork()
    if supervisor == 0:
        os.close(read_fd)
        os.close(trigger_write)
        lifecycle.prepare_writer_cohort()
        child = multiprocessing.get_context("fork").Process(
            target=_delayed_mp_writer, args=(write_fd, trigger_read)
        )
        child.start()
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


def test_retired_cohort_rejects_late_writer_and_can_be_fenced_again(tmp_path):
    path = tmp_path / "old-cohort"
    path.touch()
    asyncio.run(retire_writer_cohort(path))
    with pytest.raises(WriterCohortRetired):
        acquire_writer_guard(path)
    # An interrupted successor can retry safely without reopening admission.
    asyncio.run(retire_writer_cohort(path))
    with pytest.raises(WriterCohortRetired):
        acquire_writer_guard(path)


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
