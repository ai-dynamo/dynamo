# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import select
import signal
import sys

import pytest
from gpu_memory_service.integrations.common.process_lifecycle import (
    arm_parent_death_signal,
)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux PDEATHSIG test")
def test_shared_writer_dies_with_its_parent():
    read_fd, write_fd = os.pipe()
    supervisor = os.fork()
    if supervisor == 0:
        os.close(read_fd)
        writer = os.fork()
        if writer == 0:
            arm_parent_death_signal()
            os.write(write_fd, b"ready")
            signal.pause()
            os._exit(1)
        os.close(write_fd)
        signal.pause()
        os._exit(1)

    os.close(write_fd)
    try:
        readable, _, _ = select.select([read_fd], [], [], 2.0)
        assert readable and os.read(read_fd, 5) == b"ready"
        os.kill(supervisor, signal.SIGKILL)
        os.waitpid(supervisor, 0)
        supervisor = 0
        readable, _, _ = select.select([read_fd], [], [], 2.0)
        assert readable and os.read(read_fd, 1) == b""
    finally:
        os.close(read_fd)
        if supervisor:
            os.kill(supervisor, signal.SIGKILL)
            os.waitpid(supervisor, 0)


def _start_guard_holder(path):
    """Fork a child that joins ``path``'s cohort and parks until killed."""
    from gpu_memory_service.integrations.common.process_lifecycle import (
        acquire_writer_guard,
    )

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        os.setpgid(0, 0)
        acquire_writer_guard(path)
        # A grandchild inherits the locked descriptor across fork.
        if os.fork() == 0:
            signal.pause()
            os._exit(1)
        os.write(write_fd, b"ready")
        signal.pause()
        os._exit(1)
    os.close(write_fd)
    readable, _, _ = select.select([read_fd], [], [], 2.0)
    assert readable and os.read(read_fd, 5) == b"ready"
    os.close(read_fd)
    return pid


def _stop(pid):
    os.killpg(pid, signal.SIGKILL)
    os.waitpid(pid, 0)


def _retire(path, timeout):
    import asyncio

    from gpu_memory_service.integrations.common.process_lifecycle import (
        retire_writer_cohort,
    )

    async def run():
        await asyncio.wait_for(retire_writer_cohort(path), timeout)

    try:
        asyncio.run(run())
        return True
    except asyncio.TimeoutError:
        return False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc lock holders")
def test_guard_holders_include_inherited_descriptors(tmp_path):
    from gpu_memory_service.integrations.common import process_lifecycle as pl

    path = tmp_path / "cohort"
    path.touch()
    holder = _start_guard_holder(path)
    try:
        holders = pl._guard_lock_holders(path)
        assert holder in holders and len(holders) == 2
        assert not pl._user_space_dead(holder)
    finally:
        _stop(holder)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc lock holders")
def test_pod_fence_completes_once_every_holder_is_exiting(tmp_path, monkeypatch):
    from gpu_memory_service.integrations.common import process_lifecycle as pl

    path = tmp_path / "cohort"
    path.touch()
    holder = _start_guard_holder(path)
    try:
        monkeypatch.setenv("DYN_GMS_WRITER_COHORT_SCOPE", "pod")
        assert not _retire(path, 0.2)
        # Model holders whose threads have all entered do_exit but whose
        # descriptors are still held by driver teardown.
        monkeypatch.setattr(pl, "_user_space_dead", lambda pid: True)
        assert _retire(path, 1.0)
        assert path.read_bytes() == b"R"
        with pytest.raises(pl.WriterCohortRetired):
            pl.acquire_writer_guard(path)
    finally:
        _stop(holder)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc lock holders")
def test_live_member_joining_during_fast_fence_keeps_it_waiting(tmp_path, monkeypatch):
    from gpu_memory_service.integrations.common import process_lifecycle as pl

    path = tmp_path / "cohort"
    path.touch()
    exiting = _start_guard_holder(path)
    joiner = _start_guard_holder(path)
    try:
        monkeypatch.setenv("DYN_GMS_WRITER_COHORT_SCOPE", "pod")
        joiner_checks = iter([True])

        def dead(pid):
            # The first scan sees only exiting members; the joiner is live on
            # the re-check after the tombstone.
            if pid == joiner:
                return next(joiner_checks, False)
            return True

        monkeypatch.setattr(pl, "_user_space_dead", dead)
        assert not _retire(path, 0.3)
        assert path.read_bytes() == b"R"
    finally:
        _stop(joiner)
        _stop(exiting)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc lock holders")
def test_fast_fence_is_pod_scope_only(tmp_path, monkeypatch):
    from gpu_memory_service.integrations.common import process_lifecycle as pl

    path = tmp_path / "cohort"
    path.touch()
    holder = _start_guard_holder(path)
    try:
        monkeypatch.delenv("DYN_GMS_WRITER_COHORT_SCOPE", raising=False)
        monkeypatch.setattr(pl, "_user_space_dead", lambda pid: True)
        assert not _retire(path, 0.2)
        assert path.read_bytes() == b""
    finally:
        _stop(holder)
    assert _retire(path, 1.0)
    assert path.read_bytes() == b"R"


def test_killed_thread_blocked_in_a_driver_counts_as_exiting(tmp_path):
    from gpu_memory_service.integrations.common import process_lifecycle as pl

    def task(sigpnd, shdpnd="0000000000000000"):
        path = tmp_path / f"task-{sigpnd}-{shdpnd}"
        path.mkdir()
        (path / "status").write_text(
            f"State:\tD (disk sleep)\nSigPnd:\t{sigpnd}\nShdPnd:\t{shdpnd}\n"
        )
        return str(path)

    assert pl._sigkill_pending(task("0000000000000100"))
    assert pl._sigkill_pending(task("0000000000000000", "0000000000000100"))
    # SIGTERM (or any catchable signal) can still return to user space.
    assert not pl._sigkill_pending(task("0000000000004000"))
    assert not pl._user_space_dead(os.getpid())


@pytest.mark.skipif(
    sys.platform != "linux" or os.uname().machine not in ("x86_64", "aarch64"),
    reason="Linux thread-exit syscall",
)
def test_holder_found_after_its_main_thread_exits(tmp_path):
    """A zombie main thread hides /proc/<pid>/fd; other threads keep the lock."""
    import ctypes
    import threading

    from gpu_memory_service.integrations.common import process_lifecycle as pl

    path = tmp_path / "cohort"
    path.touch()
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        pl.acquire_writer_guard(path)
        threading.Thread(target=signal.pause, daemon=True).start()
        os.write(write_fd, b"ready")
        # exit(2) ends only the calling thread, unlike os._exit (exit_group).
        ctypes.CDLL(None).syscall({"x86_64": 60, "aarch64": 93}[os.uname().machine], 0)
    os.close(write_fd)
    try:
        readable, _, _ = select.select([read_fd], [], [], 2.0)
        assert readable and os.read(read_fd, 5) == b"ready"
        for _ in range(200):
            with open(f"/proc/{pid}/stat") as stat:
                if stat.read().rsplit(")", 1)[1].split()[0] == "Z":
                    break
            select.select([], [], [], 0.01)
        assert os.listdir(f"/proc/{pid}/fd") == []
        assert pl._guard_lock_holders(path) == {pid}
        assert not pl._user_space_dead(pid)
    finally:
        os.close(read_fd)
        os.kill(pid, signal.SIGKILL)
        os.waitpid(pid, 0)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux /proc/locks")
def test_lock_owners_count_even_without_a_visible_descriptor(tmp_path, monkeypatch):
    """Between closing descriptors and the final close, only /proc/locks shows
    the exiting owner. A live owner there must still block the fast fence."""
    from gpu_memory_service.integrations.common import process_lifecycle as pl

    path = tmp_path / "cohort"
    path.touch()
    holder = _start_guard_holder(path)
    try:
        assert holder in pl._lock_owner_pids(os.stat(path))
        monkeypatch.setattr(pl, "_guard_lock_holders", lambda _path: set())
        assert not pl._holders_all_exiting(path)
        monkeypatch.setattr(pl, "_user_space_dead", lambda pid: True)
        assert pl._holders_all_exiting(path)
        # A busy lock whose owner is not visible here never takes the fast path.
        monkeypatch.setattr(pl, "_lock_owner_pids", lambda _target: [])
        assert not pl._holders_all_exiting(path)
    finally:
        _stop(holder)
