# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cleanup of the shared memory a killed engine leaves in /dev/shm."""

import os
import signal
import subprocess
import sys

import pytest

from dynamo.vllm import engine_shm

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0]

# Creates a segment the way vLLM's queues do, reports its name and waits to be
# killed. Its resource tracker is in its process group and dies with it.
_ENGINE = """
import sys
from multiprocessing import shared_memory
shm = shared_memory.SharedMemory(create=True, size=4096)
print(shm.name, flush=True)
sys.stdin.read()
"""


class FakeHost:
    """A fake /proc and /dev/shm."""

    def __init__(self, root):
        self.proc, self.shm = root / "proc", root / "shm"
        self.proc.mkdir()
        self.shm.mkdir()

    def process(self, pid, pgid, maps=(), comm=b"python3"):
        d = self.proc / str(pid)
        d.mkdir()
        stat = b"%d (%s) S 1 %d %d 0 -1 4194560\n" % (pid, comm, pgid, pgid)
        (d / "stat").write_bytes(stat)
        lines = [
            f"7f00000{i:05x}000-7f00000{i:05x}fff rw-s 00000000 00:1a "
            f"{self.inode(path)} {path}\n"
            for i, path in enumerate(maps)
        ]
        lines.append("7fff00000000-7fff00021000 rw-p 00000000 00:00 0 [stack]\n")
        (d / "maps").write_text("".join(lines))

    def segment(self, name):
        (self.shm / name).write_bytes(b"\0" * 16)
        return str(self.shm / name)

    @staticmethod
    def inode(path):
        path = path.removesuffix(" (deleted)")
        return os.stat(path).st_ino if os.path.exists(path) else 42

    def names(self):
        return sorted(p.name for p in self.shm.iterdir())


@pytest.fixture
def host(tmp_path):
    return FakeHost(tmp_path)


def _recorded(host, pgid):
    members = engine_shm.process_group_members(pgid, str(host.proc))
    return engine_shm.mapped_segments(members, str(host.proc), str(host.shm))


@pytest.mark.unit
def test_records_only_the_engine_groups_queues(host):
    queue, other_engines = host.segment("psm_engine"), host.segment("psm_other")
    (host.shm / "psm_dir").mkdir()
    nested = host.segment("psm_dir/psm_nested")
    nccl = host.segment("nccl-shm")
    gone = str(host.shm / "psm_gone") + " (deleted)"
    # A command name with spaces, a parenthesis and a byte that is not UTF-8.
    host.process(100, pgid=100, maps=[queue, nccl, gone, nested], comm=b"a) b (\xff")
    host.process(101, pgid=100, maps=[queue, "/usr/lib/libc.so.6"])
    host.process(200, pgid=200, maps=[other_engines])

    assert _recorded(host, 100) == {"psm_engine": os.stat(queue).st_ino}


@pytest.mark.unit
def test_removes_recorded_segments_once_unmapped(host):
    left, in_use = host.segment("psm_left"), host.segment("psm_in_use")
    host.segment("psm_other")  # never recorded: another engine's
    host.process(100, pgid=100, maps=[left, in_use])
    recorded = _recorded(host, 100)
    # The engine is gone; a process of this container still maps one segment.
    (host.proc / "100" / "maps").unlink()
    (host.proc / "100" / "stat").unlink()
    (host.proc / "100").rmdir()
    host.process(300, pgid=300, maps=[in_use])

    removed = engine_shm.remove_segments(recorded, str(host.proc), str(host.shm))

    assert removed == ["psm_left"]
    assert host.names() == ["psm_in_use", "psm_other"]


@pytest.mark.unit
def test_keeps_a_name_that_now_belongs_to_another_file(host):
    host.process(100, pgid=100, maps=[host.segment("psm_reused")])
    recorded = _recorded(host, 100)
    # The engine unlinked its segment, and another container created the name.
    (host.shm / "psm_reused").rename(host.shm / "old")
    host.segment("psm_reused")

    assert engine_shm.remove_segments(recorded, str(host.proc), str(host.shm)) == []
    assert host.names() == ["old", "psm_reused"]


@pytest.mark.unit
def test_a_segment_that_cannot_be_removed_does_not_stop_the_rest(host, monkeypatch):
    recorded = {"psm_a": os.stat(host.segment("psm_a")).st_ino}
    recorded["psm_b"] = os.stat(host.segment("psm_b")).st_ino
    unlink = os.unlink

    def unlink_all_but_a(path):
        if path.endswith("psm_a"):
            raise PermissionError(13, "Permission denied")
        unlink(path)

    monkeypatch.setattr(engine_shm.os, "unlink", unlink_all_but_a)

    removed = engine_shm.remove_segments(recorded, str(host.proc), str(host.shm))

    assert removed == ["psm_b"]
    assert host.names() == ["psm_a"]


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_segments_of_a_killed_engine_are_found_and_removed():
    with subprocess.Popen(
        [sys.executable, "-c", _ENGINE],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as proc:
        path = None
        try:
            name = proc.stdout.readline().strip()
            assert name.startswith(engine_shm.SEGMENT_PREFIX)
            path = os.path.join(engine_shm.SHM_DIR, name)
            recorded = engine_shm.mapped_segments(
                engine_shm.process_group_members(proc.pid)
            )
            assert recorded == {name: os.stat(path).st_ino}
            assert engine_shm.remove_segments(recorded) == []  # still mapped
            # The whole group, resource tracker included, as a failed engine's is.
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            assert os.path.exists(path)
            assert engine_shm.remove_segments(recorded) == [name]
            assert not os.path.exists(path)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
            if path is not None and os.path.exists(path):
                os.unlink(path)
