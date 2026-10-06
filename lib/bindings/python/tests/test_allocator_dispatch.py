# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A jemalloc preload moves dynamo._core's Rust allocations off mimalloc.

Each probe runs in a fresh interpreter, because the extension picks its allocator once, at
its first Rust allocation. A probe grows a RadixTree, whose nodes live on the Rust heap, and
reports how much of that growth the C library's malloc (glibc, or a preloaded jemalloc) saw.
"""

import ctypes.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="probes glibc and /proc"
    ),
]

BLOCKS = 400_000
# The tree holds well over this much once it stores BLOCKS blocks, and the Python side of
# the probe allocates far less than this from malloc while the tree grows.
TREE_MIB = 8
# mimalloc reserves arenas of this size; the system allocators reserve less at a time.
ARENA_MIB = 1024

_PROBE = """
import ctypes, json, sys

from dynamo.llm import RadixTree

BLOCKS, ARENA_MIB = (int(arg) for arg in sys.argv[1:])
libc = ctypes.CDLL(None)


class Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd",
        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost",
    )]


def glibc_in_use():
    if not hasattr(libc, "mallinfo2"):
        return None
    libc.mallinfo2.restype = Mallinfo2
    info = libc.mallinfo2()
    return info.uordblks + info.hblkhd


def jemalloc_allocated():
    if not hasattr(libc, "mallctl"):
        return None
    epoch = ctypes.c_uint64(1)
    libc.mallctl(b"epoch", None, None, ctypes.byref(epoch), ctypes.c_size_t(8))
    allocated = ctypes.c_size_t()
    size = ctypes.c_size_t(ctypes.sizeof(allocated))
    if libc.mallctl(b"stats.allocated", ctypes.byref(allocated), ctypes.byref(size), None, 0):
        return None
    return allocated.value


def sample():
    return {"glibc": glibc_in_use(), "jemalloc": jemalloc_allocated()}


blocks = [{"block_hash": i, "tokens_hash": i} for i in range(BLOCKS)]
event = json.dumps(
    {"event_id": 1, "data": {"stored": {"parent_hash": None, "blocks": blocks}}}
).encode()
del blocks

tree = RadixTree()
# Each find_matches waits for the tree's worker thread, so the tree is settled when sampled.
tree.find_matches([0])
before = sample()
tree.apply_event(0, event)
assert tree.find_matches([0]).scores, "the tree did not store the event"
after = sample()

growth = {
    name: None if after[name] is None else (after[name] - before[name]) >> 20
    for name in after
}
large_anonymous = 0
with open("/proc/self/maps") as maps:
    for line in maps:
        fields = line.split()
        if len(fields) > 5 and not fields[5].startswith("[anon:"):
            continue
        start, end = (int(bound, 16) for bound in fields[0].split("-"))
        large_anonymous += (end - start) >> 20 >= ARENA_MIB
print(json.dumps({"growth_mib": growth, "large_anonymous_maps": large_anonymous}))
"""


# The extension decides from the LD_PRELOAD entry's name alone. The loader skips a preload it
# cannot find, so this name moves the Rust heap to glibc, where mallinfo2 can see it.
MISSING_JEMALLOC = "/nonexistent/libjemalloc.so.2"


def _probe(preload: str | None = None, jemalloc_flag: str | None = None) -> dict:
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("DYN_FRONTEND_JEMALLOC", "LD_PRELOAD")
        and not k.upper().startswith("MIMALLOC_")
    }
    if preload is not None:
        env["LD_PRELOAD"] = preload
    if jemalloc_flag is not None:
        env["DYN_FRONTEND_JEMALLOC"] = jemalloc_flag
    result = subprocess.run(
        [sys.executable, "-c", _PROBE, str(BLOCKS), str(ARENA_MIB)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _require_mallinfo2():
    if not hasattr(ctypes.CDLL(None), "mallinfo2"):
        pytest.skip("needs glibc 2.33 or later for mallinfo2")


def _strict_overcommit() -> bool:
    try:
        return Path("/proc/sys/vm/overcommit_memory").read_text().strip() == "2"
    except OSError:
        return False


def test_rust_allocations_use_mimalloc_by_default():
    _require_mallinfo2()
    probe = _probe()

    assert probe["growth_mib"]["glibc"] < TREE_MIB, probe
    if not _strict_overcommit():
        # The arena marker the system-allocator tests rely on.
        assert probe["large_anonymous_maps"] >= 1, probe


@pytest.mark.parametrize(
    "preload",
    [
        MISSING_JEMALLOC,
        f"libother.so:{MISSING_JEMALLOC}",
        f"libother.so {MISSING_JEMALLOC}",
    ],
)
def test_jemalloc_preload_moves_rust_allocations_to_the_system_allocator(preload):
    _require_mallinfo2()
    probe = _probe(preload)

    assert probe["growth_mib"]["glibc"] >= TREE_MIB, probe
    # mimalloc reserves no arena when it serves nothing.
    assert probe["large_anonymous_maps"] == 0, probe


@pytest.mark.parametrize(
    "preload, jemalloc_flag",
    [
        ("/nonexistent/libjemalloc/libother.so", None),
        (None, "1"),
    ],
)
def test_rust_allocations_stay_on_mimalloc_without_a_jemalloc_preload(
    preload, jemalloc_flag
):
    # Only the preload counts, so processes that inherit the frontend's flag without the
    # preload keep mimalloc.
    _require_mallinfo2()
    probe = _probe(preload, jemalloc_flag)

    assert probe["growth_mib"]["glibc"] < TREE_MIB, probe


def test_preloaded_jemalloc_serves_rust_allocations():
    jemalloc = ctypes.util.find_library("jemalloc")
    if not jemalloc:
        pytest.skip("needs libjemalloc")
    probe = _probe(jemalloc)

    assert probe["growth_mib"]["jemalloc"] is not None, "jemalloc did not load"
    assert probe["growth_mib"]["jemalloc"] >= TREE_MIB, probe
    assert probe["large_anonymous_maps"] == 0, probe
