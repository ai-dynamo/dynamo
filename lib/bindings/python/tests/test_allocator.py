# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""dynamo._core serves its Rust allocations from its own jemalloc (the default `jemalloc` feature).

Each probe runs in a fresh interpreter, because jemalloc reads its options once, when the
extension loads (jemalloc initializes in its library constructor). A probe grows a RadixTree, whose nodes live on the Rust
heap, and reports how much of that growth glibc's malloc and a preloaded jemalloc saw.
"""

import ctypes.util
import json
import os
import shutil
import subprocess
import sys

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
BACKGROUND_THREAD = "jemalloc_bg_thd"

_PROBE = """
import ctypes, json, os, sys, time

from dynamo.llm import RadixTree

BLOCKS = int(sys.argv[1])
EXPECT_BACKGROUND_THREADS = sys.argv[2] == "1"
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


def preloaded_jemalloc_allocated():
    # The extension's jemalloc is prefixed and keeps its symbols local, so only a
    # preloaded jemalloc answers here.
    if not hasattr(libc, "mallctl"):
        return None
    epoch = ctypes.c_uint64(1)
    libc.mallctl(b"epoch", None, None, ctypes.byref(epoch), ctypes.c_size_t(8))
    allocated = ctypes.c_size_t()
    size = ctypes.c_size_t(ctypes.sizeof(allocated))
    if libc.mallctl(b"stats.allocated", ctypes.byref(allocated), ctypes.byref(size), None, 0):
        return None
    return allocated.value


def rss():
    with open("/proc/self/status") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) << 10


def sample():
    return {"glibc": glibc_in_use(), "jemalloc": preloaded_jemalloc_allocated(), "rss": rss()}


def background_threads():
    count = 0
    for task in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{task}/comm") as comm:
                count += comm.read().strip() == "jemalloc_bg_thd"
        except FileNotFoundError:
            pass
    return count


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

# jemalloc starts its first background thread when the extension loads, and the thread
# names itself once it runs, so a short grace period suffices when none is expected.
deadline = time.monotonic() + (5 if EXPECT_BACKGROUND_THREADS else 0.5)
while not background_threads() and time.monotonic() < deadline:
    time.sleep(0.05)

growth = {
    name: None if after[name] is None else (after[name] - before[name]) >> 20
    for name in after
}
print(json.dumps({"growth_mib": growth, "background_threads": background_threads()}))
"""


def _probe(
    env_overrides: dict[str, str] | None = None, expect_background_threads: bool = True
) -> dict:
    # Drop allocator settings inherited from the test environment.
    env = {
        k: v
        for k, v in os.environ.items()
        if k
        not in (
            "DYN_FRONTEND_JEMALLOC",
            "LD_PRELOAD",
            "MALLOC_CONF",
            "_RJEM_MALLOC_CONF",
        )
    }
    env.update(env_overrides or {})
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _PROBE,
            str(BLOCKS),
            "1" if expect_background_threads else "0",
        ],
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


def test_rust_allocations_bypass_glibc():
    _require_mallinfo2()
    probe = _probe()

    assert probe["growth_mib"]["rss"] >= TREE_MIB, probe
    assert probe["growth_mib"]["glibc"] < TREE_MIB, probe


def test_jemalloc_starts_background_threads():
    probe = _probe()

    assert probe["background_threads"] >= 1, probe


def test_rjem_malloc_conf_overrides_the_built_in_options():
    probe = _probe(
        {"_RJEM_MALLOC_CONF": "background_thread:false"},
        expect_background_threads=False,
    )

    assert probe["background_threads"] == 0, probe


def test_preloaded_jemalloc_leaves_rust_allocations_on_the_extension_jemalloc():
    jemalloc = ctypes.util.find_library("jemalloc")
    if not jemalloc:
        pytest.skip("needs libjemalloc")
    probe = _probe({"LD_PRELOAD": jemalloc, "MALLOC_CONF": "background_thread:false"})

    assert probe["growth_mib"]["jemalloc"] is not None, "jemalloc did not load"
    assert probe["growth_mib"]["rss"] >= TREE_MIB, probe
    assert probe["growth_mib"]["jemalloc"] < TREE_MIB, probe
    # The preload's own background threads are off, so these are the extension's.
    assert probe["background_threads"] >= 1, probe


def _extension_path() -> str:
    # Imported here, not at module scope, so a missing build fails only these tests.
    import dynamo._core

    return dynamo._core.__file__


def _readelf(*args: str) -> str:
    readelf = shutil.which("readelf")
    if not readelf:
        pytest.skip("needs readelf")
    return subprocess.run(
        [readelf, "-W", *args, _extension_path()],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_extension_loads_without_static_tls():
    # A dlopen'ed library that uses initial-exec TLS draws on a small static TLS reserve
    # and can fail with "cannot allocate memory in static TLS block".
    assert "STATIC_TLS" not in _readelf("--dynamic")


def test_extension_exports_no_allocator_entry_points():
    entry_points = {
        "malloc",
        "calloc",
        "realloc",
        "free",
        "posix_memalign",
        "aligned_alloc",
        "memalign",
        "valloc",
        "malloc_usable_size",
        "mallctl",
    }
    defined = set()
    for line in _readelf("--dyn-syms").splitlines():
        fields = line.split()
        # Num: Value Size Type Bind Vis Ndx Name; undefined symbols have Ndx UND.
        if len(fields) >= 8 and fields[6] != "UND":
            defined.add(fields[7].split("@")[0])

    exported = sorted(
        name for name in defined if name in entry_points or name.startswith("_rjem_")
    )
    assert not exported, exported
