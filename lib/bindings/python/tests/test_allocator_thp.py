# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""dynamo._core keeps mimalloc's arenas off transparent huge pages unless opted in.

Each probe imports the extension in a fresh interpreter, because mimalloc reserves its
first arena during that import and reads MIMALLOC_* only when the library loads.
"""

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
        not Path("/sys/kernel/mm/transparent_hugepage/enabled").exists(),
        reason="needs Linux transparent huge pages",
    ),
]

# mimalloc reserves arenas of this size; a smaller advised total cannot hold one.
ARENA_MIB = 1024

_PROBE = """
import ctypes, json
PR_GET_THP_DISABLE = 42
prctl = ctypes.CDLL(None).prctl
thp_disabled_before = prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0)
import dynamo._core
advised = size = 0
with open("/proc/self/smaps") as smaps:
    for line in smaps:
        fields = line.split()
        if not fields[0].endswith(":"):
            start, end = (int(bound, 16) for bound in fields[0].split("-"))
            size = end - start
        elif fields[0] == "VmFlags:" and "hg" in fields[1:]:
            advised += size
print(json.dumps({
    "advised_mib": advised >> 20,
    "thp_disabled": [thp_disabled_before, prctl(PR_GET_THP_DISABLE, 0, 0, 0, 0)],
}))
"""


def _probe(allow_thp: str | None) -> dict:
    # mimalloc matches its variables case-insensitively. A jemalloc preload would take the
    # extension's Rust heap off mimalloc.
    env = {
        k: v
        for k, v in os.environ.items()
        if k != "LD_PRELOAD" and not k.upper().startswith("MIMALLOC_")
    }
    if allow_thp is not None:
        env["MIMALLOC_ALLOW_THP"] = allow_thp
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_mimalloc_arenas_skip_huge_pages_by_default():
    probe = _probe(None)

    assert probe["advised_mib"] < ARENA_MIB, probe
    # Only mimalloc's own arenas change; the process keeps the host's THP policy.
    assert probe["thp_disabled"][0] == probe["thp_disabled"][1], probe


def _strict_overcommit() -> bool:
    try:
        return Path("/proc/sys/vm/overcommit_memory").read_text().strip() == "2"
    except OSError:
        return False


@pytest.mark.skipif(
    _strict_overcommit(),
    reason="mimalloc never advises arenas it reserves under strict overcommit",
)
def test_mimalloc_allow_thp_still_opts_in():
    probe = _probe("1")

    assert probe["advised_mib"] >= ARENA_MIB, probe
