# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`DYN_DISABLE_JEMALLOC` switches `dynamo._core`'s Rust allocator to the system allocator."""

import os
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.skipif(
        not sys.platform.startswith("linux"), reason="jemalloc is built in on Linux"
    ),
]

# Imports the extension and reports whether jemalloc's background purge thread is running. jemalloc
# starts it when the extension loads, and the opt-out stops it at the first Rust allocation, during
# the import. The thread names itself after it starts, so the positive case polls.
PROBE = """
import glob, time
import dynamo._core

def running():
    names = set()
    for path in glob.glob("/proc/self/task/*/comm"):
        try:
            names.add(open(path).read().strip())
        except OSError:
            pass
    return "jemalloc_bg_thd" in names

deadline = time.monotonic() + 5
while not running() and time.monotonic() < deadline and {poll}:
    time.sleep(0.01)
print(running())
"""


@pytest.mark.parametrize(
    ("value", "jemalloc"), [(None, True), ("0", True), ("1", False)]
)
def test_disable_jemalloc(value, jemalloc):
    env = {k: v for k, v in os.environ.items() if k != "DYN_DISABLE_JEMALLOC"}
    if value is not None:
        env["DYN_DISABLE_JEMALLOC"] = value
    result = subprocess.run(
        [sys.executable, "-c", PROBE.format(poll=jemalloc)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout.strip() == str(jemalloc)
