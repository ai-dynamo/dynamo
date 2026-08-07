# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test: KvRouter construction must not hold the GIL.

``KvRouter.__new__`` blocks until ``min_initial_workers`` register. That wait is
unbounded by design, so the caller is responsible for its own deadline -- but the
caller can only enforce one if the GIL is free. When the binding held the GIL
across ``block_on`` (dynamo#12762), a router that never found workers wedged the
whole interpreter: ``Thread.join(timeout=)`` could not resume, ``@pytest.mark.timeout``
never fired, and only an external SIGKILL ended it. In CI that burned a full 4h
GPU slot per occurrence.

The scenario runs in a subprocess: it installs a SIGALRM handler and leaves a
thread parked inside the router build, neither of which is safe to do in-process
under pytest-timeout.
"""

import subprocess
import sys
import textwrap

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
]

# Long enough that a real hang cannot pass by finishing early, short enough that
# a regression fails the suite in seconds rather than at the CI step limit.
_SUBPROCESS_TIMEOUT_S = 90

_SCENARIO = textwrap.dedent(
    """
    import asyncio, os, signal, sys, threading, time

    os.environ["DYN_ROUTER_MIN_INITIAL_WORKERS"] = "1"
    from dynamo._core import DistributedRuntime, KvRouter, KvRouterConfig

    alarm_fired = threading.Event()

    def on_alarm(signum, frame):
        alarm_fired.set()
        raise TimeoutError

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runtime = DistributedRuntime(loop, "mem", "tcp")

    # Namespace with no registered workers, so the startup wait never completes.
    endpoint = runtime.endpoint("gilcheck.backend.generate")

    def build_router():
        try:
            KvRouter(endpoint, 16, KvRouterConfig())
        except BaseException:
            pass

    t = threading.Thread(target=build_router, daemon=True)
    t.start()
    time.sleep(1.0)

    # The interpreter must keep executing bytecode while the router thread blocks.
    ticks = 0
    start = time.monotonic()
    while time.monotonic() - start < 3.0:
        ticks += 1
        time.sleep(0.001)
    assert ticks > 100, f"main thread starved ({ticks} ticks): GIL held across block_on"

    # ...and a Python-level signal handler must still be reachable.
    signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, 2)
    try:
        while t.is_alive():
            t.join(timeout=2)
    except TimeoutError:
        pass
    signal.setitimer(signal.ITIMER_REAL, 0)
    assert alarm_fired.is_set(), "SIGALRM handler never ran: GIL held across block_on"

    print("OK")
    """
)


def test_kv_router_init_releases_gil():
    """A KvRouter that never finds workers must not wedge the interpreter."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SCENARIO],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            "KvRouter init pinned the GIL: the scenario hung and needed an external "
            "kill, which is the CI failure this guards against."
        )

    assert proc.returncode == 0, (
        f"scenario failed (rc={proc.returncode})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    assert "OK" in proc.stdout
