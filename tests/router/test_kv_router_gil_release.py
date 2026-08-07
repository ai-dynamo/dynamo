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

Releasing the GIL makes that wait *supervisable*, not cancellable -- ``block_on``
still runs to completion, so the builder thread stays parked for the life of the
process. What changes is that everyone else can now run, which is the whole point.

The scenario runs in a subprocess because a regression leaves a thread parked in
the constructor holding the GIL -- which would wedge the pytest worker itself,
uninterruptibly. Isolating it means a regression fails one test instead of
hanging the run.
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
    import asyncio, os, threading, time

    os.environ["DYN_ROUTER_MIN_INITIAL_WORKERS"] = "1"
    from dynamo._core import DistributedRuntime, KvRouter, KvRouterConfig

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runtime = DistributedRuntime(loop, "mem", "tcp")

    # Namespace with no registered workers, so the startup wait never completes.
    endpoint = runtime.endpoint("gilcheck.backend.generate")

    # Surface constructor failures instead of swallowing them -- otherwise an
    # unrelated break (API change, bad endpoint) exits the thread immediately and
    # reports as a bogus GIL assertion below.
    router_error = []

    def build_router():
        try:
            KvRouter(endpoint, 16, KvRouterConfig())
        except BaseException as exc:
            router_error.append(exc)

    t = threading.Thread(target=build_router, daemon=True)
    t.start()
    time.sleep(1.0)

    # The premise is that the thread is parked inside the constructor. If it
    # already returned, this run proves nothing either way.
    if not t.is_alive():
        raise AssertionError(
            "router thread exited during setup, so the blocking path was never "
            f"exercised; scenario is invalid. error={router_error!r}"
        )

    # The interpreter must keep executing bytecode while the router thread blocks.
    ticks = 0
    start = time.monotonic()
    while time.monotonic() - start < 3.0:
        ticks += 1
        time.sleep(0.001)
    assert ticks > 100, f"main thread starved ({ticks} ticks): GIL held across block_on"

    # A free GIL is the whole property: SIGALRM delivery, Thread.join(timeout=)
    # returning, and pytest-timeout firing are all downstream of it, so asserting
    # them separately would only re-prove the line above.

    # A constructor failure would also free the main thread; report it rather than
    # letting it pass as a successful run.
    assert not router_error, f"router constructor raised: {router_error!r}"

    print("OK")
    """
)


@pytest.mark.timeout(120)  # outer bound; must exceed the child deadline above
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
