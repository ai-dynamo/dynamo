# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify that KvRouter's initial-worker wait releases the GIL.

The scenario runs in a subprocess because a regression would pin the GIL and
hang the pytest worker.
"""

import os
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

_SUBPROCESS_TIMEOUT_S = 45


def _child_env() -> dict:
    env = os.environ.copy()
    env["DYN_ROUTER_MIN_INITIAL_WORKERS"] = "1"
    return env


_SCENARIO = textwrap.dedent(
    """
    import asyncio, faulthandler, threading

    # Fires before the parent deadline and does not need the GIL, so a regression
    # exits with thread stacks instead of only reporting an external kill.
    faulthandler.dump_traceback_later(30, exit=True)

    from dynamo._core import DistributedRuntime, KvRouter, KvRouterConfig

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    runtime = DistributedRuntime(loop, "mem", "tcp", event_plane="zmq")

    # Namespace with no registered workers, so the startup wait never completes.
    endpoint = runtime.endpoint("gilcheck.backend.generate")

    # Surface constructor failures instead of swallowing them -- otherwise an
    # unrelated break (API change, bad endpoint) exits the thread immediately and
    # reports as a bogus GIL assertion below.
    router_starting = threading.Event()
    router_error = []

    def build_router():
        try:
            router_starting.set()
            KvRouter(endpoint, 16, KvRouterConfig())
        except Exception as exc:
            router_error.append(exc)

    t = threading.Thread(target=build_router, daemon=True)
    t.start()
    assert router_starting.wait(timeout=5), "router thread did not reach KvRouter"

    # join() releases the GIL while waiting. With the binding fix it returns at
    # the timeout; with a regression it cannot resume and the watchdog exits.
    t.join(timeout=3)

    assert not router_error, f"router constructor raised: {router_error!r}"
    assert t.is_alive(), "router constructor returned before a worker registered"

    faulthandler.cancel_dump_traceback_later()
    print("OK")
    """
)


@pytest.mark.timeout(60)  # outer bound; must exceed the child deadline above
def test_kv_router_init_releases_gil():
    """A KvRouter that never finds workers must not wedge the interpreter."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _SCENARIO],
            capture_output=True,
            text=True,
            timeout=_SUBPROCESS_TIMEOUT_S,
            check=False,
            env=_child_env(),
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
