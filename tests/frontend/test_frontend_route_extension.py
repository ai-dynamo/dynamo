# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end test for the frontend route extension mechanism.

Launches ``python -m dynamo.frontend`` with ``--frontend-route-extension``
pointing at a custom route provider, then calls the custom route and validates
the response. No worker or model is needed — extension routes are served by the
frontend independently of the inference path.

Both selector forms are exercised:

* a **registered entry-point name** (via a throwaway ``.dist-info`` on the
  subprocess ``PYTHONPATH``, so the test installs nothing and leaves the shipped
  ``ai-dynamo`` metadata untouched), and
* a direct **``module:function``** path.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

from tests.frontend.route_extension_provider import (
    BLOCK_LOOP_PATH,
    ECHO_ASYNC_PATH,
    ECHO_MAX_BODY_BYTES,
    HELLO_BODY,
    HELLO_PATH,
    HOLD_PATH,
    LATE_PATH,
    OWN_TIMEOUT_PATH,
    READINESS_PATH,
    RELEASE_PATH,
    SLOW_PATH,
    STATE_PATH,
    UNBLOCK_LOOP_PATH,
)
from tests.utils.constants import DynamoPortRange
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.parallel,
    # Frontend-only: the custom route is served without a worker or GPU.
    pytest.mark.gpu_0,
]

ENTRY_POINT_NAME = "test-route-extension"
PROVIDER_TARGET = "tests.frontend.route_extension_provider:routes"
# tests/frontend/test_frontend_route_extension.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

# Keep the frontend fully standalone: mem discovery + TCP request plane +
# explicit zmq event plane. The explicit event plane is required — the runtime
# only treats the event plane as "no NATS" when it is set explicitly, so an
# ambient NATS_SERVER (CI sets one) would otherwise force a NATS connection.
# No etcd/NATS/worker is needed for a static custom route.
STANDALONE_ARGS = [
    "--discovery-backend",
    "mem",
    "--request-plane",
    "tcp",
    "--event-plane",
    "zmq",
]


def _pythonpath(*parts: str) -> str:
    entries = [p for p in parts if p]
    if os.environ.get("PYTHONPATH"):
        entries.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(entries)


def _register_entry_point(
    tmp_path: Path, name: str = ENTRY_POINT_NAME, target: str = PROVIDER_TARGET
) -> None:
    """Write a throwaway ``.dist-info`` registering ``name -> target`` under the
    ``dynamo.frontend.routes`` group, without installing a package."""
    dist_info = tmp_path / "route_ext_test-0.0.0.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: route-ext-test\nVersion: 0.0.0\n"
    )
    (dist_info / "entry_points.txt").write_text(
        f"[dynamo.frontend.routes]\n{name} = {target}\n"
    )


def _wait_for_state(base: str, done, what: str) -> None:
    """Poll the provider's state route until ``done(state)`` holds."""
    deadline = time.time() + 5
    while True:
        response = requests.get(f"{base}{STATE_PATH}", timeout=5)
        if response.status_code == 200 and done(response.json()):
            return
        assert time.time() < deadline, f"timed out waiting for {what}"
        time.sleep(0.02)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("selector", ["entry_point", "module_path"])
def test_frontend_route_extension_serves_custom_route(request, tmp_path, selector):
    if selector == "entry_point":
        _register_entry_point(tmp_path)
        extension = ENTRY_POINT_NAME
        pythonpath = _pythonpath(str(tmp_path), str(REPO_ROOT))
    else:
        # module:function path — no packaging, just an importable module.
        extension = PROVIDER_TARGET
        pythonpath = _pythonpath(str(REPO_ROOT))

    with DynamoFrontendProcess(
        request,
        frontend_port=0,
        extra_args=[*STANDALONE_ARGS, "--frontend-route-extension", extension],
        extra_env={"PYTHONPATH": pythonpath},
    ) as frontend:
        url = f"http://localhost:{frontend.frontend_port}{HELLO_PATH}"

        response = None
        deadline = time.time() + 60
        while time.time() < deadline:
            if not frontend.is_running():
                pytest.fail(f"frontend exited during startup:\n{frontend.read_logs()}")
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)

        assert (
            response is not None and response.status_code == 200
        ), f"custom route {url} never returned 200"
        assert response.json() == HELLO_BODY

        # Verify FrontendExtensionContext is wired through the PyO3 bridge.
        # No model is registered, so has_any_ready_model() must return False.
        ctx_response = requests.get(
            f"http://localhost:{frontend.frontend_port}{READINESS_PATH}", timeout=5
        )
        assert ctx_response.status_code == 200
        assert ctx_response.json() == {"ready": False}

        base = f"http://localhost:{frontend.frontend_port}"

        def post(path, **kwargs):
            return requests.post(f"{base}{path}", timeout=15, **kwargs)

        payload = [{"ticket_id": "t1"}, {"ticket_id": "t2"}]
        echo_async = requests.post(
            f"{base}{ECHO_ASYNC_PATH}",
            json=payload,
            headers={"Authorization": "Bearer test-token"},
            timeout=5,
        )
        assert echo_async.status_code == 200, echo_async.text
        assert echo_async.json() == {
            "body": payload,
            "authorization": "Bearer test-token",
            "runtime": "DistributedRuntime",
        }

        # A body over the route's max_body_bytes is rejected with a JSON 413.
        too_large = post(ECHO_ASYNC_PATH, data=b"x" * (ECHO_MAX_BODY_BYTES + 1))
        assert too_large.status_code == 413, too_large.text
        assert "error" in too_large.json()

        # The route's own timeout_s (0.5 s) bounds an async handler.
        slow = requests.post(f"{base}{SLOW_PATH}", json={}, timeout=10)
        assert slow.status_code == 503, slow.text
        # The event loop cancels the handler and answers, but the HTTP timer
        # can answer first, so the cancellation may land just after the 503.
        _wait_for_state(
            base, lambda s: s["slow_cancelled"], "the slow handler's cancellation"
        )

        # A TimeoutError the handler raises itself is a handler failure (500),
        # not the route's timeout (503).
        own_timeout = requests.post(f"{base}{OWN_TIMEOUT_PATH}", json={}, timeout=5)
        assert own_timeout.status_code == 500, own_timeout.text

        # A call the event loop picks up after its deadline never starts.
        with ThreadPoolExecutor(max_workers=1) as pool:
            blocked = pool.submit(post, BLOCK_LOOP_PATH)
            try:
                _wait_for_state(
                    base, lambda s: s["loop_blocked"], "the event loop to block"
                )
                assert post(LATE_PATH).status_code == 503
            finally:
                post(UNBLOCK_LOOP_PATH)
            assert blocked.result().status_code == 200
        # One round trip through the loop: a late handler would have run by now.
        assert post(ECHO_ASYNC_PATH, json={}).status_code == 200
        assert not requests.get(f"{base}{STATE_PATH}", timeout=5).json()["late_ran"]

        # A route admits 64 calls at once and sheds the rest (529), while the
        # other routes keep serving. The slots come back when the handlers end.
        with ThreadPoolExecutor(max_workers=64) as pool:
            held = [pool.submit(post, HOLD_PATH) for _ in range(64)]
            try:
                _wait_for_state(base, lambda s: s["held"] == 64, "64 held calls")
                assert post(HOLD_PATH).status_code == 529
                assert requests.get(url, timeout=5).status_code == 200
            finally:
                post(RELEASE_PATH)
            assert {call.result().status_code for call in held} == {200}
        assert post(HOLD_PATH).status_code == 200

        # The slots also come back when handlers are cancelled at their timeout.
        def slow_status(_=None):
            return post(SLOW_PATH).status_code

        with ThreadPoolExecutor(max_workers=64) as pool:
            assert set(pool.map(slow_status, range(64))) == {503}
        admitted_by = time.time() + 5
        while (status := slow_status()) == 529:
            assert time.time() < admitted_by, "the slow route's slots never came back"
            time.sleep(0.05)
        assert status == 503


@pytest.mark.timeout(120)
def test_frontend_route_extension_rejects_duplicate_routes(tmp_path):
    """Two routes with the same method+path must fail with a clean error, not a
    panic. All Python providers fold into one axum Router, and Router::route
    panics on an overlapping method+path; the loader must reject the collision
    up front instead."""
    _register_entry_point(
        tmp_path,
        name="dup",
        target="tests.frontend.route_extension_provider:duplicate_routes",
    )
    port = allocate_port(DynamoPortRange.FRONTEND.value)
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = _pythonpath(str(tmp_path), str(REPO_ROOT))
        # Standalone, NATS-free (see STANDALONE_ARGS); drop any ambient NATS_SERVER.
        env.pop("NATS_SERVER", None)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "dynamo.frontend",
                "--http-port",
                str(port),
                "--router-mode",
                "round-robin",
                *STANDALONE_ARGS,
                "--frontend-route-extension",
                "dup",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
    finally:
        deallocate_port(port)

    output = proc.stdout + proc.stderr
    assert proc.returncode != 0, f"frontend should have exited non-zero:\n{output}"
    assert "duplicate frontend route registered" in output, output
    # The clean error must replace the raw axum overlap panic.
    assert "Overlapping method route" not in output, output
