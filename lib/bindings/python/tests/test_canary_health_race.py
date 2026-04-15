# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Reproduction test for DIS-1185: canary health check race condition.

Proves that when DYN_HEALTH_CHECK_ENABLED=true, the transport layer (push_endpoint)
eagerly sets endpoint to Ready, but the canary health check later overrides it to
NotReady — causing /health to flip from 200 to 503.

The root cause: push_endpoint and the canary health check are two competing writers
to the same endpoint health status. The transport layer optimistically sets Ready
before the canary has verified end-to-end functionality.

The fix: when canary health checks are enabled, the transport layer should NOT set
Ready. Only the canary should control readiness.

Run with:
    pytest lib/bindings/python/tests/test_canary_health_race.py -v -s
"""

import asyncio
import os
import socket

import pytest
import requests

# ── Environment must be set BEFORE any fixture creates a DistributedRuntime ──
# DRT reads config from env vars at construction time.


def _get_free_port(low=10000, high=32000):
    """Find an available port in the i16-safe range."""
    for _ in range(20):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        if low <= port <= high:
            return port
    # Fallback: bind to a specific range
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return min(port, high)


# Module-level port allocation (before fixtures run)
_SYSTEM_PORT = _get_free_port()

# Set health check env vars at import time so the runtime fixture picks them up.
# Save previous values so teardown_module can restore them (prevent env pollution).
_ENV_OVERRIDES = {
    "DYN_SYSTEM_PORT": str(_SYSTEM_PORT),
    "DYN_HEALTH_CHECK_ENABLED": "true",
    "DYN_CANARY_WAIT_TIME": "2",
    "DYN_HEALTH_CHECK_REQUEST_TIMEOUT": "1",
}
_SAVED_ENV = {}
for _key, _value in _ENV_OVERRIDES.items():
    _SAVED_ENV[_key] = os.environ.get(_key)
    os.environ[_key] = _value


def teardown_module(module):
    """Restore environment variables to prevent pollution of other test modules."""
    for key, prev in _SAVED_ENV.items():
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev


pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.pre_merge,
    pytest.mark.timeout(30),
]


async def never_responds(request, context):
    """
    Handler that sleeps forever. The canary health check sends a real request
    to this handler — it will always time out (DYN_HEALTH_CHECK_REQUEST_TIMEOUT=1s).
    """
    await asyncio.sleep(999)
    yield {"error": "should never reach here"}


def _poll_health(port, path="/health", timeout=2):
    """Poll health endpoint, return (status_code, body_dict) or None on connection error."""
    try:
        resp = requests.get(f"http://localhost:{port}{path}", timeout=timeout)
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text}
        return resp.status_code, body
    except requests.ConnectionError:
        return None, None


@pytest.mark.asyncio
async def test_canary_overrides_transport_ready(runtime):
    """
    Proves the DIS-1185 race condition:

    Timeline:
      t=0s   Endpoint registers → push_endpoint sets Ready → /health returns 200
      t=2s   Canary wait time expires, canary fires
      t=3s   Canary request times out (handler never responds) → sets NotReady
      t=3s+  /health returns 503

    Before fix: transport_status=200, then canary_status=503 (the race)
    After fix:  transport_status=503 from the start (transport doesn't set Ready)
    """
    port = _SYSTEM_PORT

    # Create endpoint and start serving in background.
    # serve_endpoint returns a Future, so wrap in a coroutine for create_task.
    endpoint = runtime.endpoint("test.backend.generate")

    async def serve():
        await endpoint.serve_endpoint(
            never_responds,
            health_check_payload={"prompt": "test", "max_tokens": 1},
        )

    serve_task = asyncio.create_task(serve())

    # Wait for system_status_server and endpoint registration
    await asyncio.sleep(2)

    # Phase 1: Right after endpoint starts — transport layer set Ready
    transport_status, transport_body = _poll_health(port)
    print(f"\n[Phase 1] After endpoint start: /health = {transport_status}")
    print(f"  body: {transport_body}")

    # Phase 2: Wait for canary to fire and timeout
    # canary_wait_time(2s) + request_timeout(1s) + buffer(2s) = 5s
    # But we already waited 2s, so wait 5 more
    await asyncio.sleep(5)

    # Phase 3: After canary failure
    canary_status, canary_body = _poll_health(port)
    print(f"[Phase 2] After canary timeout: /health = {canary_status}")
    print(f"  body: {canary_body}")

    # Also check /live — currently same handler as /health (Track 2 will fix this)
    live_status, live_body = _poll_health(port, path="/live")
    print(f"[Info]    /live = {live_status}")
    print(f"  body: {live_body}")

    # Cleanup
    if not serve_task.done():
        serve_task.cancel()
        try:
            await serve_task
        except asyncio.CancelledError:
            pass

    # ── Assertions ──
    # These prove the bug: transport eagerly set Ready, then canary overrode to NotReady.
    #
    # BEFORE FIX (current behavior):
    #   transport_status = 200 (push_endpoint set Ready)
    #   canary_status = 503 (canary overrode to NotReady)
    #
    # AFTER FIX (expected):
    #   transport_status = 503 (transport doesn't set Ready when canary is enabled)
    #   canary_status = 503 (canary can't succeed with non-responsive handler)
    assert transport_status is not None, (
        "Could not connect to system_status_server. "
        f"Is DYN_SYSTEM_ENABLED=true and DYN_SYSTEM_PORT={port} set?"
    )

    # With the fix: transport does NOT set Ready when canary is enabled.
    # Endpoint stays NotReady until canary verifies end-to-end (which can't
    # succeed here because never_responds never returns).
    assert transport_status == 503, (
        f"Expected /health=503 from start (transport should not set Ready "
        f"when canary is enabled), got {transport_status}."
    )
    assert canary_status == 503, (
        f"Expected /health=503 (canary can't succeed with non-responsive "
        f"handler), got {canary_status}."
    )
