# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the GlobalPlanner entrypoint's endpoint registration.

serve_endpoint returns a long-running future that only completes on shutdown.
Awaiting the two endpoints sequentially would block forever on the first and
never register the second — leaving system_health permanently NotReady and
the pod stuck at 0/1 Ready (operator-injected probes 503 indefinitely).

These tests guard against that regression by verifying:
1. Both endpoints' serve_endpoint are invoked (concurrent awaiting).
2. The health endpoint receives a health_check_payload so the runtime's
   system_health flag can flip to Ready.
"""

import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest

from dynamo.global_planner import __main__ as gp_main

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
    pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20"),
]


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_main_registers_both_endpoints_concurrently():
    """Both scale_request and health endpoints must be registered before the
    entrypoint blocks. Sequential awaits would leave health unregistered.
    """
    # Track which endpoint names get created and their serve_endpoint calls.
    created_endpoints: dict[str, MagicMock] = {}

    def make_endpoint(name: str) -> MagicMock:
        """Build a mock endpoint whose serve_endpoint never resolves."""
        ep = MagicMock()
        # serve_endpoint is a sync function that returns an awaitable (Rust's
        # future_into_py). Return a never-completing Future so ``await
        # ep.serve_endpoint(...)`` blocks forever — matching real behavior.
        # If the entrypoint awaits the endpoints sequentially, the first
        # call blocks forever and the second is never registered.
        ep.serve_endpoint = MagicMock(
            side_effect=lambda *a, **kw: asyncio.get_running_loop().create_future()
        )
        created_endpoints[name] = ep
        return ep

    runtime = MagicMock()
    runtime.endpoint = MagicMock(side_effect=make_endpoint)

    args = MagicMock()
    args.environment = "kubernetes"
    args.managed_namespaces = None
    args.no_operation = False
    args.max_total_gpus = -1

    with patch.dict(
        os.environ, {"DYN_NAMESPACE": "gp-ns", "POD_NAMESPACE": "default"}
    ), patch("dynamo.global_planner.__main__.ScaleRequestHandler") as mock_handler_cls:
        mock_handler_cls.return_value = MagicMock()

        # main never returns on its own (it awaits the endpoint futures
        # forever). Run it in a task, yield control so the gather starts, then
        # cancel and inspect what was registered.
        # Calling the underlying coroutine (the @dynamo_worker-wrapped `main`
        # expects a runtime injected by the decorator; we bypass it).
        task = asyncio.create_task(gp_main.main.__wrapped__(runtime, args))
        # Let the event loop run. With asyncio.gather both serve_endpoint
        # futures are scheduled in one step; with sequential awaits only
        # the first one is ever called, which is exactly the bug.
        for _ in range(50):
            await asyncio.sleep(0)
            if len(created_endpoints) >= 2 and all(
                ep.serve_endpoint.call_count >= 1 for ep in created_endpoints.values()
            ):
                break
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    # Both endpoints must have been created and served.
    scale_key = next(k for k in created_endpoints if k.endswith(".scale_request"))
    health_key = next(k for k in created_endpoints if k.endswith(".health"))
    assert created_endpoints[scale_key].serve_endpoint.called, (
        "scale_request endpoint was never served — regression: both "
        "endpoints must be awaited concurrently via asyncio.gather."
    )
    assert created_endpoints[health_key].serve_endpoint.called, (
        "health endpoint was never served — regression: sequential await on "
        "scale_request blocks forever and never yields control back to "
        "register the health endpoint."
    )

    # The health endpoint must register a health_check_payload so the
    # runtime's system_health flips to Ready.
    health_call = created_endpoints[health_key].serve_endpoint.call_args
    assert "health_check_payload" in health_call.kwargs, (
        "health endpoint must pass health_check_payload so system_health "
        "flips to Ready; without it the pod stays permanently NotReady."
    )
    payload = health_call.kwargs["health_check_payload"]
    assert isinstance(payload, dict) and "text" in payload, (
        f"health_check_payload must be a dict matching HealthCheckRequest "
        f"(has a 'text' field); got {payload!r}"
    )
