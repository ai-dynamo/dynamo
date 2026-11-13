# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A minimal frontend route provider for the route-extension e2e test.

The e2e test registers this provider under the ``dynamo.frontend.routes``
entry-point group (via a throwaway ``.dist-info`` on ``PYTHONPATH``, so no real
package install is required) and launches::

    python -m dynamo.frontend --frontend-route-extension <name>

then calls ``/hello_world`` and validates the response. ``FrontendRoute`` is
imported lazily inside ``routes()`` so this module stays importable (for the
shared constants below) without the ai-dynamo wheel present.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

HELLO_PATH = "/hello_world"
HELLO_BODY = {"message": "hello world!"}

READINESS_PATH = "/test/ctx/ready"
ECHO_ASYNC_PATH = "/test/echo_async"
SLOW_PATH = "/test/slow"
OWN_TIMEOUT_PATH = "/test/own_timeout"
STATE_PATH = "/test/state"

# Handler side effects the test reads back through STATE_PATH.
_state = {"slow_cancelled": False}


def _hello(ctx: Any) -> dict:
    """Static handler — independent of any model or worker."""
    return dict(HELLO_BODY)


def _readiness(ctx: Any) -> dict:
    """Exercises FrontendExtensionContext — returns live model-readiness state."""
    return {"ready": ctx.has_any_ready_model()}


async def _echo_async(ctx: Any) -> dict:
    """Async POST handler, run on the frontend's event loop."""
    await asyncio.sleep(0)
    return {
        "body": json.loads(ctx.body),
        "authorization": ctx.headers.get("authorization"),
        "runtime": type(ctx.runtime).__name__,
    }


async def _slow(ctx: Any) -> dict:
    """Outlives its route's timeout, and records that the frontend cancelled it."""
    try:
        await asyncio.sleep(5)
    except asyncio.CancelledError:
        _state["slow_cancelled"] = True
        raise
    return {}


async def _own_timeout(ctx: Any) -> dict:
    """Raises its own TimeoutError long before the route's 30 s timeout."""
    async with asyncio.timeout(0.01):
        await asyncio.sleep(1)
    return {}


def _read_state(ctx: Any) -> dict:
    return dict(_state)


def routes():
    """Provider callable: return the custom routes to register on the frontend."""
    from dynamo.llm import FrontendRoute

    return [
        FrontendRoute("GET", HELLO_PATH, _hello),
        FrontendRoute("GET", READINESS_PATH, _readiness),
        FrontendRoute("POST", ECHO_ASYNC_PATH, _echo_async),
        FrontendRoute("POST", SLOW_PATH, _slow, timeout_s=0.5),
        FrontendRoute("POST", OWN_TIMEOUT_PATH, _own_timeout),
        FrontendRoute("GET", STATE_PATH, _read_state),
    ]


def duplicate_routes():
    """Provider that returns two routes for the same method+path.

    Used to check the loader rejects the collision with a clean error instead
    of letting the merged axum Router panic at startup.
    """
    from dynamo.llm import FrontendRoute

    return [
        FrontendRoute("GET", HELLO_PATH, _hello),
        FrontendRoute("GET", HELLO_PATH, _hello),
    ]
