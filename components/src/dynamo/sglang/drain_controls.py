# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental lifecycle controls on the worker system listener."""

import asyncio
import logging
from collections.abc import Mapping
from functools import partial

from dynamo.common.utils.graceful_shutdown import get_grace_period_seconds
from dynamo.sglang.drain import DrainConflict, WorkerDrain
from dynamo.sglang.engine_routes import parse_engine_route_descriptors

_LIFECYCLE_METHODS = {"drain": "POST", "resume": "POST", "status": "GET"}


def configure_worker_drain(
    handler,
    runtime,
    ready_event,
    shutdown_event,
    terminal_event,
    *,
    enable_admin=True,
):
    """Install LLM admission fencing, with optional administrative HTTP controls."""

    def pending_engine_requests() -> int | None:
        states = getattr(handler.engine.tokenizer_manager, "rid_to_state", None)
        return len(states) if isinstance(states, Mapping) else None

    controller = WorkerDrain(
        handler.generate_endpoint,
        lock=handler._engine_route_lock,
        shutdown_event=shutdown_event,
        terminal_event=terminal_event,
        ready_event=ready_event,
        is_paused=lambda: bool(handler.engine.tokenizer_manager.is_pause),
        grace_period=get_grace_period_seconds(),
        pending_engine_requests=pending_engine_requests if enable_admin else None,
    )

    handler._worker_drain = controller
    if not enable_admin or runtime.system_status_server_url() is None:
        return controller
    for descriptor in parse_engine_route_descriptors(
        handler.config.dynamo_args.engine_routes
    ):
        if descriptor.path.split("/", 1)[0] in _LIFECYCLE_METHODS:
            raise ValueError(
                f"/engine/{descriptor.path} is reserved by worker lifecycle controls"
            )

    endpoint_started = False
    discovery_client = None
    discovery_lock = asyncio.Lock()

    async def control(action, request):
        nonlocal endpoint_started, discovery_client
        method = _LIFECYCLE_METHODS[action]
        if request["path"]:
            return {"status": 404, "body": {"error": "Unknown worker lifecycle route"}}
        if request["method"] != method:
            return {"status": 405, "body": {"error": f"Use {method}"}}
        body = request["body"]
        if body is None:
            body = {}
        if not isinstance(body, dict) or body:
            return {"status": 400, "body": {"error": "Expected an empty JSON object"}}
        if action == "drain" and handler.config.dynamo_args.response_plane != "tcp":
            return {
                "status": 409,
                "body": {
                    "error": "manual worker drain requires the TCP response plane"
                },
            }
        try:
            if not endpoint_started and ready_event.is_set():
                async with discovery_lock:
                    if discovery_client is None:
                        discovery_client = await handler.generate_endpoint.client()
                    endpoint_started = endpoint_started or (
                        handler.generate_endpoint.connection_id()
                        in discovery_client.instance_ids()
                    )
            if action != "status" and not endpoint_started:
                if ready_event.is_set() and handler.engine.tokenizer_manager.is_pause:
                    raise DrainConflict("worker generation is paused")
                raise DrainConflict("worker endpoint is not ready")
            response = await getattr(controller, action)(body)
        except DrainConflict as error:
            return {"status": 409, "body": {"error": str(error)}}
        except Exception as error:
            logging.exception("Worker lifecycle operation failed")
            return {"status": 503, "body": {"error": str(error)}}
        state = response["state"]
        reason = response["reason"]
        if state == "serving" and handler.engine.tokenizer_manager.is_pause:
            state = "paused"
        elif not endpoint_started and state != "stopping":
            state = "starting"
        elif state in ("withdrawing", "withdrawn"):
            state = "draining"
        status_code = 202 if action == "drain" and state == "draining" else 200
        return {
            "status": status_code,
            "body": {
                "worker_id": str(handler.generate_endpoint.connection_id()),
                "state": state,
                "reason": reason,
            },
        }

    for action in _LIFECYCLE_METHODS:
        runtime.register_engine_http_handler(action, partial(control, action))
    handler._drain_controls_enabled = True
    return controller


def unregister_worker_drain_controls(handler, runtime) -> None:
    if getattr(handler, "_drain_controls_enabled", False):
        for action in _LIFECYCLE_METHODS:
            runtime.unregister_engine_http_handler(action)
        handler._drain_controls_enabled = False
