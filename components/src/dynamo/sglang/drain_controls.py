# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental lifecycle controls on the worker system listener."""

import asyncio
import logging

from dynamo.common.utils.env import env_bool
from dynamo.common.utils.graceful_shutdown import get_grace_period_seconds
from dynamo.sglang.drain import DrainConflict, WorkerDrain
from dynamo.sglang.engine_routes import parse_engine_route_descriptors


def configure_worker_drain(
    handler, runtime, ready_event, shutdown_event, terminal_event
):
    """Enable explicit withdrawal only for the Python LLM serving path."""
    if not env_bool("DYN_EXPERIMENTAL_WORKER_WITHDRAWAL"):
        return None
    if handler.config.dynamo_args.response_plane == "quic":
        raise ValueError(
            "worker withdrawal does not support the QUIC response plane: "
            "discovery removal cancels active responses"
        )
    if runtime.system_status_server_url() is None:
        raise ValueError("worker drain requires DYN_SYSTEM_PORT")
    for descriptor in parse_engine_route_descriptors(
        handler.config.dynamo_args.engine_routes
    ):
        if descriptor.path.split("/", 1)[0] == "worker_lifecycle":
            raise ValueError("worker_lifecycle is reserved by worker drain controls")

    controller = WorkerDrain(
        handler.generate_endpoint,
        lock=handler._engine_route_lock,
        shutdown_event=shutdown_event,
        terminal_event=terminal_event,
        ready_event=ready_event,
        is_paused=lambda: bool(handler.engine.tokenizer_manager.is_pause),
        grace_period=get_grace_period_seconds(),
    )

    endpoint_started = False
    discovery_client = None
    discovery_lock = asyncio.Lock()

    async def control(request):
        nonlocal endpoint_started, discovery_client
        action = request["path"].strip("/")
        method = "GET" if action == "status" else "POST"
        if action not in ("drain", "resume", "status"):
            return {"status": 404, "body": {"error": "Unknown worker lifecycle route"}}
        if request["method"] != method:
            return {"status": 405, "body": {"error": f"Use {method}"}}
        body = request["body"]
        if body is None:
            body = {}
        if not isinstance(body, dict) or body:
            return {"status": 400, "body": {"error": "Expected an empty JSON object"}}
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
                raise DrainConflict("worker endpoint is not ready")
            response = await getattr(controller, action)(body)
        except DrainConflict as error:
            return {"status": 409, "body": {"error": str(error)}}
        except Exception as error:
            logging.exception("Worker lifecycle operation failed")
            return {"status": 503, "body": {"error": str(error)}}
        if not endpoint_started and response["state"] != "stopping":
            response["state"] = "starting"
        response["engine_paused"] = bool(handler.engine.tokenizer_manager.is_pause)
        return {"status": 200, "body": response}

    runtime.register_engine_http_handler("worker_lifecycle", control)
    handler._worker_drain = controller
    return controller
