# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for ``python -m dynamo.fastvideo``."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Sequence

import uvloop

from dynamo.common.config_dump import dump_config
from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.common.utils.runtime import create_runtime
from dynamo.runtime.logging import configure_dynamo_logging

from .args import parse_fastvideo_args
from .backend import FastVideoHandler, register_fastvideo_model
from .health_check import FastVideoHealthCheckPayload

if "FASTVIDEO_LOG_LEVEL" not in os.environ:
    if dyn_log_level := os.environ.get("DYN_LOG"):
        os.environ["FASTVIDEO_LOG_LEVEL"] = dyn_log_level.upper()

configure_dynamo_logging()
logger = logging.getLogger(__name__)
shutdown_endpoints: list = []


async def run_fastvideo(argv: Sequence[str] | None = None) -> None:
    """Parse config, initialize the backend, and serve requests."""
    config = parse_fastvideo_args(argv)
    dump_config(config.dump_config_to, config)

    shutdown_event = asyncio.Event()
    runtime, loop = create_runtime(
        discovery_backend=config.discovery_backend,
        request_plane=config.request_plane,
        event_plane=config.event_plane,
        use_kv_events=False,
    )
    install_signal_handlers(loop, runtime, shutdown_endpoints, shutdown_event)

    endpoint = runtime.endpoint(
        f"{config.namespace}.{config.component}.{config.endpoint}"
    )
    shutdown_endpoints[:] = [endpoint]

    handler = FastVideoHandler(config)
    health_check_payload = FastVideoHealthCheckPayload(
        model_path=config.model_path
    ).to_dict()
    try:
        await handler.initialize()
        await register_fastvideo_model(endpoint, config)

        logger.info(
            "Serving FastVideo on %s.%s.%s for model %s",
            config.namespace,
            config.component,
            config.endpoint,
            config.served_model_name,
        )

        await endpoint.serve_endpoint(
            handler.generate,
            graceful_shutdown=True,
            health_check_payload=health_check_payload,
        )
    finally:
        handler.cleanup()


def main(argv: Sequence[str] | None = None) -> None:
    uvloop.run(run_fastvideo(argv))
