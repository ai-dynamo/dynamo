# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common base classes and utilities for engine tests (vLLM, TRT-LLM, etc.)"""

import logging
from typing import Any, Dict, Optional

from tests.utils.client import send_request
from tests.utils.engine_process import EngineConfig, EngineProcess

DEFAULT_TIMEOUT = 10


def run_serve_deployment(
    config: EngineConfig,
    request: Any,
    extra_env: Optional[Dict[str, str]] = None,
) -> None:
    """Run a standard serve deployment test for any EngineConfig.

    - Launches the engine via EngineProcess.from_script
    - Builds a payload (with optional override/mutator)
    - Iterates configured endpoints and validates responses and logs
    """

    logger = logging.getLogger(request.node.name)
    logger.info("Starting %s test_deployment", config.name)

    assert (
        config.request_payloads is not None and len(config.request_payloads) > 0
    ), "request_payloads must be provided on EngineConfig"

    logger.info("Using model: %s", config.model)
    logger.info("Script: %s", config.script_name)

    with EngineProcess.from_script(
        config, request, extra_env=extra_env
    ) as server_process:
        for payload in config.request_payloads:
            logger.info("TESTING: Payload: %s", payload.__class__.__name__)

            payload_item = payload
            # inject model
            if hasattr(payload_item, "with_model"):
                payload_item = payload_item.with_model(config.model)

            if payload_item.port != config.models_port:
                logger.warning(
                    f"Current payload port: {payload_item.port} doesn't match the model port: {config.models_port}"
                )

            for _ in range(payload_item.repeat_count):
                response = send_request(
                    url=payload_item.url(),
                    payload=payload_item.body,
                    timeout=payload_item.timeout,
                    method=payload_item.method,
                )
                server_process.check_response(payload_item, response)
