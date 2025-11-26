# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common base classes and utilities for engine tests (vLLM, TRT-LLM, etc.)"""

import logging
import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, Optional

import pytest

from dynamo.common.utils.paths import WORKSPACE_DIR
from tests.utils.client import send_request
from tests.utils.engine_process import EngineConfig, EngineProcess
from tests.utils.payloads import MetricsPayload
from tests.utils.port_utils import allocate_free_port, free_ports

DEFAULT_TIMEOUT = 10

SERVE_TEST_DIR = os.path.join(WORKSPACE_DIR, "tests/serve")


def run_serve_deployment(
    config: EngineConfig,
    request: Any,
    extra_env: Optional[Dict[str, str]] = None,
    runtime_services: Optional[Any] = None,
) -> None:
    """Run a standard serve deployment test for any EngineConfig.

    - Launches the engine via EngineProcess.from_script
    - Builds a payload (with optional override/mutator)
    - Iterates configured endpoints and validates responses and logs

    Args:
        config: EngineConfig with test configuration
        request: pytest request object
        extra_env: Optional dict of additional environment variables
        runtime_services: Optional tuple of (nats_process, etcd_process) from runtime_services fixture.
                        If provided, NATS_SERVER and ETCD_ENDPOINTS will be set from their ports.
    """

    logger = logging.getLogger(request.node.name)
    logger.info("Starting %s test_deployment", config.name)

    assert (
        config.request_payloads is not None and len(config.request_payloads) > 0
    ), "request_payloads must be provided on EngineConfig"

    logger.info("Using model: %s", config.model)
    logger.info("Script: %s", config.script_name)

    # Allocate dynamic HTTP port for frontend (starting from 8100 to avoid conflicts)
    http_port = allocate_free_port(8100)
    logger.info(f"Allocated dynamic HTTP port: {http_port}")

    # Allocate dynamic system metrics ports (starting from 9100 to avoid conflicts)
    # For disaggregated deployments with multiple workers, allocate multiple ports
    # TODO: Find a better way to determine num_workers automatically by parsing
    # the launch script or adding metadata to config instead of pattern matching
    # Disaggregated scripts (disagg*.sh) launch 2 workers, so need 2 ports
    num_workers = (
        2 if config.script_name and config.script_name.startswith("disagg") else 1
    )
    system_ports = [allocate_free_port(9100 + i) for i in range(num_workers)]
    system_port = system_ports[0]
    logger.info(f"Allocated dynamic system metrics port(s): {system_ports}")

    # Merge environment variables - start with empty dict to avoid inheriting
    # potentially incorrect NATS_SERVER/ETCD_ENDPOINTS from os.environ
    env = {}

    # Copy extra_env if provided
    if extra_env:
        env.update(extra_env)

    env["DYN_HTTP_PORT"] = str(http_port)
    # Set DYN_SYSTEM_PORT1, DYN_SYSTEM_PORT2, ... for each worker
    for i, port in enumerate(system_ports, start=1):
        env[f"DYN_SYSTEM_PORT{i}"] = str(port)
    # Also set DYN_SYSTEM_PORT to the first port for aggregated scripts that expect it
    env["DYN_SYSTEM_PORT"] = str(system_ports[0])
    # Note: DYN_SYSTEM_PORT1/2/... are set here, but frontend main.py will unset them
    # to prevent the frontend's DRT from starting a system server
    # Only the backend workers should start the system metrics server

    # Set NATS_SERVER and ETCD_ENDPOINTS from runtime_services if provided
    # This MUST override any values from extra_env or os.environ
    if runtime_services is not None:
        nats_process, etcd_process = runtime_services
        env["NATS_SERVER"] = f"nats://localhost:{nats_process.port}"
        env["ETCD_ENDPOINTS"] = f"http://localhost:{etcd_process.port}"
        logger.info(
            f"Using runtime services - NATS: {nats_process.port}, Etcd: {etcd_process.port}, HTTP: {http_port}"
        )
    else:
        # If runtime_services not provided, log warning but don't fail
        # (some tests might use default ports)
        logger.warning(
            "runtime_services not provided - subprocess will use default or os.environ NATS/Etcd ports"
        )

    # Create a modified config with the allocated HTTP port for health checks
    # We need to update models_port so health checks use the correct port
    from dataclasses import replace

    config_with_port = replace(config, models_port=http_port)

    try:
        with EngineProcess.from_script(
            config_with_port, request, extra_env=env
        ) as server_process:
            for payload in config.request_payloads:
                logger.info(
                    "TESTING: Payload: %s (port: %s)",
                    payload.__class__.__name__,
                    payload.port,
                )

                payload_item = payload
                # inject model
                if hasattr(payload_item, "with_model"):
                    payload_item = payload_item.with_model(config.model)

                # Update payload port to use allocated HTTP port
                # MetricsPayload uses system metrics port, not HTTP frontend
                if isinstance(payload_item, MetricsPayload):
                    # Map default ports (8081, 8082, ...) to allocated system_ports
                    # Default base port for metrics is 8081
                    default_metrics_base = 8081
                    port_index = payload_item.port - default_metrics_base
                    new_port = (
                        system_ports[port_index]
                        if port_index < len(system_ports)
                        else system_port
                    )

                    if payload_item.port != new_port:
                        payload_item = deepcopy(payload_item)
                        old_port = payload_item.port
                        payload_item.port = new_port
                        logger.info(
                            f"Updated MetricsPayload port from {old_port} to allocated system metrics port: {new_port}"
                        )
                elif payload_item.port != http_port:
                    payload_item = deepcopy(payload_item)
                    payload_item.port = http_port
                    logger.info(
                        f"Updated payload port from {payload.port} to allocated HTTP port: {http_port}"
                    )

                for _ in range(payload_item.repeat_count):
                    response = send_request(
                        url=payload_item.url(),
                        payload=payload_item.body,
                        timeout=payload_item.timeout,
                        method=payload_item.method,
                    )
                    server_process.check_response(payload_item, response)
    finally:
        # Release allocated ports
        free_ports([http_port] + system_ports)
        logger.info(
            f"Released HTTP port: {http_port}, system metrics port(s): {system_ports}"
        )


def params_with_model_mark(configs: Mapping[str, EngineConfig]):
    """Return pytest params for a config dict, adding a model marker per param.

    This enables simple model collection after pytest filtering.
    """
    params = []
    for config_name, cfg in configs.items():
        marks = list(getattr(cfg, "marks", []))
        marks.append(pytest.mark.model(cfg.model))
        params.append(pytest.param(config_name, marks=marks))
    return params
