# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common base classes and utilities for engine tests (vLLM, TRT-LLM, etc.)"""

import dataclasses
import logging
import os
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Dict, Optional

import pytest

from dynamo.common.utils.paths import WORKSPACE_DIR
from tests.conftest import ServicePorts
from tests.utils.client import send_request
from tests.utils.constants import DefaultPort
from tests.utils.engine_process import EngineConfig, EngineProcess

DEFAULT_TIMEOUT = 10

SERVE_TEST_DIR = os.path.join(WORKSPACE_DIR, "tests/serve")


def run_serve_deployment(
    config: EngineConfig,
    request: Any,
    *,
    ports: ServicePorts | None = None,  # pass `dynamo_dynamic_ports` here
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

    merged_env: dict[str, str] = {}
    if extra_env:
        merged_env.update(extra_env)

    if ports is not None:
        dynamic_frontend_port = int(ports.frontend_port)
        dynamic_system_ports = [int(p) for p in ports.system_ports]

        # The environments are used by the bash scripts to set the ports.
        merged_env["DYN_HTTP_PORT"] = str(dynamic_frontend_port)

        # If no system ports are provided, explicitly ensure we don't pass any
        # stale DYN_SYSTEM_PORT* values via extra_env.
        if not dynamic_system_ports:
            for k in list(merged_env.keys()):
                if k == "DYN_SYSTEM_PORT":
                    merged_env.pop(k, None)
                    continue
                if k.startswith("DYN_SYSTEM_PORT") and k != "DYN_SYSTEM_PORT":
                    suffix = k.removeprefix("DYN_SYSTEM_PORT")
                    if suffix.isdigit():
                        merged_env.pop(k, None)
        else:
            # Alias for PORT1 (many scripts only read this).
            merged_env["DYN_SYSTEM_PORT"] = str(dynamic_system_ports[0])
            merged_env["DYN_SYSTEM_PORT1"] = str(dynamic_system_ports[0])
            for idx, port in enumerate(dynamic_system_ports, start=1):
                merged_env[f"DYN_SYSTEM_PORT{idx}"] = str(port)

        # Ensure EngineProcess health checks hit the correct frontend port.
        config = dataclasses.replace(config, frontend_port=dynamic_frontend_port)
    else:
        # Backward compat: infer from config/extra_env if no explicit ports are passed.
        dynamic_frontend_port = int(config.frontend_port)
        # Preserve the historical two-port behavior in this branch. Tests that
        # need tighter control should pass `ports=...` to avoid default port
        # collisions under xdist.
        dynamic_system_ports = [
            int(
                merged_env.get("DYN_SYSTEM_PORT1")
                or merged_env.get("DYN_SYSTEM_PORT")
                or DefaultPort.SYSTEM1.value
            ),
            int(merged_env.get("DYN_SYSTEM_PORT2") or DefaultPort.SYSTEM2.value),
        ]

    with EngineProcess.from_script(
        config, request, extra_env=merged_env
    ) as server_process:
        for _payload in config.request_payloads:
            logger.info("TESTING: Payload: %s", _payload.__class__.__name__)

            # Make a per-iteration copy so tests can safely override ports/fields
            # without mutating shared config instances across parametrized cases.
            payload = deepcopy(_payload)
            # inject model
            if hasattr(payload, "with_model"):
                payload = payload.with_model(config.model)

            # Default behavior: requests go to the frontend port, except metrics which target
            # worker system ports (mapped from DefaultPort -> per-test ports).
            if getattr(payload, "endpoint", "") == "/metrics":
                if payload.port == DefaultPort.SYSTEM1.value:
                    if len(dynamic_system_ports) < 1:
                        raise RuntimeError(
                            "Payload targets SYSTEM_PORT1 but no system ports were provided "
                            f"(payload={payload.__class__.__name__})"
                        )
                    payload.port = dynamic_system_ports[0]
                elif payload.port == DefaultPort.SYSTEM2.value:
                    if len(dynamic_system_ports) < 2:
                        raise RuntimeError(
                            "Payload targets SYSTEM_PORT2 but only 1 system port was provided "
                            f"(payload={payload.__class__.__name__})"
                        )
                    payload.port = dynamic_system_ports[1]
            else:
                payload.port = dynamic_frontend_port

            # Optional extra system ports for specialized payloads (e.g. LoRA control-plane APIs).
            # BasePayload always defines `system_ports` (usually empty); map defaults
            # (SYSTEM_PORT1/2) to per-test system ports when present.
            if payload.system_ports:
                mapped_system_ports: list[int] = []
                for p in payload.system_ports:
                    if p == DefaultPort.SYSTEM1.value:
                        if len(dynamic_system_ports) < 1:
                            raise RuntimeError(
                                "Payload.system_ports includes SYSTEM_PORT1 but no system ports were provided "
                                f"(payload={payload.__class__.__name__})"
                            )
                        mapped_system_ports.append(dynamic_system_ports[0])
                    elif p == DefaultPort.SYSTEM2.value:
                        if len(dynamic_system_ports) < 2:
                            raise RuntimeError(
                                "Payload.system_ports includes SYSTEM_PORT2 but only 1 system port was provided "
                                f"(payload={payload.__class__.__name__})"
                            )
                        mapped_system_ports.append(dynamic_system_ports[1])
                    else:
                        mapped_system_ports.append(p)
                payload.system_ports = mapped_system_ports

            for _ in range(payload.repeat_count):
                response = send_request(
                    url=payload.url(),
                    payload=payload.body,
                    timeout=payload.timeout,
                    method=payload.method,
                )
                server_process.check_response(payload, response)


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
