# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-merge CPU integration test for the unified-backend canary pipeline."""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any

import pytest
import requests

from dynamo.common.utils.paths import WORKSPACE_DIR
from tests.utils.engine_process import EngineConfig, EngineProcess

CANARY_READY_BUDGET_S = 60
SAMPLE_DIR = os.path.join(WORKSPACE_DIR, "examples/backends/sample")
MODEL = "Qwen/Qwen3-0.6B"

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.integration,
    pytest.mark.unified,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.model(MODEL),
    pytest.mark.timeout(300),
]


def _wait_ready(url: str, deadline_s: float) -> None:
    deadline = time.monotonic() + deadline_s
    last_body: dict[str, Any] = {}
    last_status = 0
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=2.0)
            last_status = resp.status_code
            last_body = resp.json() if resp.content else {}
            if resp.status_code == 200 and last_body.get("status") == "ready":
                return
        except (requests.exceptions.RequestException, ValueError):
            pass
        time.sleep(0.5)
    raise AssertionError(
        f"{url} never reached ready within {deadline_s}s "
        f"(last_status={last_status} body={last_body})"
    )


def _sample_config(mode: str) -> EngineConfig:
    return EngineConfig(
        name=f"unified-canary-{mode}",
        directory=SAMPLE_DIR,
        script_name="disagg.sh" if mode == "disagg" else "agg.sh",
        script_args=["--model-name", MODEL],
        marks=[],
        model=MODEL,
        request_payloads=[],
        timeout=300,
        health_check_workers=(mode == "disagg"),
    )


@pytest.mark.parametrize("mode", ["agg", "disagg"])
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_unified_canary_marks_endpoints_ready(
    mode: str,
    request: Any,
    runtime_services_dynamic_ports,  # noqa: ARG001
    dynamo_dynamic_ports,
    num_system_ports,  # noqa: ARG001
    predownload_models,  # noqa: ARG001
) -> None:
    is_disagg = mode == "disagg"
    config = dataclasses.replace(
        _sample_config(mode), frontend_port=dynamo_dynamic_ports.frontend_port
    )

    system_ports = [int(p) for p in dynamo_dynamic_ports.system_ports]
    extra_env: dict[str, str] = {
        "DYN_HEALTH_CHECK_ENABLED": "true",
        "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": '["generate"]',
        "DYN_HTTP_PORT": str(dynamo_dynamic_ports.frontend_port),
        "DYN_SYSTEM_PORT": str(system_ports[0]),
    }
    if is_disagg:
        extra_env["DYN_SYSTEM_PORT1"] = str(system_ports[0])
        extra_env["DYN_SYSTEM_PORT2"] = str(system_ports[1])

    worker_ports = system_ports if is_disagg else system_ports[:1]
    with EngineProcess.from_config(config, request, extra_env=extra_env):
        for port in worker_ports:
            _wait_ready(f"http://localhost:{port}/health", CANARY_READY_BUDGET_S)


@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_unified_canary_honors_operator_override(
    request: Any,
    runtime_services_dynamic_ports,  # noqa: ARG001
    dynamo_dynamic_ports,
    num_system_ports,  # noqa: ARG001
    predownload_models,  # noqa: ARG001
) -> None:
    """A minimal `DYN_HEALTH_CHECK_PAYLOAD` overrides the engine default end-to-end."""
    config = dataclasses.replace(
        _sample_config("agg"), frontend_port=dynamo_dynamic_ports.frontend_port
    )
    system_port = int(dynamo_dynamic_ports.system_ports[0])
    extra_env = {
        "DYN_HEALTH_CHECK_ENABLED": "true",
        "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS": '["generate"]',
        "DYN_HEALTH_CHECK_PAYLOAD": '{"token_ids": [1]}',
        "DYN_HTTP_PORT": str(dynamo_dynamic_ports.frontend_port),
        "DYN_SYSTEM_PORT": str(system_port),
    }

    with EngineProcess.from_config(config, request, extra_env=extra_env):
        _wait_ready(f"http://localhost:{system_port}/health", CANARY_READY_BUDGET_S)
