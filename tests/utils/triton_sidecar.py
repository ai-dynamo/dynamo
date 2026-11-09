# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared launchers for end-to-end `dynamo-triton-sidecar` tests."""

import os
import shutil
import uuid
from pathlib import Path

import pytest

from tests.utils.constants import DynamoPortRange
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

TRITON_MODEL_REPO = (
    Path(__file__).resolve().parents[2]
    / "components"
    / "src"
    / "dynamo"
    / "triton"
    / "models"
)

STARTUP_TIMEOUT_SECS = 180


def resolve_bin(env_var: str, default: str) -> str:
    """Prefer `env_var` for local overrides, else PATH, else skip the test."""
    override = os.environ.get(env_var)
    if override:
        return override
    found = shutil.which(default)
    if not found:
        pytest.skip(f"missing {default}; set {env_var} to its path")
    return found


class TritonServerProcess(ManagedProcess):
    """Real tritonserver subprocess loading the shipped Triton example repository."""

    def __init__(self, request, grpc_port: int, http_port: int, log_dir: str):
        command = [
            resolve_bin("TRITONSERVER_BIN", "tritonserver"),
            "--model-repository",
            str(TRITON_MODEL_REPO),
            "--grpc-port",
            str(grpc_port),
            "--http-port",
            str(http_port),
            "--allow-metrics",
            "false",
        ]
        super().__init__(
            command=command,
            health_check_urls=[
                (
                    f"http://localhost:{http_port}/v2/health/ready",
                    lambda r: r.status_code == 200,
                )
            ],
            timeout=STARTUP_TIMEOUT_SECS,
            display_output=True,
            log_dir=log_dir,
            terminate_all_matching_process_names=False,
        )


class TritonSidecarProcess(ManagedProcess):
    """Real `dynamo-triton-sidecar` subprocess registering one task-bound model."""

    def __init__(
        self,
        request,
        triton_grpc_port: int,
        system_port: int,
        namespace: str,
        task_flag: str,
        model: str,
        log_dir: str,
    ):
        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_PORT"] = str(system_port)
        command = [
            resolve_bin("DYNAMO_TRITON_SIDECAR", "dynamo-triton-sidecar"),
            "--namespace",
            namespace,
            "--grpc",
            f"localhost:{triton_grpc_port}",
            task_flag,
            model,
        ]
        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (
                    f"http://localhost:{system_port}/health",
                    lambda r: r.json().get("status") == "ready",
                )
            ],
            timeout=STARTUP_TIMEOUT_SECS,
            display_output=True,
            log_dir=log_dir,
            terminate_all_matching_process_names=False,
        )


def sidecar_stack(request, tmp_path, frontend_ports, task_flag, model):
    """Launch tritonserver + dynamo-triton-sidecar on top of a Dynamo frontend."""
    frontend_port, system_port = frontend_ports
    triton_ports = allocate_ports(2, DynamoPortRange.SERVE.value)
    triton_grpc_port, triton_http_port = triton_ports
    namespace = f"dynamo-{uuid.uuid4().hex[:8]}"
    try:
        with TritonServerProcess(
            request,
            grpc_port=triton_grpc_port,
            http_port=triton_http_port,
            log_dir=str(tmp_path / "tritonserver"),
        ):
            with TritonSidecarProcess(
                request,
                triton_grpc_port=triton_grpc_port,
                system_port=system_port,
                namespace=namespace,
                task_flag=task_flag,
                model=model,
                log_dir=str(tmp_path / "sidecar"),
            ):
                yield frontend_port, model
    finally:
        deallocate_ports(triton_ports)
