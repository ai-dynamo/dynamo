# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Request-level SSRF tests for the two frontend media-fetch paths.

``vllm_processor`` fetches through ``DynamoMediaConnector``.
``rust_decoding`` fetches through the Rust ``MediaLoader``. Both must
reject a loopback URL before opening a connection.
"""

from __future__ import annotations

import os
import tempfile
import time
from typing import Any, Generator

import pytest
import requests

from tests.conftest import EtcdServer, NatsServer
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import ManagedProcess
from tests.utils.network_canary import ConnectionCanary, running_canary
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_ports

VLLM_MM_MODEL = os.getenv("DYN_TEST_VLLM_MM_MODEL", "Qwen/Qwen3-VL-2B-Instruct")
BLOCK_SIZE = 16
# Distinct namespace so concurrent mm_router suites never cross-register.
NAMESPACE = "frontend-ssrf"

pytestmark = [
    pytest.mark.post_merge,
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.gpu_1,
    pytest.mark.model(VLLM_MM_MODEL),
    pytest.mark.requested_vllm_kv_cache_bytes(1_719_075_000),
    pytest.mark.profiled_vram_gib(7.6),
]

_COMMON_PROCESS_KWARGS: dict[str, Any] = {
    "display_output": False,
    "terminate_all_matching_process_names": False,
}


def _check_ready(response) -> bool:
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


def _strict_media_env(**extra: str) -> dict[str, str]:
    """Build a child environment with internal media access disabled."""
    env = os.environ.copy()
    env.pop("DYN_MM_ALLOW_INTERNAL", None)
    env.pop("DYN_MM_LOCAL_PATH", None)
    env["DYN_LOG"] = "info"
    env["DYN_NAMESPACE"] = NAMESPACE
    env["DYN_REQUEST_PLANE"] = "tcp"
    env.update(extra)
    return env


def _log_dir(request, suffix: str) -> str:
    return tempfile.mkdtemp(prefix=f"{request.node.name}_{suffix}_")


class _VllmWorkerProcess(ManagedProcess):
    def __init__(self, request, *, topology: str, system_port: int) -> None:
        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            VLLM_MM_MODEL,
            "--enable-multimodal",
            "--block-size",
            str(BLOCK_SIZE),
            "--enforce-eager",
            *(
                build_gpu_mem_args("build_vllm_gpu_mem_args")
                or ["--gpu-memory-utilization", "0.40"]
            ),
            "--max-model-len",
            "4096",
        ]
        if topology == "rust_decoding":
            command.append("--frontend-decoding")

        super().__init__(
            command=command,
            env=_strict_media_env(DYN_SYSTEM_PORT=str(system_port)),
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready)
            ],
            timeout=900,
            straggler_commands=["-m dynamo.vllm"],
            log_dir=_log_dir(request, f"worker-{topology}"),
            **_COMMON_PROCESS_KWARGS,
        )


class _FrontendProcess(ManagedProcess):
    def __init__(self, request, *, topology: str, frontend_port: int) -> None:
        command = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(frontend_port),
            "--model-name",
            VLLM_MM_MODEL,
        ]
        if topology == "vllm_processor":
            command += ["--dyn-chat-processor", "vllm"]

        super().__init__(
            command=command,
            env=_strict_media_env(),
            health_check_urls=[
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            ],
            timeout=240,
            straggler_commands=["-m dynamo.frontend"],
            log_dir=_log_dir(request, f"frontend-{topology}"),
            **_COMMON_PROCESS_KWARGS,
        )


@pytest.fixture(scope="module")
def runtime_services(request):
    with (
        NatsServer(request, port=0) as nats,
        EtcdServer(request, port=0) as etcd,
        pytest.MonkeyPatch.context() as mp,
    ):
        mp.setenv("NATS_SERVER", f"nats://localhost:{nats.port}")
        mp.setenv("ETCD_ENDPOINTS", f"http://localhost:{etcd.port}")
        yield


@pytest.fixture(scope="module", params=["vllm_processor", "rust_decoding"])
def frontend_topology(
    request, runtime_services, predownload_models
) -> Generator[tuple[str, int], None, None]:
    topology = request.param
    frontend_port, system_port = allocate_ports(count=2, start_port=13000)
    with _VllmWorkerProcess(request, topology=topology, system_port=system_port):
        time.sleep(2)  # let discovery settle before the frontend registers
        with _FrontendProcess(request, topology=topology, frontend_port=frontend_port):
            yield topology, frontend_port


@pytest.fixture
def canary() -> Generator[ConnectionCanary, None, None]:
    with running_canary() as server:
        yield server


def _image_request(image_url: str) -> dict[str, Any]:
    return {
        "model": VLLM_MM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_tokens": 1,
    }


@pytest.mark.timeout(1200)
def test_blocked_url_is_refused_and_never_fetched(
    frontend_topology: tuple[str, int], canary: ConnectionCanary
) -> None:
    """A forbidden destination must yield 4xx without a TCP connection."""
    topology, frontend_port = frontend_topology

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json=_image_request(canary.blocked_url()),
        timeout=180,
    )

    assert 400 <= response.status_code < 500, (
        f"[{topology}] expected a 4xx for a blocked media URL, got "
        f"HTTP {response.status_code}: {response.text[:2000]}"
    )
    expected_detail = {
        "vllm_processor": "blocked range",
        "rust_decoding": "Direct IP access is not allowed",
    }[topology]
    assert expected_detail in response.text, response.text
    canary.assert_no_connection()
