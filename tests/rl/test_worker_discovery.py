# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke coverage for Dynamo RL worker discovery and direct admin routes."""

from __future__ import annotations

import os
import time
from typing import Any, Generator
from urllib.parse import urlparse, urlunparse

import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import ServicePorts, allocate_port, deallocate_port

TEST_MODEL = QWEN

pytestmark = [
    # Heavy e2e (model startup + admin flows, timeout 900s): keep out of pre-merge gating.
    pytest.mark.post_merge,
    pytest.mark.e2e,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_1,
    pytest.mark.model(TEST_MODEL),
    # VRAM/KV budget so CI schedulers can place this model-loading test.
    # Mirrors the vLLM Qwen3-0.6B e2e values in tests/router/test_router_e2e_with_vllm.py.
    pytest.mark.profiled_vram_gib(6.9),
    pytest.mark.requested_vllm_kv_cache_bytes(331_801_000),
]


def _check_ready(response: requests.Response) -> bool:
    try:
        return (response.json() or {}).get("status") == "ready"
    except ValueError:
        return False


def _check_model_registered(response: requests.Response) -> bool:
    if not check_models_api(response):
        return False
    data = response.json()
    return any(model.get("id") == TEST_MODEL for model in data.get("data", []))


def _status_is_ok(payload: dict[str, Any]) -> bool:
    return payload.get("status") in {"ok", "success"}


def _post_json(
    url: str, payload: dict[str, Any], *, timeout: int = 60
) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    assert response.status_code == 200, response.text
    data = response.json()
    assert isinstance(data, dict), data
    return data


def _http_status(url: str) -> int:
    try:
        return requests.get(url, timeout=10).status_code
    except requests.RequestException:
        return 0


def _normalize_local_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.hostname not in {"0.0.0.0", "::"}:
        return url.rstrip("/")

    netloc = "localhost"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urlunparse(parsed._replace(netloc=netloc)).rstrip("/")


def _process_env(**extra: str) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env["DYN_LOG"] = "debug"
    env["DYN_NAMESPACE"] = "dynamo"
    env.update(extra)
    return env


def _prepare_log_dir(request: pytest.FixtureRequest, suffix: str) -> str:
    # Use pytest's per-test temp dir instead of a repo-relative path so process logs
    # never land in the repo tree and never collide across parallel runs.
    tmp_path = request.getfixturevalue("tmp_path")
    log_dir = tmp_path / suffix
    log_dir.mkdir(parents=True, exist_ok=True)
    return str(log_dir)


def _vllm_gpu_mem_args(default_utilization: str) -> list[str]:
    # Honor the GPU scheduler's per-worker KV-cache budget under bin-packing;
    # fall back to a conservative utilization for serial runs.
    return build_gpu_mem_args("build_vllm_gpu_mem_args") or [
        "--gpu-memory-utilization",
        default_utilization,
    ]


class RLVllmWorkerProcess(ManagedProcess):
    def __init__(
        self,
        request: pytest.FixtureRequest,
        *,
        frontend_port: int,
        system_port: int,
    ):
        env = _process_env(
            DYN_ENABLE_RL="true",
            DYN_SYSTEM_PORT=str(system_port),
            DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS='["generate"]',
        )

        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                TEST_MODEL,
                "--enforce-eager",
                *_vllm_gpu_mem_args("0.4"),
                "--max-model-len",
                "2048",
                "--max-num-seqs",
                "2",
                "--enable-rl",
                "--worker-extension-cls",
                "tests.rl.weight_update_worker.DynamoRLTestWorkerExtension",
            ],
            env=env,
            health_check_urls=[
                (f"http://localhost:{system_port}/health", _check_ready),
                (
                    f"http://localhost:{frontend_port}/v1/models",
                    _check_model_registered,
                ),
            ],
            timeout=600,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m dynamo.vllm"],
            log_dir=_prepare_log_dir(request, "rl-vllm-worker"),
            display_name="rl-vllm-worker",
        )


@pytest.fixture(scope="function")
def rl_discovery_port() -> Generator[int, None, None]:
    port = allocate_port(8001)
    try:
        yield port
    finally:
        deallocate_port(port)


@pytest.fixture(scope="function")
def start_rl_services(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports: object,
    dynamo_dynamic_ports: ServicePorts,
    rl_discovery_port: int,
    predownload_models: object,
) -> Generator[tuple[int, int, int], None, None]:
    _ = runtime_services_dynamic_ports, predownload_models
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env=_process_env(
            DYN_ENABLE_RL="true",
            DYN_RL_PORT=str(rl_discovery_port),
        ),
        terminate_all_matching_process_names=False,
        display_name="rl-frontend",
    ):
        with RLVllmWorkerProcess(
            request,
            frontend_port=frontend_port,
            system_port=system_port,
        ):
            yield frontend_port, rl_discovery_port, system_port


def _wait_for_discovered_worker(
    *, rl_port: int, timeout_s: int = 180
) -> tuple[dict[str, Any], str]:
    deadline = time.monotonic() + timeout_s
    last_response = ""
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"http://localhost:{rl_port}/v1/rl/workers",
                timeout=10,
            )
            last_response = response.text
            if response.status_code == 200:
                data = response.json()
                for worker in data.get("workers", []):
                    system_url = worker.get("system_url")
                    if system_url:
                        return worker, _normalize_local_url(system_url)
        except requests.RequestException as exc:
            last_response = str(exc)
        time.sleep(3)

    raise AssertionError(
        "Timed out waiting for an RL worker with system_url from "
        f"/v1/rl/workers. Last response: {last_response}"
    )


@pytest.mark.timeout(900)
def test_rl_worker_discovery_and_engine_admin_routes(
    start_rl_services, tmp_path
) -> None:
    frontend_port, rl_port, _system_port = start_rl_services
    worker, admin_url = _wait_for_discovered_worker(rl_port=rl_port)

    assert worker["endpoint"] == "rl"
    assert worker["request_plane_url"] == "dyn://dynamo.backend.rl"
    assert worker["system_url"]
    assert worker.get("model") == TEST_MODEL

    routes = set(worker.get("routes", []))
    assert {
        "liveness_probe",
        "pause_generation",
        "resume_generation",
        "get_weight_version",
        "update_weights_from_disk",
        "update_weights_from_distributed",
        "init_weights_update_group",
        "destroy_weights_update_group",
    }.issubset(routes)
    assert "load_lora" not in routes
    assert "unload_lora" not in routes

    assert _http_status(f"http://localhost:{rl_port}/v1/rl/engine") in {404, 405}
    assert _http_status(f"http://localhost:{rl_port}/v1/rl/engines") in {404, 405}
    assert _http_status(f"http://localhost:{frontend_port}/v1/rl/workers") in {
        404,
        405,
    }

    liveness = _post_json(f"{admin_url}/engine/liveness_probe", {})
    assert _status_is_ok(liveness)

    pause = _post_json(
        f"{admin_url}/engine/pause_generation",
        {"mode": "keep", "clear_cache": False},
    )
    assert _status_is_ok(pause)

    update = _post_json(
        f"{admin_url}/engine/update_weights_from_disk",
        {
            "model_path": str(tmp_path),
            "weight_version": "smoke_v1",
            "engine_rpc": "update_weights_from_path",
        },
        timeout=300,
    )
    assert _status_is_ok(update)

    version = _post_json(f"{admin_url}/engine/get_weight_version", {})
    assert version.get("version", version.get("weight_version")) == "smoke_v1"

    resume = _post_json(f"{admin_url}/engine/resume_generation", {})
    assert _status_is_ok(resume)

    inference = requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Say: hello"}],
            "max_tokens": 8,
            "stream": False,
            "nvext": {"extra_fields": ["engine_data"]},
        },
        timeout=60,
    )
    assert inference.status_code == 200, inference.text
    assert inference.json().get("choices")
