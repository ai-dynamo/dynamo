# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only end-to-end coverage for reported-load text KV routing."""

from __future__ import annotations

import contextlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

import pytest
import requests

from tests.router.router_process import FrontendRouterProcess
from tests.utils.managed_process import ManagedProcess
from tests.utils.port_utils import allocate_ports, deallocate_ports

MODEL_PATH = "Qwen/Qwen3-0.6B"
MODEL_NAME = "text-kv-e2e"
FAKE_ENGINE = Path(__file__).with_name("fake_openai_text_engine.py")

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.router,
    pytest.mark.model(MODEL_PATH),
    pytest.mark.timeout(90),
]


def _fake_engine_process(
    request: pytest.FixtureRequest,
    *,
    label: str,
    port: int,
    loads: str,
) -> ManagedProcess:
    return ManagedProcess(
        command=[
            sys.executable,
            str(FAKE_ENGINE),
            "--port",
            str(port),
            "--label",
            label,
            "--loads",
            loads,
        ],
        health_check_urls=[f"http://127.0.0.1:{port}/health"],
        timeout=15,
        display_output=True,
        terminate_all_matching_process_names=False,
        display_name=f"fake-openai-{label}",
        log_dir=request.node.name,
    )


def _worker_process(
    request: pytest.FixtureRequest,
    *,
    label: str,
    namespace: str,
    engine_port: int,
    system_port: int,
) -> ManagedProcess:
    env = os.environ.copy()
    env.update(
        {
            "DYN_NAMESPACE": namespace,
            "DYN_REQUEST_PLANE": "nats",
            "DYN_SYSTEM_PORT": str(system_port),
            "DYN_OPENAI_BACKEND_LOAD_REPORT_INTERVAL_SECS": "0.05",
        }
    )
    return ManagedProcess(
        command=[
            sys.executable,
            "-m",
            "dynamo.openai_backend._worker",
            "--model",
            MODEL_PATH,
            "--served-model-name",
            MODEL_NAME,
            "--upstream-base-url",
            f"http://127.0.0.1:{engine_port}/v1",
            "--upstream-health-path",
            "/health",
            "--abort-base-url",
            f"http://127.0.0.1:{engine_port}",
        ],
        env=env,
        health_check_ports=[system_port],
        timeout=30,
        display_output=True,
        terminate_all_matching_process_names=False,
        display_name=f"text-worker-{label}",
        log_dir=request.node.name,
    )


def _wait_for_model(frontend_port: int) -> None:
    deadline = time.monotonic() + 20
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                f"http://127.0.0.1:{frontend_port}/v1/models", timeout=1
            )
            model_ids = {
                model["id"]
                for model in response.json().get("data", [])
                if "id" in model
            }
            if MODEL_NAME in model_ids:
                return
            last_error = f"available models: {sorted(model_ids)}"
        except (requests.RequestException, ValueError) as error:
            last_error = str(error)
        time.sleep(0.1)
    raise AssertionError(
        f"model {MODEL_NAME!r} did not appear on the frontend: {last_error}"
    )


def _completion(frontend_port: int, session_id: str) -> str:
    response = requests.post(
        f"http://127.0.0.1:{frontend_port}/v1/chat/completions",
        headers={"x-dynamo-session-id": session_id},
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "route this"}],
            "max_tokens": 1,
            "stream": True,
        },
        stream=True,
        timeout=10,
    )
    response.raise_for_status()

    content = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line.startswith("data: "):
            continue
        data = raw_line.removeprefix("data: ")
        if data == "[DONE]":
            break
        chunk = json.loads(data)
        for choice in chunk.get("choices", []):
            text = choice.get("delta", {}).get("content")
            if text:
                content.append(text)
    return "".join(content)


def _wait_for_stable_target(frontend_port: int, expected: str, prefix: str) -> str:
    deadline = time.monotonic() + 10
    observed = []
    consecutive = 0
    while time.monotonic() < deadline:
        session_id = f"{prefix}-{uuid.uuid4().hex}"
        actual = _completion(frontend_port, session_id)
        observed.append(actual)
        if actual == expected:
            consecutive += 1
            if consecutive == 3:
                return session_id
        else:
            consecutive = 0
        time.sleep(0.1)
    raise AssertionError(
        f"never selected {expected!r} three consecutive times; observed {observed!r}"
    )


def _set_loads(engine_port: int, loads: list[float]) -> None:
    response = requests.post(
        f"http://127.0.0.1:{engine_port}/admin/loads",
        json={"loads": loads},
        timeout=2,
    )
    response.raise_for_status()


def test_text_kv_routes_new_sessions_by_rank_and_reuses_affinity(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8000)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-a", port=engine_a_port, loads="0.50,0.50"
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-b", port=engine_b_port, loads="0.50,0.50"
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="a",
                namespace=namespace,
                engine_port=engine_a_port,
                system_port=system_a_port,
            )
        )
        stack.enter_context(
            _worker_process(
                request,
                label="b",
                namespace=namespace,
                engine_port=engine_b_port,
                system_port=system_b_port,
            )
        )
        stack.enter_context(
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="kv",
                min_initial_workers=2,
                request_plane="nats",
            )
        )
        _wait_for_model(frontend_port)

        # Change the synthetic engine gauges after the frontend subscribes so
        # the non-durable metrics event plane emits fresh reports.
        _set_loads(engine_a_port, [0.80, 0.70])
        _set_loads(engine_b_port, [0.60, 0.10])
        sticky_session = _wait_for_stable_target(
            frontend_port, "worker-b:rank-1", "initial"
        )

        _set_loads(engine_a_port, [0.01, 0.90])
        _set_loads(engine_b_port, [0.80, 0.99])
        _wait_for_stable_target(frontend_port, "worker-a:rank-0", "changed")

        assert _completion(frontend_port, sticky_session) == "worker-b:rank-1"
