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
from concurrent.futures import ThreadPoolExecutor
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


def _wait_for_session_target(
    frontend_port: int, session_id: str, expected: str
) -> None:
    deadline = time.monotonic() + 15
    observed: list[str] = []
    errors: list[str] = []
    while time.monotonic() < deadline:
        try:
            actual = _completion(frontend_port, session_id)
        except requests.RequestException as error:
            errors.append(str(error))
            time.sleep(0.1)
            continue
        observed.append(actual)
        if actual == expected:
            return
        time.sleep(0.1)
    raise AssertionError(
        f"session {session_id!r} never selected {expected!r}; "
        f"observed={observed!r}, errors={errors!r}"
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

        concurrent_session = f"concurrent-{uuid.uuid4().hex}"
        with ThreadPoolExecutor(max_workers=8) as executor:
            concurrent_targets = list(
                executor.map(
                    lambda _: _completion(frontend_port, concurrent_session),
                    range(8),
                )
            )
        assert set(concurrent_targets) == {"worker-b:rank-1"}

        _set_loads(engine_a_port, [0.01, 0.90])
        _set_loads(engine_b_port, [0.80, 0.99])
        _wait_for_stable_target(frontend_port, "worker-a:rank-0", "changed")

        assert _completion(frontend_port, sticky_session) == "worker-b:rank-1"


def test_text_kv_rebalances_existing_sessions_when_worker_scales_up(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8100)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-scale-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-a", port=engine_a_port, loads="0.20,0.30"
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-b", port=engine_b_port, loads="0.10,0.10"
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
            FrontendRouterProcess(
                request,
                block_size=16,
                frontend_port=frontend_port,
                namespace=namespace,
                router_mode="kv",
                min_initial_workers=1,
                request_plane="nats",
            )
        )
        _wait_for_model(frontend_port)

        # Give the model-scoped runtime-config watch time to establish worker A
        # as its startup baseline before creating affinity entries.
        time.sleep(0.5)
        session_ids = [f"scale-session-{uuid.uuid4().hex}" for _ in range(100)]
        initial_targets = {
            session_id: _completion(frontend_port, session_id)
            for session_id in session_ids
        }
        assert all(
            selected.startswith("worker-a:rank-")
            for selected in initial_targets.values()
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
        _set_loads(engine_a_port, [0.90, 0.90])
        _set_loads(engine_b_port, [0.10, 0.20])
        _wait_for_stable_target(frontend_port, "worker-b:rank-0", "worker-ready")
        time.sleep(0.5)

        after_scale = {
            session_id: _completion(frontend_port, session_id)
            for session_id in initial_targets
        }
        migrated = {
            session_id: selected
            for session_id, selected in after_scale.items()
            if selected.startswith("worker-b:rank-")
        }

        # Both workers advertise the same KV capacity, so the deterministic
        # cohort should be approximately half of the 100 existing sessions.
        assert 35 <= len(migrated) <= 65
        for session_id, selected in after_scale.items():
            if session_id not in migrated:
                assert selected == initial_targets[session_id]

        # The scale event is consumed once per key. Both migrated and retained
        # sessions now keep their exact worker/rank binding.
        for session_id, selected in after_scale.items():
            assert _completion(frontend_port, session_id) == selected


def test_text_kv_routes_plain_workers_without_dp_rank(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8200)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-plain-{uuid.uuid4().hex[:12]}"

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-a", port=engine_a_port, loads="0.50"
            )
        )
        stack.enter_context(
            _fake_engine_process(
                request, label="worker-b", port=engine_b_port, loads="0.50"
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
        _set_loads(engine_a_port, [0.80])
        _set_loads(engine_b_port, [0.10])
        session_id = _wait_for_stable_target(
            frontend_port, "worker-b:rank-none", "plain"
        )

        _set_loads(engine_a_port, [0.01])
        _set_loads(engine_b_port, [0.99])
        _wait_for_stable_target(frontend_port, "worker-a:rank-none", "plain-new")
        assert _completion(frontend_port, session_id) == "worker-b:rank-none"


def test_text_kv_rebinds_only_sessions_on_a_removed_worker(
    request: pytest.FixtureRequest,
    runtime_services_dynamic_ports,
    predownload_tokenizers,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ = runtime_services_dynamic_ports, predownload_tokenizers
    monkeypatch.setenv("DYN_ROUTER_SESSION_AFFINITY_TTL_SECS", "60")
    ports = allocate_ports(5, 8300)
    request.addfinalizer(lambda: deallocate_ports(ports))
    frontend_port, engine_a_port, engine_b_port, system_a_port, system_b_port = ports
    namespace = f"text-kv-down-{uuid.uuid4().hex[:12]}"

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
        worker_b_stack = contextlib.ExitStack()
        stack.callback(worker_b_stack.close)
        worker_b_stack.enter_context(
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

        _set_loads(engine_a_port, [0.80, 0.80])
        _set_loads(engine_b_port, [0.10, 0.20])
        removed_session = _wait_for_stable_target(
            frontend_port, "worker-b:rank-0", "removed"
        )

        _set_loads(engine_a_port, [0.10, 0.20])
        _set_loads(engine_b_port, [0.80, 0.80])
        surviving_session = _wait_for_stable_target(
            frontend_port, "worker-a:rank-0", "surviving"
        )

        # Make rank 1 the least-loaded surviving target before removing B.
        _set_loads(engine_a_port, [0.90, 0.01])
        worker_b_stack.close()

        # A request that races discovery may fail once. Retrying the same
        # request/session must invalidate B, select A/rank 1, and remain sticky.
        _wait_for_session_target(frontend_port, removed_session, "worker-a:rank-1")
        assert _completion(frontend_port, removed_session) == "worker-a:rank-1"
        assert _completion(frontend_port, removed_session) == "worker-a:rank-1"

        # Affinity already pointing at surviving A remains untouched.
        assert _completion(frontend_port, surviving_session) == "worker-a:rank-0"
