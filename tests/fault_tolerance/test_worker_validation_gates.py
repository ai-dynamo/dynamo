# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for worker validation gates on CPU mocker workers.

Each test runs a real frontend and mocker workers under etcd discovery, with
the reference check runner serving the publish gate's checks. A held worker
must get no client traffic and report not ready, while /live stays up; passing
checks release it; and a restarted worker is checked again.
"""

from __future__ import annotations

import logging
import os
import re
import sys
import threading
import time
from typing import Any, Callable, Iterator, Optional

import pytest
import requests

from dynamo.validation import runner
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

logger = logging.getLogger(__name__)

MODEL = QWEN
READY_BUDGET_S = 120

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.model(MODEL),
    pytest.mark.timeout(600),
]


class GatedMockerWorker(ManagedProcess):
    """A mocker worker whose startup check is ours: it may never be ready."""

    def __init__(
        self,
        request: Any,
        *,
        name: str,
        system_port: int,
        publish_gate: Optional[list[str]],
    ):
        env = os.environ.copy()
        env["DYN_SYSTEM_PORT"] = str(system_port)
        # Engine health comes from the generate endpoint, as in the shared
        # mocker fixture; validation adds to /health on top of it.
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_LOG"] = "info"
        env.pop("DYN_PUBLISH_GATE", None)
        env.pop("DYN_READY_GATE", None)
        if publish_gate:
            env["DYN_PUBLISH_GATE"] = ",".join(publish_gate)
        super().__init__(
            command=[
                sys.executable,
                "-m",
                "dynamo.mocker",
                "--model-path",
                MODEL,
                "--speedup-ratio",
                "100",
            ],
            env=env,
            # /live only: a held worker never passes /health.
            health_check_urls=[(f"http://localhost:{system_port}/live", _is_200)],
            timeout=READY_BUDGET_S,
            display_output=True,
            terminate_all_matching_process_names=False,
            straggler_commands=["-m dynamo.mocker"],
            log_dir=f"{request.node.name}_{name}",
        )


@pytest.fixture
def check_runner() -> Iterator[Callable[[dict[str, list[str]]], Callable[[str], str]]]:
    """Start the reference check runner on loopback; returns each check's URL."""
    servers = []

    def start(checks: dict[str, list[str]]) -> Callable[[str], str]:
        server = runner.make_server(0, checks, timeout=120)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        port = server.server_address[1]
        return lambda name: f"http://127.0.0.1:{port}/checks/{name}"

    yield start
    for server in servers:
        server.shutdown()


def _is_200(response: requests.Response) -> bool:
    return response.status_code == 200


def _health(system_port: int) -> tuple[int, dict[str, Any]]:
    response = requests.get(f"http://localhost:{system_port}/health", timeout=5)
    return response.status_code, response.json()


def _wait_for(
    predicate: Callable[[], bool], what: str, budget_s: float = READY_BUDGET_S
) -> None:
    deadline = time.monotonic() + budget_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return
        except (requests.RequestException, ValueError):
            pass  # Not up yet, or a half-written response; poll again.
        time.sleep(0.5)
    raise AssertionError(f"timed out waiting for {what}")


def _check(system_port: int, url: str) -> dict[str, Any]:
    _, body = _health(system_port)
    return body.get("validation", {}).get("publish", {}).get(url, {})


def _model_listed(frontend_port: int) -> bool:
    response = requests.get(f"http://localhost:{frontend_port}/v1/models", timeout=5)
    return any(model["id"] == MODEL for model in response.json().get("data", []))


def _chat(frontend_port: int) -> requests.Response:
    return requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
        },
        timeout=30,
    )


def _generate_requests(system_port: int) -> float:
    """Requests this worker's generate endpoint has served, from its metrics."""
    text = requests.get(f"http://localhost:{system_port}/metrics", timeout=5).text
    total = 0.0
    for line in text.splitlines():
        if line.startswith("dynamo_component_requests_total") and re.search(
            r'dynamo_endpoint="generate"', line
        ):
            total += float(line.rsplit(" ", 1)[1])
    return total


@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_failing_gate_holds_the_worker_and_old_worker_keeps_serving(
    request: Any,
    runtime_services_dynamic_ports,  # noqa: ARG001
    dynamo_dynamic_ports,
    num_system_ports,  # noqa: ARG001
    predownload_tokenizers,  # noqa: ARG001
    check_runner,
) -> None:
    frontend_port = dynamo_dynamic_ports.frontend_port
    old_port, new_port = (int(p) for p in dynamo_dynamic_ports.system_ports)

    with DynamoFrontendProcess(request, frontend_port=frontend_port):
        with GatedMockerWorker(
            request, name="old", system_port=old_port, publish_gate=None
        ):
            _wait_for(lambda: _model_listed(frontend_port), "the old worker's model")

            url = check_runner(
                {"smoke": ["sh", "-c", "echo tool parser mismatch >&2; exit 1"]}
            )
            smoke = url("smoke")
            with GatedMockerWorker(
                request, name="new", system_port=new_port, publish_gate=[smoke]
            ):
                _wait_for(
                    lambda: _check(new_port, smoke).get("state") == "failed",
                    "the failing check's verdict",
                )
                status, body = _health(new_port)
                assert status == 503, body
                assert "tool parser mismatch" in _check(new_port, smoke)["reason"], body
                live = requests.get(f"http://localhost:{new_port}/live", timeout=5)
                assert live.status_code == 200

                for _ in range(20):
                    assert _chat(frontend_port).status_code == 200
                assert _generate_requests(new_port) == 0
                assert _generate_requests(old_port) >= 20


@pytest.mark.parametrize("num_system_ports", [1], indirect=True)
def test_passing_gates_release_the_worker(
    request: Any,
    runtime_services_dynamic_ports,  # noqa: ARG001
    dynamo_dynamic_ports,
    num_system_ports,  # noqa: ARG001
    predownload_tokenizers,  # noqa: ARG001
    check_runner,
    tmp_path,
) -> None:
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = int(dynamo_dynamic_ports.system_ports[0])
    flag = tmp_path / "release"

    url = check_runner(
        {
            # Goes through the worker's loopback OpenAI endpoint.
            "completion": [sys.executable, "-m", "dynamo.validation.completion"],
            # Holds until the test creates the flag file.
            "manual": ["sh", "-c", f"while [ ! -f {flag} ]; do sleep 0.2; done"],
        }
    )
    publish_gate = [url("completion"), url("manual")]

    with DynamoFrontendProcess(request, frontend_port=frontend_port):
        with GatedMockerWorker(
            request, name="gated", system_port=system_port, publish_gate=publish_gate
        ):
            _wait_for(
                lambda: _check(system_port, url("completion")).get("state") == "passed",
                "the completion check to pass through the loopback endpoint",
            )
            # Still held by the manual check: not ready and not routable. The
            # frontend has no card for the model, so it reports it unknown.
            status, body = _health(system_port)
            assert status == 503, body
            assert not _model_listed(frontend_port)
            assert _chat(frontend_port).status_code == 404
            # The completion check's own request went through the loopback
            # endpoint, not the frontend; the client request must not have.
            served_while_held = _generate_requests(system_port)

            flag.touch()
            _wait_for(lambda: _health(system_port)[0] == 200, "release")
            _wait_for(lambda: _model_listed(frontend_port), "the released card")
            assert _chat(frontend_port).status_code == 200
            assert _generate_requests(system_port) == served_while_held + 1


@pytest.mark.parametrize("num_system_ports", [1], indirect=True)
def test_a_restarted_worker_is_checked_again(
    request: Any,
    runtime_services_dynamic_ports,  # noqa: ARG001
    dynamo_dynamic_ports,
    num_system_ports,  # noqa: ARG001
    predownload_tokenizers,  # noqa: ARG001
    check_runner,
    tmp_path,
) -> None:
    system_port = int(dynamo_dynamic_ports.system_ports[0])
    runs = tmp_path / "runs"
    url = check_runner({"count": ["sh", "-c", f"echo run >> {runs}"]})

    for attempt in (1, 2):
        with GatedMockerWorker(
            request,
            name=f"worker-{attempt}",
            system_port=system_port,
            publish_gate=[url("count")],
        ):
            _wait_for(lambda: _health(system_port)[0] == 200, "release")
            assert runs.read_text().splitlines() == ["run"] * attempt
