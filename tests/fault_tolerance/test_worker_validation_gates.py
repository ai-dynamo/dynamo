# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for worker validation gates on CPU mocker workers.

Each test runs a real frontend and mocker workers under etcd discovery. A held
worker must get no client traffic and report not ready, while /live stays up;
a passing gate releases it; and a restarted worker is checked again.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from typing import Any, Callable, Optional

import pytest
import requests

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
        gates: Optional[dict[str, Any]],
    ):
        env = os.environ.copy()
        env["DYN_SYSTEM_PORT"] = str(system_port)
        # Engine health comes from the generate endpoint, as in the shared
        # mocker fixture; validation adds to /health on top of it.
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_LOG"] = "info"
        env.pop("DYN_WORKER_VALIDATION_GATES", None)
        if gates is not None:
            env["DYN_WORKER_VALIDATION_GATES"] = json.dumps(gates)
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


def _gate_state(system_port: int, gate: str) -> Optional[str]:
    _, body = _health(system_port)
    return body.get("validation", {}).get("gates", {}).get(gate, {}).get("state")


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
) -> None:
    frontend_port = dynamo_dynamic_ports.frontend_port
    old_port, new_port = (int(p) for p in dynamo_dynamic_ports.system_ports)

    with DynamoFrontendProcess(request, frontend_port=frontend_port):
        with GatedMockerWorker(request, name="old", system_port=old_port, gates=None):
            _wait_for(lambda: _model_listed(frontend_port), "the old worker's model")

            failing = {
                "smoke": {
                    "exec": ["sh", "-c", "echo tool parser mismatch >&2; exit 1"],
                    "timeout_secs": 60,
                }
            }
            with GatedMockerWorker(
                request, name="new", system_port=new_port, gates=failing
            ):
                _wait_for(
                    lambda: _gate_state(new_port, "smoke") == "failed",
                    "the failing gate's verdict",
                )
                status, body = _health(new_port)
                assert status == 503, body
                assert (
                    "tool parser mismatch"
                    in body["validation"]["gates"]["smoke"]["reason"]
                ), body
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
    tmp_path,
) -> None:
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = int(dynamo_dynamic_ports.system_ports[0])
    flag = tmp_path / "release"

    gates = {
        # Goes through the worker's loopback OpenAI endpoint.
        "completion": {
            "exec": [sys.executable, "-m", "dynamo.validation.completion"],
            "timeout_secs": 120,
        },
        # Holds until the test creates the flag file.
        "manual": {
            "exec": ["sh", "-c", f"while [ ! -f {flag} ]; do sleep 0.2; done"],
            "timeout_secs": 120,
        },
    }

    with DynamoFrontendProcess(request, frontend_port=frontend_port):
        with GatedMockerWorker(
            request, name="gated", system_port=system_port, gates=gates
        ):
            _wait_for(
                lambda: _gate_state(system_port, "completion") == "passed",
                "the completion gate to pass through the loopback endpoint",
            )
            # Still held by the manual gate: not ready and not routable. The
            # frontend has no card for the model, so it reports it unknown.
            status, body = _health(system_port)
            assert status == 503, body
            assert not _model_listed(frontend_port)
            assert _chat(frontend_port).status_code == 404
            # The completion gate's own request went through the loopback
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
    tmp_path,
) -> None:
    system_port = int(dynamo_dynamic_ports.system_ports[0])
    runs = tmp_path / "runs"
    gates = {
        "count": {
            "exec": ["sh", "-c", f"echo run >> {runs}"],
            "timeout_secs": 60,
        }
    }

    for attempt in (1, 2):
        with GatedMockerWorker(
            request, name=f"worker-{attempt}", system_port=system_port, gates=gates
        ):
            _wait_for(lambda: _health(system_port)[0] == 200, "release")
            assert runs.read_text().splitlines() == ["run"] * attempt
