# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real TCP/frontend coverage with discovery removal held behind a file-store mirror."""

import os
import sys
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

import pytest
import requests

from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.core,
]


def _until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = predicate()
            if result:
                return result
        except requests.ConnectionError:
            pass
        time.sleep(0.05)
    raise AssertionError("withdrawal integration condition did not become true")


class _DiscoveryMirror:
    """Delay deletions only; the real file discovery backend watches the mirror."""

    def __init__(self, source: Path, target: Path):
        self.source, self.target = source, target
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error = None

    def _run(self):
        try:
            while not self.stop.is_set():
                for path in self.source.rglob("*"):
                    if not path.is_file() or path.name.startswith("."):
                        continue
                    relative = path.relative_to(self.source)
                    destination = self.target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        payload = path.read_bytes()
                    except FileNotFoundError:
                        continue
                    if not destination.exists() or destination.read_bytes() != payload:
                        temporary = destination.with_name(".tmp_" + destination.name)
                        temporary.write_bytes(payload)
                        temporary.replace(destination)
                for path in self.target.rglob("*"):
                    if path.is_file() and not path.name.startswith("."):
                        # File discovery expires unrefreshed records after ten seconds.
                        path.touch()
                self.stop.wait(0.1)
        except OSError as error:
            self.error = error

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=5)
        assert not self.thread.is_alive()
        assert self.error is None, self.error


@pytest.mark.timeout(30)
@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_withdrawal_survives_stale_discovery_and_healthy_probes(
    request, tmp_path, dynamo_dynamic_ports
):
    model_path = (
        Path(__file__).resolve().parents[2]
        / "lib/llm/tests/data/sample-models/TinyLlama_v1.1"
    )
    namespace = f"withdrawal-{uuid4().hex}"
    live, delayed = tmp_path / "live", tmp_path / "delayed"
    live.mkdir()
    delayed.mkdir()
    bases = [f"http://127.0.0.1:{port}" for port in dynamo_dynamic_ports.system_ports]

    def control(index, route, method="GET"):
        response = requests.request(method, f"{bases[index]}/engine/{route}", timeout=5)
        response.raise_for_status()
        return response.json()

    def state(index):
        return control(index, "test/state", "POST")

    env = {
        **os.environ,
        "DYN_FILE_KV": str(live),
        "DYN_EXPERIMENTAL_WORKER_WITHDRAWAL": "1",
        "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS": "5",
        "DYN_HEALTH_CHECK_ENABLED": "true",
        "DYN_CANARY_WAIT_TIME": "1",
        "CUDA_VISIBLE_DEVICES": "",
    }
    with ExitStack() as stack:
        for index, name in enumerate(("a", "b")):
            stack.enter_context(
                ManagedProcess(
                    command=[
                        sys.executable,
                        "-m",
                        "tests.runtime.withdrawal_worker",
                        namespace,
                        name,
                        str(model_path),
                    ],
                    env={
                        **env,
                        "DYN_SYSTEM_PORT": str(
                            dynamo_dynamic_ports.system_ports[index]
                        ),
                    },
                    log_dir=f"{request.node.name}_{name}",
                    terminate_all_matching_process_names=False,
                    display_output=True,
                    display_name=f"withdrawal-worker-{name}",
                )
            )
        for base_url in bases:
            _until(
                lambda: requests.post(
                    f"{base_url}/engine/test/state", timeout=5
                ).status_code
                == 200
            )
        _until(lambda: len(state(1)["instances"]) == 2)
        worker_a = state(0)["worker_id"]
        stack.enter_context(_DiscoveryMirror(live, delayed))
        frontend = stack.enter_context(
            DynamoFrontendProcess(
                request,
                frontend_port=dynamo_dynamic_ports.frontend_port,
                migration_limit=0,
                extra_args=[
                    "--namespace",
                    namespace,
                    "--discovery-backend",
                    "file",
                    "--request-plane",
                    "tcp",
                ],
                extra_env={
                    **env,
                    "DYN_FILE_KV": str(delayed),
                    "DYN_RESPONSE_PLANE": "tcp",
                },
            )
        )
        base = f"http://127.0.0.1:{frontend.frontend_port}"
        _until(
            lambda: "withdrawal-test"
            in {
                model["id"]
                for model in requests.get(f"{base}/v1/models", timeout=5)
                .json()
                .get("data", [])
            }
        )
        _until(lambda: control(0, "worker_lifecycle/status")["state"] == "serving")

        def generate(max_tokens=2, stream=False):
            response = requests.post(
                f"{base}/v1/chat/completions",
                json={
                    "model": "withdrawal-test",
                    "messages": [{"role": "user", "content": "hello"}],
                    "max_tokens": max_tokens,
                    "stream": stream,
                },
                stream=stream,
                timeout=15,
            )
            response.raise_for_status()
            if not stream:
                body = response.json()
                assert "error" not in body, body
                assert body["choices"][0]["finish_reason"] == "stop", body
            return response

        # Round robin selects A within two calls; no explicit worker pin bypasses reselection.
        streams = []
        for _ in range(2):
            other_before = state(1)["business"]
            response = generate(64, True)
            streams.append(response)
            stack.callback(response.close)
            _until(lambda: state(0)["held"] or state(1)["business"] > other_before)
            if state(0)["held"]:
                break
            list(response.iter_lines())
        assert state(0)["held"]
        drained = control(0, "worker_lifecycle/drain", "POST")
        assert drained["state"] == "withdrawing"
        assert drained["accepting_requests"]
        admitted = state(0)["business"]
        for _ in range(2):
            generate()
        assert state(0)["business"] > admitted
        _until(lambda: worker_a not in state(1)["instances"])
        _until(lambda: state(0)["state"] == "withdrawn")
        before = state(0)
        other = state(1)["business"]
        for _ in range(2):
            generate()
        after = state(0)
        assert after["rejected"] > before["rejected"]
        assert after["business"] == before["business"]
        assert state(1)["business"] > other

        probes = after["probes"]
        for target in (probes + 1, probes + 2):
            _until(lambda: state(0)["probes"] >= target)
            response = requests.get(f"{bases[0]}/health", timeout=5)
            assert response.status_code == 200, response.text
            assert worker_a not in state(1)["instances"]
            withdrawn = state(0)
            assert withdrawn["state"] == "withdrawn"
            assert not withdrawn["accepting_requests"]
        control(0, "test/release", "POST")
        completed = list(streams[-1].iter_lines())
        assert any(b"[DONE]" in line for line in completed), completed
        assert any(
            b'"finish_reason":"stop"' in line.replace(b" ", b"") for line in completed
        ), completed
        assert not any(b'"error"' in line for line in completed), completed
        assert state(0)["state"] == "withdrawn"
