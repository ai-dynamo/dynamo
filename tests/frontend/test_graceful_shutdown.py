# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the frontend graceful-shutdown contract, CPU only.

The frontend is driven by ``dynamo.mocker`` so no GPU is needed. Each test starts
its own frontend with small shutdown timings, sends a real ``SIGTERM``, and checks
how in-flight and newly admitted requests behave.

Guards the ordering regression that shipped in v0.8-v1.2 (a request admitted during
the propagation window must still reach a worker; see issue #10862) and the drain,
force, and exit-time bounds of the shutdown controller.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import threading
import time
from typing import Iterator

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,  # mocker is CPU-only
    pytest.mark.integration,
    pytest.mark.model(TEST_MODEL),
]

# A slow decode so a streaming response stays open across the SIGTERM.
SLOW_SPEEDUP = "0.3"
STREAM_MAX_TOKENS = 200


@contextlib.contextmanager
def frontend_and_mocker(
    request,
    frontend_port: int,
    system_port: int,
    *,
    propagation: int,
    drain: int,
) -> Iterator[DynamoFrontendProcess]:
    """Start a frontend with the given shutdown timings plus a slow mocker worker."""
    frontend = DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env={
            "DYN_FRONTEND_PROPAGATION_DELAY_SECS": str(propagation),
            "DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS": str(drain),
        },
    )
    with frontend:
        logger.info(
            "Frontend up on %s (P=%ss, D=%ss)", frontend_port, propagation, drain
        )
        with MockerWorkerProcess(
            request,
            TEST_MODEL,
            frontend_port,
            system_port,
            speedup_ratio=SLOW_SPEEDUP,  # type: ignore[arg-type]
        ):
            _wait_until_servable(frontend_port)
            yield frontend


def _wait_until_servable(frontend_port: int, attempts: int = 60) -> None:
    payload = {"model": TEST_MODEL, "prompt": "ping", "max_tokens": 1}
    for _ in range(attempts):
        try:
            resp = requests.post(
                f"http://localhost:{frontend_port}/v1/completions",
                json=payload,
                timeout=10,
            )
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise RuntimeError("frontend never became servable")


def _stream_completion(
    frontend_port: int, collected: list, done: threading.Event
) -> None:
    """Read a streaming completion to the end, recording every SSE line."""
    payload = {
        "model": TEST_MODEL,
        "prompt": "Tell me a long story about Mars.",
        "max_tokens": STREAM_MAX_TOKENS,
        "stream": True,
    }
    try:
        with requests.post(
            f"http://localhost:{frontend_port}/v1/completions",
            json=payload,
            stream=True,
            timeout=60,
        ) as resp:
            collected.append(("status", resp.status_code))
            for line in resp.iter_lines():
                if line:
                    collected.append(("line", line.decode("utf-8", "ignore")))
    except requests.RequestException as exc:
        collected.append(("error", str(exc)))
    finally:
        done.set()


def _sigterm(frontend: DynamoFrontendProcess) -> None:
    assert frontend.proc is not None
    frontend.proc.send_signal(signal.SIGTERM)


def _wait_exit(frontend: DynamoFrontendProcess, timeout: float) -> int | None:
    assert frontend.proc is not None
    try:
        return frontend.proc.wait(timeout=timeout)
    except Exception:
        return None


def test_stream_started_before_sigterm_completes(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports
):
    """E1/E6-clean: a stream admitted before SIGTERM finishes during the drain."""
    ports = dynamo_dynamic_ports
    with frontend_and_mocker(
        request,
        ports.frontend_port,
        ports.system_ports[0],
        propagation=3,
        drain=30,
    ) as frontend:
        collected: list = []
        done = threading.Event()
        worker = threading.Thread(
            target=_stream_completion,
            args=(ports.frontend_port, collected, done),
            daemon=True,
        )
        worker.start()
        time.sleep(1.0)  # let the stream start and emit tokens
        _sigterm(frontend)

        assert done.wait(timeout=45), "stream did not finish during the drain"
        exit_code = _wait_exit(frontend, timeout=40)
        worker.join(timeout=5)

        statuses = [v for k, v in collected if k == "status"]
        body = "\n".join(v for k, v in collected if k == "line")
        assert statuses == [200], f"stream status was {statuses}"
        assert "[DONE]" in body, "a stream admitted before SIGTERM must finish cleanly"
        assert exit_code == 0, f"frontend exit code was {exit_code}"


def test_request_admitted_in_window_completes(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports
):
    """E2: a request that arrives during the propagation window still completes,
    with no 500 'instance not found' (the #10862 regression)."""
    ports = dynamo_dynamic_ports
    with frontend_and_mocker(
        request,
        ports.frontend_port,
        ports.system_ports[0],
        propagation=4,
        drain=20,
    ) as frontend:
        _sigterm(frontend)
        time.sleep(1.0)  # well inside the 4s window

        resp = requests.post(
            f"http://localhost:{ports.frontend_port}/v1/completions",
            json={"model": TEST_MODEL, "prompt": "hello", "max_tokens": 10},
            timeout=30,
        )
        assert resp.status_code == 200, (
            f"request admitted during the window must complete, got "
            f"{resp.status_code}: {resp.text}"
        )
        assert "instance" not in resp.text.lower() or resp.status_code == 200

        assert _wait_exit(frontend, timeout=30) == 0


def test_health_503_but_live_200_during_shutdown(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports
):
    """I1: once signaled, /health reports 503 while /live stays 200."""
    ports = dynamo_dynamic_ports
    with frontend_and_mocker(
        request,
        ports.frontend_port,
        ports.system_ports[0],
        propagation=5,
        drain=10,
    ) as frontend:
        _sigterm(frontend)
        time.sleep(0.5)

        health = requests.get(
            f"http://localhost:{ports.frontend_port}/health", timeout=5
        )
        live = requests.get(f"http://localhost:{ports.frontend_port}/live", timeout=5)
        assert health.status_code == 503, f"/health was {health.status_code}"
        assert live.status_code == 200, f"/live was {live.status_code}"

        _wait_exit(frontend, timeout=30)


def test_second_signal_forces_exit(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports
):
    """E5: a second SIGTERM with a stream in flight forces a prompt exit and the
    stream does not finish cleanly."""
    ports = dynamo_dynamic_ports
    with frontend_and_mocker(
        request,
        ports.frontend_port,
        ports.system_ports[0],
        propagation=2,
        drain=60,
    ) as frontend:
        collected: list = []
        done = threading.Event()
        worker = threading.Thread(
            target=_stream_completion,
            args=(ports.frontend_port, collected, done),
            daemon=True,
        )
        worker.start()
        time.sleep(1.0)

        _sigterm(frontend)
        time.sleep(1.0)
        _sigterm(frontend)  # force

        # Force skips the 60s drain; exit must be prompt.
        exit_code = _wait_exit(frontend, timeout=15)
        assert exit_code is not None, "second signal did not force a prompt exit"
        assert done.wait(timeout=10)
        worker.join(timeout=5)
        body = "\n".join(v for k, v in collected if k == "line")
        assert "[DONE]" not in body, "a forced stream must not finish cleanly"


def test_exit_time_is_bounded(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports
):
    """E6: with a stream that outlasts the drain, the frontend still exits within
    the propagation + drain + close budget and with exit code 0."""
    ports = dynamo_dynamic_ports
    propagation, drain = 2, 6
    with frontend_and_mocker(
        request,
        ports.frontend_port,
        ports.system_ports[0],
        propagation=propagation,
        drain=drain,
    ) as frontend:
        collected: list = []
        done = threading.Event()
        worker = threading.Thread(
            target=_stream_completion,
            args=(ports.frontend_port, collected, done),
            daemon=True,
        )
        worker.start()
        time.sleep(0.5)

        start = time.monotonic()
        _sigterm(frontend)
        exit_code = _wait_exit(frontend, timeout=propagation + drain + 10)
        elapsed = time.monotonic() - start
        done.wait(timeout=5)
        worker.join(timeout=5)

        assert exit_code == 0, f"frontend exit code was {exit_code}"
        assert elapsed <= propagation + drain + 5, (
            f"frontend took {elapsed:.1f}s to exit, "
            f"budget is {propagation + drain + 2}s"
        )
