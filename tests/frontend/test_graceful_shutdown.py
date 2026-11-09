# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the frontend graceful-shutdown contract, CPU only.

The frontend is driven by ``dynamo.mocker`` so no GPU is needed. Each test starts
its own frontend with small shutdown timings, sends a real ``SIGTERM``, and checks
how in-flight and newly admitted requests behave.

Guards the ordering regression that shipped in v0.8-v1.2 (a request admitted during
the propagation window must still reach a worker; see issue #10862) and the drain,
force, and exit-time bounds of the shutdown controller.

Timing: each test launches one frontend and one mocker (about 5 s) and then spends
2-10 s in the shutdown it exercises; the whole file runs in about 40 s.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import subprocess
import threading
import time
from typing import Iterator

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,  # mocker is CPU-only
    pytest.mark.integration,
    pytest.mark.core,
    pytest.mark.timeout(120),  # ~10 s average per test, launch included
    pytest.mark.model(TEST_MODEL),
]

# A slow decode so a streaming response stays open across the SIGTERM.
SLOW_SPEEDUP = 0.3
STREAM_MAX_TOKENS = 200
# Long enough to outlast a short drain in the exit-time test.
LONG_STREAM_MAX_TOKENS = 4000


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
            speedup_ratio=SLOW_SPEEDUP,
        ):
            wait_for_http_completions_ready(
                frontend_port=frontend_port, model=TEST_MODEL
            )
            yield frontend


def _stream_completion(
    frontend_port: int,
    collected: list,
    done: threading.Event,
    max_tokens: int = STREAM_MAX_TOKENS,
) -> None:
    """Read a streaming completion to the end, recording every SSE line."""
    payload = {
        "model": TEST_MODEL,
        "prompt": "Tell me a long story about Mars.",
        "max_tokens": max_tokens,
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


def _start_stream(
    frontend_port: int, max_tokens: int = STREAM_MAX_TOKENS
) -> tuple[list, threading.Event, threading.Thread]:
    """Start a streaming request and return once its first token has arrived, so
    the test knows the stream was admitted before whatever it does next."""
    collected: list = []
    done = threading.Event()
    worker = threading.Thread(
        target=_stream_completion,
        args=(frontend_port, collected, done, max_tokens),
        daemon=True,
    )
    worker.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if any(kind == "line" for kind, _ in collected):
            return collected, done, worker
        if done.is_set():
            break
        time.sleep(0.05)
    raise AssertionError(f"stream produced no token in time: {collected}")


def _body(collected: list) -> str:
    return "\n".join(value for kind, value in collected if kind == "line")


def _sigterm(frontend: DynamoFrontendProcess) -> None:
    assert frontend.proc is not None
    frontend.proc.send_signal(signal.SIGTERM)


def _wait_health(frontend_port: int, status: int, timeout: float = 10) -> None:
    """Poll /health until it reports `status`; the signal handler runs on the
    frontend's event loop, so the stage change is not instantaneous."""
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = requests.get(
                f"http://localhost:{frontend_port}/health", timeout=2
            ).status_code
            if last == status:
                return
        except requests.RequestException as exc:
            last = str(exc)
        time.sleep(0.05)
    raise AssertionError(f"/health did not reach {status}; last was {last}")


def _wait_exit(frontend: DynamoFrontendProcess, timeout: float) -> int | None:
    assert frontend.proc is not None
    try:
        return frontend.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def test_stream_started_before_sigterm_completes(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """E1: a stream admitted before SIGTERM finishes cleanly during the drain and the
    frontend exits 0 (invariants I6, I11)."""
    ports = dynamo_dynamic_ports
    with frontend_and_mocker(
        request,
        ports.frontend_port,
        ports.system_ports[0],
        propagation=3,
        drain=30,
    ) as frontend:
        collected, done, worker = _start_stream(ports.frontend_port)
        _sigterm(frontend)

        assert done.wait(timeout=45), "stream did not finish during the drain"
        exit_code = _wait_exit(frontend, timeout=40)
        worker.join(timeout=5)

        statuses = [value for kind, value in collected if kind == "status"]
        assert statuses == [200], f"stream status was {statuses}"
        assert "[DONE]" in _body(
            collected
        ), "a stream admitted before SIGTERM must finish cleanly"
        assert exit_code == 0, f"frontend exit code was {exit_code}"


def test_request_admitted_in_window_completes(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """E2 + I1: once signaled, /health reports 503 while /live stays 200, and a new
    request inside the propagation window still completes with 200 rather than the
    503 and then 500 'instance not found' of #16519 / #10862 (invariants I1, I2,
    I11)."""
    ports = dynamo_dynamic_ports
    with frontend_and_mocker(
        request,
        ports.frontend_port,
        ports.system_ports[0],
        propagation=4,
        drain=20,
    ) as frontend:
        _sigterm(frontend)
        # Readiness is withdrawn as soon as the signal handler runs; the window
        # (4 s) is still open at that point.
        _wait_health(ports.frontend_port, 503)
        live = requests.get(f"http://localhost:{ports.frontend_port}/live", timeout=5)
        assert live.status_code == 200, f"/live was {live.status_code}"

        resp = requests.post(
            f"http://localhost:{ports.frontend_port}/v1/completions",
            json={"model": TEST_MODEL, "prompt": "hello", "max_tokens": 10},
            timeout=30,
        )
        assert resp.status_code == 200, (
            f"request admitted during the window must complete, got "
            f"{resp.status_code}: {resp.text}"
        )

        assert _wait_exit(frontend, timeout=30) == 0


def test_second_signal_forces_exit(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """E5: a second SIGTERM with a stream in flight forces a prompt exit and the
    stream does not finish cleanly (invariants I13, I7)."""
    ports = dynamo_dynamic_ports
    with frontend_and_mocker(
        request,
        ports.frontend_port,
        ports.system_ports[0],
        propagation=2,
        drain=60,
    ) as frontend:
        collected, done, worker = _start_stream(ports.frontend_port)

        _sigterm(frontend)
        _wait_health(ports.frontend_port, 503)
        forced_at = time.monotonic()
        _sigterm(frontend)  # force

        # Force skips the 60 s drain; exit must be prompt and still orderly (a
        # non-zero code would mean the default handler killed the process).
        exit_code = _wait_exit(frontend, timeout=15)
        elapsed = time.monotonic() - forced_at
        assert exit_code == 0, f"second signal did not force a prompt exit: {exit_code}"
        assert elapsed <= 10, f"forced exit took {elapsed:.1f}s"
        assert done.wait(timeout=10)
        worker.join(timeout=5)
        assert "[DONE]" not in _body(
            collected
        ), "a forced stream must not finish cleanly"


def test_exit_time_is_bounded(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """E6: with a stream that outlasts the drain, the frontend still exits within
    the propagation + drain + close budget, with exit code 0, and the stream is cut
    rather than completed (invariants I10, I7)."""
    ports = dynamo_dynamic_ports
    propagation, drain = 2, 4
    with frontend_and_mocker(
        request,
        ports.frontend_port,
        ports.system_ports[0],
        propagation=propagation,
        drain=drain,
    ) as frontend:
        collected, done, worker = _start_stream(
            ports.frontend_port, max_tokens=LONG_STREAM_MAX_TOKENS
        )

        start = time.monotonic()
        _sigterm(frontend)
        exit_code = _wait_exit(frontend, timeout=propagation + drain + 10)
        elapsed = time.monotonic() - start
        done.wait(timeout=5)
        worker.join(timeout=5)

        budget = propagation + drain + 5
        assert exit_code == 0, f"frontend exit code was {exit_code}"
        assert (
            elapsed <= budget
        ), f"frontend took {elapsed:.1f}s to exit, budget is {budget}s"
        assert "[DONE]" not in _body(
            collected
        ), "the long stream must have outlasted the drain and been cut"
