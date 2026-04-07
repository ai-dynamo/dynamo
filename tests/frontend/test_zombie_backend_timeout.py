# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for backend stream inactivity timeout (issue #7545).

These tests verify that the frontend's zombie backend detection works end-to-end:
when a backend holds a live TCP connection but produces no output (or stalls
mid-stream), the inactivity timeout fires, the request is cancelled, and the
inflight gauge recovers.

The fix under test is the third ``select!`` arm in ``monitor_for_disconnects``
(lib/llm/src/http/service/disconnect.rs), gated by the
``DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS`` environment variable.

These tests are CPU-only (no GPU required).
"""

import logging
import os
import re
import shutil
import time

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

logger = logging.getLogger(__name__)

MODEL = QWEN

# Short timeout for tests -- long enough for the backend to register but short
# enough that tests don't take forever waiting for the timeout to fire.
BACKEND_STREAM_TIMEOUT_SECS = 5

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.integration,
    pytest.mark.model(MODEL),
]

# Path to the zombie backend script
ZOMBIE_BACKEND_SCRIPT = os.path.join(os.path.dirname(__file__), "zombie_backend.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class ZombieWorkerProcess(ManagedProcess):
    """Worker process that runs the zombie backend script.

    The zombie backend registers as a normal Dynamo backend via etcd/TCP but
    stalls when handling requests, simulating a zombie worker.
    """

    def __init__(
        self,
        request,
        model,
        frontend_port,
        system_port,
        stall_after_tokens=0,
        worker_id="zombie-worker",
    ):
        self.worker_id = worker_id
        self.frontend_port = frontend_port
        self.system_port = system_port

        command = [
            "python3",
            ZOMBIE_BACKEND_SCRIPT,
            "--model-path",
            model,
            "--stall-after-tokens",
            str(stall_after_tokens),
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "debug"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(system_port)

        log_dir = f"{request.node.name}_{worker_id}"

        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (
                    f"http://localhost:{frontend_port}/v1/models",
                    self._check_models_api,
                ),
                (f"http://localhost:{system_port}/health", self.is_ready),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=[],
            straggler_commands=["zombie_backend.py"],
            log_dir=log_dir,
        )

    def _check_models_api(self, response):
        try:
            if response.status_code != 200:
                return False
            data = response.json()
            models = data.get("data", [])
            return len(models) > 0
        except Exception:
            return False

    def is_ready(self, response) -> bool:
        try:
            status = (response.json() or {}).get("status")
        except ValueError:
            logger.warning("%s health response is not valid JSON", self.worker_id)
            return False

        is_ready = status == "ready"
        if is_ready:
            logger.info("%s status is ready", self.worker_id)
        else:
            logger.warning("%s status is not ready: %s", self.worker_id, status)
        return is_ready


def _start_frontend_with_timeout(
    request, frontend_port, timeout_secs=BACKEND_STREAM_TIMEOUT_SECS
):
    """Start a DynamoFrontendProcess with the backend stream timeout configured."""
    return DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env={
            "DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS": str(timeout_secs),
        },
        terminate_all_matching_process_names=False,
    )


def _streaming_chat_request(frontend_port, model, max_tokens=10, timeout=30):
    """Send a streaming chat completion request and return the raw response."""
    return requests.post(
        f"http://localhost:{frontend_port}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
            "max_tokens": max_tokens,
        },
        stream=True,
        timeout=timeout,
    )


def _get_inflight_count(frontend_port):
    """Query the frontend /metrics endpoint and extract the inflight gauge value.

    Returns the sum of all inflight requests across all models, or 0 if the
    metric is not present. Returns None on fetch failure.
    """
    try:
        resp = requests.get(f"http://localhost:{frontend_port}/metrics", timeout=5)
        if resp.status_code != 200:
            return None
        total = 0
        found = False
        for line in resp.text.splitlines():
            if "dynamo_frontend_inflight_requests" in line and not line.startswith("#"):
                match = re.search(r"}\s+(\d+(?:\.\d+)?)", line)
                if match:
                    total += int(float(match.group(1)))
                    found = True
        return total if found else 0
    except Exception as e:
        logger.warning("Failed to query metrics: %s", e)
        return None


def _drain_sse_stream(resp):
    """Consume an SSE response stream, returning (chunks, saw_done, saw_error)."""
    chunks = []
    saw_done = False
    saw_error = False
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            payload = line[len("data: ") :]
            if payload == "[DONE]":
                saw_done = True
                break
            chunks.append(payload)
        if "error" in line.lower():
            saw_error = True
    return chunks, saw_done, saw_error


# ---------------------------------------------------------------------------
# Test 1: Zombie backend is detected by the inactivity timeout
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_zombie_backend_detected_by_timeout(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """A pure zombie backend (stall_after_tokens=0) is detected and timed out.

    Without the fix, the streaming request would hang forever. With the
    inactivity timeout, the frontend closes the stream within ~TIMEOUT seconds.
    """
    ports = dynamo_dynamic_ports
    frontend_port = ports.frontend_port
    system_port = ports.system_ports[0]

    with _start_frontend_with_timeout(request, frontend_port):
        with ZombieWorkerProcess(
            request, MODEL, frontend_port, system_port, stall_after_tokens=0
        ):
            wait_for_http_completions_ready(frontend_port=frontend_port, model=MODEL)
            logger.info("Zombie worker registered, sending streaming request")

            start = time.monotonic()
            resp = _streaming_chat_request(
                frontend_port,
                MODEL,
                max_tokens=10,
                timeout=BACKEND_STREAM_TIMEOUT_SECS + 30,
            )

            chunks, saw_done, saw_error = _drain_sse_stream(resp)
            elapsed = time.monotonic() - start

            logger.info(
                "Streaming request completed in %.1fs (timeout=%ds), "
                "chunks=%d, saw_done=%s, saw_error=%s",
                elapsed,
                BACKEND_STREAM_TIMEOUT_SECS,
                len(chunks),
                saw_done,
                saw_error,
            )

            # The request should complete within a reasonable window around
            # the configured timeout (allow generous margin for process startup).
            assert elapsed < BACKEND_STREAM_TIMEOUT_SECS + 20, (
                f"Request took {elapsed:.1f}s which is too long -- "
                f"the inactivity timeout ({BACKEND_STREAM_TIMEOUT_SECS}s) "
                f"may not be working"
            )

            logger.info("Test passed: zombie detected within timeout window")


# ---------------------------------------------------------------------------
# Test 2: Inflight gauge recovers after timeout fires
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_inflight_gauge_recovers_after_timeout(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """After the inactivity timeout fires on a zombie backend, the
    dynamo_frontend_inflight_requests gauge must return to 0.

    This is the core regression test for issue #7545: without the fix,
    InflightGuard::drop() never fires and the gauge leaks.
    """
    ports = dynamo_dynamic_ports
    frontend_port = ports.frontend_port
    system_port = ports.system_ports[0]

    with _start_frontend_with_timeout(request, frontend_port):
        with ZombieWorkerProcess(
            request, MODEL, frontend_port, system_port, stall_after_tokens=0
        ):
            wait_for_http_completions_ready(frontend_port=frontend_port, model=MODEL)

            initial_inflight = _get_inflight_count(frontend_port)
            logger.info("Initial inflight count: %s", initial_inflight)

            resp = _streaming_chat_request(
                frontend_port,
                MODEL,
                max_tokens=10,
                timeout=BACKEND_STREAM_TIMEOUT_SECS + 30,
            )

            _drain_sse_stream(resp)

            # Give the frontend a moment to process the timeout and update metrics
            time.sleep(2)

            inflight_after = _get_inflight_count(frontend_port)
            logger.info("Inflight count after timeout: %s", inflight_after)

            assert (
                inflight_after is not None
            ), "Could not read inflight metric from /metrics endpoint"
            assert inflight_after == 0, (
                f"Inflight gauge is {inflight_after} after timeout -- "
                f"InflightGuard was not dropped (issue #7545 regression)"
            )


# ---------------------------------------------------------------------------
# Test 3: Normal requests work fine with the timeout enabled
# ---------------------------------------------------------------------------


@pytest.mark.timeout(120)
def test_normal_requests_work_with_timeout_enabled(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """With DYN_HTTP_BACKEND_STREAM_TIMEOUT_SECS configured, normal (non-zombie)
    streaming requests should still complete successfully.

    This proves the timeout doesn't kill healthy streams.
    """
    ports = dynamo_dynamic_ports
    frontend_port = ports.frontend_port
    system_port = ports.system_ports[0]

    with _start_frontend_with_timeout(request, frontend_port):
        with MockerWorkerProcess(
            request, MODEL, frontend_port, system_port, speedup_ratio=100
        ):
            wait_for_http_completions_ready(frontend_port=frontend_port, model=MODEL)

            resp = _streaming_chat_request(
                frontend_port,
                MODEL,
                max_tokens=10,
                timeout=30,
            )
            resp.raise_for_status()

            chunks, saw_done, _ = _drain_sse_stream(resp)

            assert saw_done, "Missing [DONE] marker -- stream did not complete"
            assert (
                len(chunks) > 0
            ), "Expected token chunks from normal mocker but got none"

            logger.info(
                "Normal request completed successfully with %d chunks "
                "(timeout was enabled at %ds)",
                len(chunks),
                BACKEND_STREAM_TIMEOUT_SECS,
            )

            # Verify inflight gauge is clean
            time.sleep(1)
            inflight = _get_inflight_count(frontend_port)
            if inflight is not None:
                assert (
                    inflight == 0
                ), f"Inflight gauge is {inflight} after normal completion"
