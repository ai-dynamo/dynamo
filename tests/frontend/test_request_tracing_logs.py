# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for request tracing log output (DIS-1643).

Verifies that JSONL logs contain consistent structured fields for all request
lifecycle events: "request received", "http response sent", "request completed".

Tests cover: unary success, streaming success, 404 error, 400 invalid UUID,
cancellation, frontend-worker trace_id correlation, aggregated deployment,
and disaggregated (prefill+decode) deployment.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
    pytest.mark.timeout(300),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_jsonl_logs(log_content: str) -> List[Dict[str, Any]]:
    """Parse JSONL log content into a list of dicts.

    Handles lines prefixed by ManagedProcess sed pipeline (e.g., '[PYTHON] {...}').
    """
    entries = []
    for line in log_content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Strip ManagedProcess sed prefix like "[PYTHON] " or "[PYTHON3] "
        json_start = line.find("{")
        if json_start >= 0:
            line = line[json_start:]
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def find_logs_by_request_id(
    entries: List[Dict[str, Any]], request_id: str
) -> List[Dict[str, Any]]:
    """Find all log entries that contain the given request_id anywhere in their fields."""
    return [e for e in entries if request_id in json.dumps(e)]


def read_log_file(process) -> str:
    """Read the log file from a ManagedProcess."""
    log_path = process.log_path
    if log_path and os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return ""


def _send_chat_completions(
    port: int,
    model: str = TEST_MODEL,
    request_id: Optional[str] = None,
    stream: bool = False,
    max_tokens: int = 5,
    timeout: int = 60,
) -> requests.Response:
    """Send a chat completions request with optional request ID and streaming."""
    headers = {"Content-Type": "application/json"}
    if request_id:
        headers["x-request-id"] = request_id
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": max_tokens,
        "stream": stream,
    }
    return requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=timeout,
    )


def get_request_logs(process, request_id: str) -> List[Dict[str, Any]]:
    """Read, parse, and filter logs by request_id."""
    return find_logs_by_request_id(parse_jsonl_logs(read_log_file(process)), request_id)


def assert_lifecycle_logs(req_logs, expected_status="success"):
    """Assert received/completed/http_sent exist and return them."""
    received = [e for e in req_logs if e.get("message") == "request received"]
    completed = [e for e in req_logs if e.get("message") == "request completed"]
    http_sent = [e for e in req_logs if e.get("message") == "http response sent"]
    msgs = [e.get("message") for e in req_logs]
    assert (
        len(received) == 1
    ), f"Expected 1 'request received', got {len(received)}: {msgs}"
    assert (
        len(completed) == 1
    ), f"Expected 1 'request completed', got {len(completed)}: {msgs}"
    assert (
        len(http_sent) == 1
    ), f"Expected 1 'http response sent', got {len(http_sent)}: {msgs}"
    assert completed[0].get("status") == expected_status
    return received, completed, http_sent


def assert_cancellation(req_logs):
    """Assert error completion with cancelled error_type is logged."""
    completed = [
        e
        for e in req_logs
        if e.get("message") == "request completed"
        and e.get("status") == "error"
        and e.get("error_type") == "cancelled"
    ]
    msgs = [(e.get("message"), e.get("status"), e.get("error_type")) for e in req_logs]
    assert (
        len(completed) == 1
    ), f"Expected 1 cancelled 'request completed', got {len(completed)}: {msgs}"
    return completed


def assert_error_completion(req_logs):
    """Assert exactly one error completion was logged (for crash scenarios)."""
    completed = [
        e
        for e in req_logs
        if e.get("message") == "request completed" and e.get("status") == "error"
    ]
    msgs = [e.get("message") for e in req_logs]
    assert (
        len(completed) == 1
    ), f"Expected 1 error 'request completed', got {len(completed)}: {msgs}"
    return completed


JSONL_ENV = {"DYN_LOGGING_JSONL": "1", "DYN_LOG": "info"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def tracing_services(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """Aggregated frontend + mocker with JSONL logging."""
    ports = dynamo_dynamic_ports
    with DynamoFrontendProcess(
        request,
        frontend_port=ports.frontend_port,
        terminate_all_matching_process_names=False,
        extra_env=JSONL_ENV,
    ) as frontend:
        with MockerWorkerProcess(
            request,
            model=TEST_MODEL,
            frontend_port=ports.frontend_port,
            system_port=ports.system_ports[0],
            extra_env=JSONL_ENV,
        ) as worker:
            wait_for_http_completions_ready(
                frontend_port=ports.frontend_port, model=TEST_MODEL
            )
            yield {
                "frontend_port": ports.frontend_port,
                "frontend": frontend,
                "worker": worker,
            }


@pytest.fixture(scope="function")
def tracing_services_slow(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """Aggregated frontend + slow mocker for cancellation/crash testing."""
    ports = dynamo_dynamic_ports
    with DynamoFrontendProcess(
        request,
        frontend_port=ports.frontend_port,
        terminate_all_matching_process_names=False,
        extra_env=JSONL_ENV,
    ) as frontend:
        with MockerWorkerProcess(
            request,
            model=TEST_MODEL,
            frontend_port=ports.frontend_port,
            system_port=ports.system_ports[0],
            speedup_ratio=0.1,
            extra_env=JSONL_ENV,
        ) as worker:
            wait_for_http_completions_ready(
                frontend_port=ports.frontend_port, model=TEST_MODEL
            )
            yield {
                "frontend_port": ports.frontend_port,
                "frontend": frontend,
                "worker": worker,
            }


@pytest.fixture(scope="function")
def tracing_services_disagg(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """Disaggregated frontend + prefill/decode mocker workers with JSONL logging."""
    ports = dynamo_dynamic_ports
    decode_system_port = allocate_port(8200)
    try:
        with DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            terminate_all_matching_process_names=False,
            extra_env=JSONL_ENV,
        ) as frontend:
            with MockerWorkerProcess(
                request,
                model=TEST_MODEL,
                frontend_port=ports.frontend_port,
                system_port=ports.system_ports[0],
                extra_args=["--disaggregation-mode", "prefill"],
                worker_id="prefill-worker",
                extra_env=JSONL_ENV,
            ) as prefill_worker:
                with MockerWorkerProcess(
                    request,
                    model=TEST_MODEL,
                    frontend_port=ports.frontend_port,
                    system_port=decode_system_port,
                    extra_args=["--disaggregation-mode", "decode"],
                    worker_id="decode-worker",
                    extra_env=JSONL_ENV,
                ) as decode_worker:
                    wait_for_http_completions_ready(
                        frontend_port=ports.frontend_port, model=TEST_MODEL
                    )
                    yield {
                        "frontend_port": ports.frontend_port,
                        "frontend": frontend,
                        "prefill_worker": prefill_worker,
                        "decode_worker": decode_worker,
                    }
    finally:
        deallocate_port(decode_system_port)


# ---------------------------------------------------------------------------
# Tests — Aggregated
# ---------------------------------------------------------------------------


def test_agg_unary_success(tracing_services) -> None:
    """Aggregated unary: full lifecycle logs + token counts + worker logs."""
    port = tracing_services["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid)
    assert resp.status_code == 200
    time.sleep(1)

    req_logs = get_request_logs(tracing_services["frontend"], rid)
    received, completed, http_sent = assert_lifecycle_logs(req_logs)

    assert received[0]["level"] == "INFO"
    assert received[0].get("x_request_id") == rid
    assert "request_id" in received[0]
    assert "model" in received[0]
    assert "endpoint" in received[0]
    assert "elapsed_ms" in completed[0]
    assert http_sent[0].get("status") == "200"

    # Token counts on inference span
    ic = [e for e in completed if e.get("span_name") == "http-request"]
    assert len(ic) == 1, "Expected 1 'request completed' from http-request span"
    assert "input_tokens" in ic[0]
    assert "output_tokens" in ic[0]

    # Worker lifecycle — verify both x_request_id and request_id propagated
    server_rid = received[0].get("request_id")
    wk_logs = get_request_logs(tracing_services["worker"], rid)
    wk_received = [e for e in wk_logs if e.get("message") == "request received"]
    wk_completed = [e for e in wk_logs if e.get("message") == "request completed"]
    assert len(wk_received) == 1, "Worker should log 1 'request received'"
    assert len(wk_completed) == 1, "Worker should log 1 'request completed'"
    assert wk_received[0].get("x_request_id") == rid, "Worker should have x_request_id"
    assert (
        wk_received[0].get("request_id") == server_rid
    ), "Worker request_id should match frontend"


def test_agg_streaming_success(tracing_services) -> None:
    """Aggregated streaming: lifecycle logs + token/latency metrics."""
    port = tracing_services["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid, stream=True, max_tokens=50)
    assert resp.status_code == 200
    _ = resp.content
    time.sleep(1)

    req_logs = get_request_logs(tracing_services["frontend"], rid)
    received, completed, http_sent = assert_lifecycle_logs(req_logs)

    assert received[0].get("request_type") == "stream"
    assert http_sent[0].get("status") == "200"

    # Token counts and latency on inference span
    ic = [e for e in completed if e.get("span_name") == "http-request"]
    assert len(ic) == 1, "Expected 1 'request completed' from http-request span"
    assert int(ic[0]["output_tokens"]) > 0
    assert "ttft_ms" in ic[0]
    assert "avg_itl_ms" in ic[0]


def test_agg_404_error(tracing_services) -> None:
    """Aggregated 404: ERROR lifecycle logs with not_found error type."""
    port = tracing_services["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, model="nonexistent-model", request_id=rid)
    assert resp.status_code == 404
    time.sleep(1)

    req_logs = get_request_logs(tracing_services["frontend"], rid)
    received, completed, http_sent = assert_lifecycle_logs(
        req_logs, expected_status="error"
    )

    assert completed[0]["level"] == "ERROR"
    assert completed[0].get("error_type") == "not_found"
    assert "error_detail" in completed[0]
    assert http_sent[0]["level"] == "ERROR"
    assert http_sent[0].get("status") == "404"


def test_agg_invalid_uuid_warn(tracing_services) -> None:
    """Invalid x-dynamo-request-id: WARN logged, request proceeds with generated ID."""
    port = tracing_services["frontend_port"]

    # Send deprecated x-dynamo-request-id with invalid value to test deprecation warning
    # Send both x-request-id (for log filtering) and deprecated x-dynamo-request-id (invalid)
    rid = str(uuid.uuid4())
    resp = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        headers={
            "Content-Type": "application/json",
            "x-request-id": rid,
            "x-dynamo-request-id": "NOT-A-VALID-UUID",
        },
        json={
            "model": TEST_MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5,
        },
        timeout=60,
    )
    assert resp.status_code == 200
    time.sleep(1)

    req_logs = get_request_logs(tracing_services["frontend"], rid)

    # A WARN log should be emitted about the invalid UUID
    warn_logs = [
        e
        for e in req_logs
        if e.get("level") == "WARN" and "must be a valid UUID" in e.get("message", "")
    ]
    assert len(warn_logs) == 1

    # Request still gets a valid request_id (generated, not the invalid one)
    received = [e for e in req_logs if e.get("message") == "request received"]
    assert len(received) == 1
    request_id = received[0].get("request_id", "")
    assert request_id != "NOT-A-VALID-UUID"
    try:
        uuid.UUID(request_id)
    except ValueError:
        pytest.fail(f"request_id is not a valid UUID: {request_id}")


def test_agg_request_id_propagation(tracing_services) -> None:
    """Frontend and worker share the same trace_id for a request."""
    port = tracing_services["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid, stream=True, max_tokens=20)
    assert resp.status_code == 200
    _ = resp.content
    time.sleep(1)

    fe_req = get_request_logs(tracing_services["frontend"], rid)
    assert len(fe_req) > 0, "Frontend should have logs for this request_id"

    fe_trace_ids = {e.get("trace_id") for e in fe_req if e.get("trace_id")}
    assert len(fe_trace_ids) == 1, f"Expected single trace_id, got: {fe_trace_ids}"
    trace_id = fe_trace_ids.pop()

    # Worker should have logs with same trace_id
    wk_logs = parse_jsonl_logs(read_log_file(tracing_services["worker"]))
    wk_with_trace = [e for e in wk_logs if e.get("trace_id") == trace_id]
    assert len(wk_with_trace) > 0, f"Worker should have logs with trace_id={trace_id}"


# ---------------------------------------------------------------------------
# Tests — Cancellation
# ---------------------------------------------------------------------------


def test_agg_cancellation(tracing_services_slow) -> None:
    """Client disconnect mid-stream triggers cancellation WARN log."""
    port = tracing_services_slow["frontend_port"]
    rid = str(uuid.uuid4())

    # Send streaming request, read a few bytes, then close the connection
    # to force a server-side cancellation detection.
    try:
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "x-request-id": rid,
            },
            json={
                "model": TEST_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 2000,
                "stream": True,
            },
            stream=True,  # Don't download body eagerly
            timeout=10,
        )
        # Read just enough to confirm stream started, then close
        for _ in resp.iter_lines():
            break  # read one line then stop
        resp.close()  # Force TCP connection close
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ReadTimeout,
    ):
        pass

    # Wait for cancellation to propagate
    time.sleep(3)

    fe_logs = parse_jsonl_logs(read_log_file(tracing_services_slow["frontend"]))
    req_logs = find_logs_by_request_id(fe_logs, rid)

    received = [e for e in req_logs if e.get("message") == "request received"]
    assert (
        len(received) == 1
    ), f"Expected 1 'request received', got {len(received)}: {[e.get('message') for e in req_logs]}"

    assert_cancellation(req_logs)


# ---------------------------------------------------------------------------
# Tests — Disaggregated
# ---------------------------------------------------------------------------


def test_disagg_streaming_success(tracing_services_disagg) -> None:
    """Disaggregated streaming: frontend + both workers log lifecycle with token metrics."""
    port = tracing_services_disagg["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid, stream=True, max_tokens=20)
    assert resp.status_code == 200
    _ = resp.content
    time.sleep(1)

    # Frontend lifecycle + token metrics
    fe_req = get_request_logs(tracing_services_disagg["frontend"], rid)
    received, completed, http_sent = assert_lifecycle_logs(fe_req)

    ic = [e for e in completed if e.get("span_name") == "http-request"]
    assert len(ic) == 1
    assert int(ic[0]["output_tokens"]) > 0
    assert "ttft_ms" in ic[0]

    # Both workers should log lifecycle
    for name in ("prefill_worker", "decode_worker"):
        wk_logs = get_request_logs(tracing_services_disagg[name], rid)
        wk_received = [e for e in wk_logs if e.get("message") == "request received"]
        wk_completed = [e for e in wk_logs if e.get("message") == "request completed"]
        assert len(wk_received) == 1, f"{name} should log 1 'request received'"
        assert len(wk_completed) == 1, f"{name} should log 1 'request completed'"


def test_agg_worker_crash(tracing_services_slow) -> None:
    """Kill mocker mid-stream: frontend should log ERROR with internal error."""
    port = tracing_services_slow["frontend_port"]
    worker = tracing_services_slow["worker"]
    rid = str(uuid.uuid4())

    def kill_worker_after_delay():
        """Kill the worker process after a short delay to simulate crash."""
        time.sleep(0.5)
        if worker.proc and worker.proc.poll() is None:
            worker.proc.kill()
            logger.info("Killed worker process to simulate crash")

    # Start the kill thread
    killer = threading.Thread(target=kill_worker_after_delay, daemon=True)
    killer.start()

    # Send streaming request — worker will be killed mid-stream
    try:
        resp = _send_chat_completions(
            port, request_id=rid, stream=True, max_tokens=2000, timeout=10
        )
        _ = resp.content  # try to consume
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    ):
        pass  # Expected if connection drops

    killer.join(timeout=5)
    time.sleep(2)

    req_logs = get_request_logs(tracing_services_slow["frontend"], rid)
    received = [e for e in req_logs if e.get("message") == "request received"]
    assert len(received) == 1
    assert_error_completion(req_logs)


def test_disagg_unary_success(tracing_services_disagg) -> None:
    """Disaggregated unary: frontend + both workers log lifecycle."""
    port = tracing_services_disagg["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid)
    assert resp.status_code == 200
    time.sleep(2)

    fe_req = get_request_logs(tracing_services_disagg["frontend"], rid)
    received, completed, http_sent = assert_lifecycle_logs(fe_req)
    assert http_sent[0].get("status") == "200"

    for name in ("prefill_worker", "decode_worker"):
        wk_logs = get_request_logs(tracing_services_disagg[name], rid)
        assert len(wk_logs) > 0, f"{name} should have logs for this request"


# ---------------------------------------------------------------------------
# Tests — Disaggregated crash scenarios
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def tracing_services_disagg_slow(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
):
    """Disaggregated frontend + slow prefill/decode workers for crash testing."""
    ports = dynamo_dynamic_ports
    decode_system_port = allocate_port(8200)
    try:
        with DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            terminate_all_matching_process_names=False,
            extra_env=JSONL_ENV,
        ) as frontend:
            with MockerWorkerProcess(
                request,
                model=TEST_MODEL,
                frontend_port=ports.frontend_port,
                system_port=ports.system_ports[0],
                speedup_ratio=0.1,
                extra_args=["--disaggregation-mode", "prefill"],
                worker_id="prefill-worker",
                extra_env=JSONL_ENV,
            ) as prefill_worker:
                with MockerWorkerProcess(
                    request,
                    model=TEST_MODEL,
                    frontend_port=ports.frontend_port,
                    system_port=decode_system_port,
                    speedup_ratio=0.1,
                    extra_args=["--disaggregation-mode", "decode"],
                    worker_id="decode-worker",
                    extra_env=JSONL_ENV,
                ) as decode_worker:
                    wait_for_http_completions_ready(
                        frontend_port=ports.frontend_port, model=TEST_MODEL
                    )
                    yield {
                        "frontend_port": ports.frontend_port,
                        "frontend": frontend,
                        "prefill_worker": prefill_worker,
                        "decode_worker": decode_worker,
                    }
    finally:
        deallocate_port(decode_system_port)


def test_disagg_prefill_crash(tracing_services_disagg_slow) -> None:
    """Kill prefill worker during request with large prompt: frontend should log error."""

    port = tracing_services_disagg_slow["frontend_port"]
    prefill = tracing_services_disagg_slow["prefill_worker"]
    rid = str(uuid.uuid4())

    # Use a large prompt to keep prefill busy long enough to kill it mid-request
    large_messages = [{"role": "user", "content": "Tell me a very long story. " * 100}]

    def kill_prefill_after_delay():
        time.sleep(0.1)  # Very short delay — kill during prefill processing
        if prefill.proc and prefill.proc.poll() is None:
            prefill.proc.kill()
            logger.info("Killed prefill worker mid-request")

    killer = threading.Thread(target=kill_prefill_after_delay, daemon=True)
    killer.start()

    try:
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "x-request-id": rid,
            },
            json={
                "model": TEST_MODEL,
                "messages": large_messages,
                "max_tokens": 2000,
                "stream": True,
            },
            stream=True,
            timeout=30,
        )
        for _ in resp.iter_lines():
            pass
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ReadTimeout,
    ):
        pass

    killer.join(timeout=5)
    time.sleep(3)

    req_logs = get_request_logs(tracing_services_disagg_slow["frontend"], rid)
    assert len(req_logs) > 0, f"Frontend should have logs for request {rid}"
    assert_error_completion(req_logs)


def test_disagg_decode_crash(tracing_services_disagg_slow) -> None:
    """Kill decode worker mid-stream: frontend should log error."""

    port = tracing_services_disagg_slow["frontend_port"]
    decode = tracing_services_disagg_slow["decode_worker"]
    rid = str(uuid.uuid4())

    def kill_decode_after_delay():
        time.sleep(0.5)
        if decode.proc and decode.proc.poll() is None:
            decode.proc.kill()
            logger.info("Killed decode worker to simulate crash")

    killer = threading.Thread(target=kill_decode_after_delay, daemon=True)
    killer.start()

    try:
        resp = _send_chat_completions(
            port, request_id=rid, stream=True, max_tokens=2000, timeout=10
        )
        _ = resp.content
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.ChunkedEncodingError,
    ):
        pass

    killer.join(timeout=5)
    time.sleep(3)

    req_logs = get_request_logs(tracing_services_disagg_slow["frontend"], rid)
    received = [e for e in req_logs if e.get("message") == "request received"]
    assert len(received) == 1, "Frontend should log 1 'request received'"
    assert_error_completion(req_logs)
