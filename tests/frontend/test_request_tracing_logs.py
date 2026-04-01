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
import shutil
import time
import uuid
from typing import Any, Dict, List, Optional

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import allocate_port, deallocate_port

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
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
        headers["x-dynamo-request-id"] = request_id
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


# ---------------------------------------------------------------------------
# JSONL-enabled worker subclass
# ---------------------------------------------------------------------------


class JsonlMockerWorkerProcess(MockerWorkerProcess):
    """MockerWorkerProcess with JSONL logging enabled at INFO level."""

    def __init__(
        self,
        request,
        model,
        frontend_port,
        system_port,
        speedup_ratio=100,
        extra_args=None,
        **kwargs,
    ):
        super().__init__(
            request,
            model,
            frontend_port,
            system_port,
            speedup_ratio=speedup_ratio,
            **kwargs,
        )
        # Override env to enable JSONL at INFO level
        self.env["DYN_LOGGING_JSONL"] = "1"
        self.env["DYN_LOG"] = "info"
        # Append extra mocker args (e.g., --disaggregation-mode)
        if extra_args:
            self.command.extend(extra_args)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def tracing_services(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports, predownload_tokenizers
):
    """Start frontend + mocker (aggregated) with JSONL logging at INFO level.

    Yields dict with frontend_port, frontend_process, worker_process for log access.
    """
    ports = dynamo_dynamic_ports

    jsonl_env = {
        "DYN_LOGGING_JSONL": "1",
        "DYN_LOG": "info",
    }

    with DynamoFrontendProcess(
        request,
        frontend_port=ports.frontend_port,
        terminate_all_matching_process_names=False,
        extra_env=jsonl_env,
    ) as frontend_process:
        logger.info(f"JSONL Frontend started on port {ports.frontend_port}")

        with JsonlMockerWorkerProcess(
            request,
            model=TEST_MODEL,
            frontend_port=ports.frontend_port,
            system_port=ports.system_ports[0],
        ) as worker_process:
            wait_for_http_completions_ready(
                frontend_port=ports.frontend_port, model=TEST_MODEL
            )
            logger.info("JSONL Mocker worker ready")

            yield {
                "frontend_port": ports.frontend_port,
                "frontend": frontend_process,
                "worker": worker_process,
            }


@pytest.fixture(scope="function")
def tracing_services_slow(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports, predownload_tokenizers
):
    """Start frontend + slow mocker for cancellation testing.

    Uses speedup_ratio=0.1 so streaming takes long enough to cancel mid-stream.
    """
    ports = dynamo_dynamic_ports

    jsonl_env = {
        "DYN_LOGGING_JSONL": "1",
        "DYN_LOG": "info",
    }

    with DynamoFrontendProcess(
        request,
        frontend_port=ports.frontend_port,
        terminate_all_matching_process_names=False,
        extra_env=jsonl_env,
    ) as frontend_process:
        logger.info(f"JSONL Frontend (slow) started on port {ports.frontend_port}")

        with JsonlMockerWorkerProcess(
            request,
            model=TEST_MODEL,
            frontend_port=ports.frontend_port,
            system_port=ports.system_ports[0],
            speedup_ratio=0.1,  # 10x slower than real-time for cancellation/crash testing
        ) as worker_process:
            wait_for_http_completions_ready(
                frontend_port=ports.frontend_port, model=TEST_MODEL
            )
            logger.info("JSONL Slow mocker worker ready")

            yield {
                "frontend_port": ports.frontend_port,
                "frontend": frontend_process,
                "worker": worker_process,
            }


@pytest.fixture(scope="function")
def tracing_services_disagg(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports, predownload_tokenizers
):
    """Start frontend + disaggregated prefill+decode mocker workers with JSONL logging."""
    ports = dynamo_dynamic_ports

    jsonl_env = {
        "DYN_LOGGING_JSONL": "1",
        "DYN_LOG": "info",
    }

    # Need a second system port for the decode worker
    decode_system_port = allocate_port(8200)

    try:
        with DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            terminate_all_matching_process_names=False,
            extra_env=jsonl_env,
        ) as frontend_process:
            logger.info(
                f"JSONL Frontend (disagg) started on port {ports.frontend_port}"
            )

            with JsonlMockerWorkerProcess(
                request,
                model=TEST_MODEL,
                frontend_port=ports.frontend_port,
                system_port=ports.system_ports[0],
                extra_args=["--disaggregation-mode", "prefill"],
                worker_id="prefill-worker",
            ) as prefill_worker:
                with JsonlMockerWorkerProcess(
                    request,
                    model=TEST_MODEL,
                    frontend_port=ports.frontend_port,
                    system_port=decode_system_port,
                    extra_args=["--disaggregation-mode", "decode"],
                    worker_id="decode-worker",
                ) as decode_worker:
                    wait_for_http_completions_ready(
                        frontend_port=ports.frontend_port, model=TEST_MODEL
                    )
                    logger.info("JSONL Disagg workers ready")

                    yield {
                        "frontend_port": ports.frontend_port,
                        "frontend": frontend_process,
                        "prefill_worker": prefill_worker,
                        "decode_worker": decode_worker,
                    }
    finally:
        deallocate_port(decode_system_port)


# ---------------------------------------------------------------------------
# Tests — Aggregated
# ---------------------------------------------------------------------------


def test_agg_unary_success(tracing_services) -> None:
    """Aggregated unary chat completion: verify full lifecycle logs."""
    port = tracing_services["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid)
    assert resp.status_code == 200
    time.sleep(1)

    fe_logs = parse_jsonl_logs(read_log_file(tracing_services["frontend"]))
    req_logs = find_logs_by_request_id(fe_logs, rid)

    # "request received" at INFO
    received = [e for e in req_logs if e.get("message") == "request received"]
    assert len(received) >= 1, f"Expected 'request received', got: {[e.get('message') for e in req_logs]}"
    assert received[0]["level"] == "INFO"
    assert received[0].get("request_id") == rid
    assert "model" in received[0]
    assert "endpoint" in received[0]

    # "request completed" at INFO with status=success
    completed = [e for e in req_logs if e.get("message") == "request completed"]
    assert len(completed) >= 1
    assert completed[0]["level"] == "INFO"
    assert completed[0].get("status") == "success"
    assert "elapsed_ms" in completed[0]

    # "http response sent" at INFO with status=200
    http_sent = [e for e in req_logs if e.get("message") == "http response sent"]
    assert len(http_sent) >= 1
    assert http_sent[0]["level"] == "INFO"
    assert http_sent[0].get("status") == "200"

    # Worker received and completed
    wk_logs = parse_jsonl_logs(read_log_file(tracing_services["worker"]))
    wk_req = find_logs_by_request_id(wk_logs, rid)
    wk_received = [e for e in wk_req if e.get("message") == "request received"]
    wk_completed = [e for e in wk_req if e.get("message") == "request completed"]
    assert len(wk_received) >= 1, "Worker should log 'request received'"
    assert len(wk_completed) >= 1, "Worker should log 'request completed'"


def test_agg_streaming_success(tracing_services) -> None:
    """Aggregated streaming chat completion: verify token counts on completion."""
    port = tracing_services["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid, stream=True, max_tokens=50)
    assert resp.status_code == 200
    _ = resp.content  # consume stream
    time.sleep(1)

    fe_logs = parse_jsonl_logs(read_log_file(tracing_services["frontend"]))
    req_logs = find_logs_by_request_id(fe_logs, rid)

    # "request received" with request_type=stream
    received = [e for e in req_logs if e.get("message") == "request received"]
    assert len(received) >= 1
    assert received[0].get("request_type") == "stream"

    # "http response sent" at stream start
    http_sent = [e for e in req_logs if e.get("message") == "http response sent"]
    assert len(http_sent) >= 1
    assert http_sent[0].get("status") == "200"

    # "request completed" at stream end
    completed = [e for e in req_logs if e.get("message") == "request completed"]
    assert len(completed) >= 1
    assert completed[0].get("status") == "success"


def test_agg_404_error(tracing_services) -> None:
    """Aggregated 404: ERROR request completed + ERROR http response sent."""
    port = tracing_services["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, model="nonexistent-model", request_id=rid)
    assert resp.status_code == 404
    time.sleep(1)

    fe_logs = parse_jsonl_logs(read_log_file(tracing_services["frontend"]))
    req_logs = find_logs_by_request_id(fe_logs, rid)

    # "request received"
    received = [e for e in req_logs if e.get("message") == "request received"]
    assert len(received) >= 1

    # "request completed" at ERROR
    completed = [e for e in req_logs if e.get("message") == "request completed"]
    assert len(completed) >= 1
    assert completed[0]["level"] == "ERROR"
    assert completed[0].get("status") == "error"
    assert completed[0].get("error_type") == "not_found"
    assert "error_detail" in completed[0]

    # "http response sent" at ERROR
    http_sent = [e for e in req_logs if e.get("message") == "http response sent"]
    assert len(http_sent) >= 1
    assert http_sent[0]["level"] == "ERROR"
    assert http_sent[0].get("status") == "404"


def test_agg_400_invalid_uuid(tracing_services) -> None:
    """400 invalid UUID: ERROR http response sent, no request received."""
    port = tracing_services["frontend_port"]

    resp = _send_chat_completions(port, request_id="NOT-A-VALID-UUID")
    assert resp.status_code == 400
    body = resp.json()
    assert "must be a valid UUID" in body.get("message", "")
    time.sleep(1)

    fe_logs = parse_jsonl_logs(read_log_file(tracing_services["frontend"]))

    # Find the 400 http response sent
    http_400 = [
        e
        for e in fe_logs
        if e.get("message") == "http response sent" and e.get("status") == "400"
    ]
    assert len(http_400) >= 1
    assert http_400[0]["level"] == "ERROR"

    # No "request received" for invalid UUID
    invalid_received = [
        e
        for e in fe_logs
        if e.get("message") == "request received"
        and "NOT-A-VALID-UUID" in json.dumps(e)
    ]
    assert len(invalid_received) == 0


def test_agg_request_id_propagation(tracing_services) -> None:
    """Frontend and worker share the same trace_id for a request."""
    port = tracing_services["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid, stream=True, max_tokens=20)
    assert resp.status_code == 200
    _ = resp.content
    time.sleep(1)

    fe_logs = parse_jsonl_logs(read_log_file(tracing_services["frontend"]))
    wk_logs = parse_jsonl_logs(read_log_file(tracing_services["worker"]))

    fe_req = find_logs_by_request_id(fe_logs, rid)
    assert len(fe_req) > 0, "Frontend should have logs for this request_id"

    # Get trace_id from frontend
    fe_trace_ids = {e.get("trace_id") for e in fe_req if e.get("trace_id")}
    assert len(fe_trace_ids) == 1, f"Expected single trace_id, got: {fe_trace_ids}"
    trace_id = fe_trace_ids.pop()

    # Worker should have logs with same trace_id
    wk_with_trace = [e for e in wk_logs if e.get("trace_id") == trace_id]
    assert len(wk_with_trace) > 0, f"Worker should have logs with trace_id={trace_id}"

    wk_received = [e for e in wk_with_trace if e.get("message") == "request received"]
    wk_completed = [e for e in wk_with_trace if e.get("message") == "request completed"]
    assert len(wk_received) >= 1
    assert len(wk_completed) >= 1


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
                "x-dynamo-request-id": rid,
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
    except Exception:
        pass

    # Wait for cancellation to propagate
    time.sleep(3)

    fe_logs = parse_jsonl_logs(read_log_file(tracing_services_slow["frontend"]))
    req_logs = find_logs_by_request_id(fe_logs, rid)

    # Should have "request received"
    received = [e for e in req_logs if e.get("message") == "request received"]
    assert len(received) >= 1, f"Expected 'request received', got messages: {[e.get('message') for e in req_logs]}"

    # Should have cancellation — either "request cancelled by client" (WARN from disconnect.rs)
    # or "request completed" with error_type=cancelled (ERROR from InflightGuard)
    cancel_logs = [
        e
        for e in req_logs
        if e.get("message") == "request cancelled by client"
        or (e.get("message") == "request completed" and e.get("error_type") == "cancelled")
    ]
    assert len(cancel_logs) >= 1, (
        f"Expected cancellation log, got messages: {[e.get('message') for e in req_logs]}"
    )


# ---------------------------------------------------------------------------
# Tests — Disaggregated
# ---------------------------------------------------------------------------


def test_disagg_streaming_success(tracing_services_disagg) -> None:
    """Disaggregated streaming: both prefill and decode workers log request lifecycle."""
    port = tracing_services_disagg["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid, stream=True, max_tokens=20)
    assert resp.status_code == 200
    _ = resp.content
    time.sleep(1)

    fe_logs = parse_jsonl_logs(
        read_log_file(tracing_services_disagg["frontend"])
    )
    prefill_logs = parse_jsonl_logs(
        read_log_file(tracing_services_disagg["prefill_worker"])
    )
    decode_logs = parse_jsonl_logs(
        read_log_file(tracing_services_disagg["decode_worker"])
    )

    fe_req = find_logs_by_request_id(fe_logs, rid)
    assert len(fe_req) > 0, "Frontend should have logs for this request"

    # Frontend lifecycle
    received = [e for e in fe_req if e.get("message") == "request received"]
    completed = [e for e in fe_req if e.get("message") == "request completed"]
    assert len(received) >= 1
    assert len(completed) >= 1
    assert completed[0].get("status") == "success"

    # Get trace_id
    fe_trace_ids = {e.get("trace_id") for e in fe_req if e.get("trace_id")}
    assert len(fe_trace_ids) >= 1

    # Prefill worker should have processed the request
    prefill_req = find_logs_by_request_id(prefill_logs, rid)
    prefill_received = [e for e in prefill_req if e.get("message") == "request received"]
    prefill_completed = [e for e in prefill_req if e.get("message") == "request completed"]
    assert len(prefill_received) >= 1, (
        f"Prefill worker should log 'request received'. "
        f"Total prefill log lines: {len(prefill_logs)}, "
        f"prefill request lines: {len(prefill_req)}"
    )
    assert len(prefill_completed) >= 1, "Prefill worker should log 'request completed'"

    # Decode worker should have processed the request
    decode_req = find_logs_by_request_id(decode_logs, rid)
    decode_received = [e for e in decode_req if e.get("message") == "request received"]
    decode_completed = [e for e in decode_req if e.get("message") == "request completed"]
    assert len(decode_received) >= 1, (
        f"Decode worker should log 'request received'. "
        f"Total decode log lines: {len(decode_logs)}, "
        f"decode request lines: {len(decode_req)}"
    )
    assert len(decode_completed) >= 1, "Decode worker should log 'request completed'"


def test_agg_worker_crash(tracing_services_slow) -> None:
    """Kill mocker mid-stream: frontend should log ERROR with internal error."""
    port = tracing_services_slow["frontend_port"]
    worker = tracing_services_slow["worker"]
    rid = str(uuid.uuid4())

    import threading

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
    except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
        pass  # Expected if connection drops

    killer.join(timeout=5)
    time.sleep(2)

    fe_logs = parse_jsonl_logs(read_log_file(tracing_services_slow["frontend"]))
    req_logs = find_logs_by_request_id(fe_logs, rid)

    # Should have "request received"
    received = [e for e in req_logs if e.get("message") == "request received"]
    assert len(received) >= 1

    # Should have an error — either "request completed" with error status
    # or cancellation from the broken connection
    error_logs = [
        e
        for e in req_logs
        if (e.get("message") == "request completed" and e.get("status") == "error")
        or e.get("message") == "request cancelled by client"
    ]
    assert len(error_logs) >= 1, (
        f"Expected error/cancellation log after worker crash, "
        f"got messages: {[e.get('message') for e in req_logs]}"
    )


def test_disagg_unary_success(tracing_services_disagg) -> None:
    """Disaggregated unary: both workers process request, frontend logs full lifecycle."""
    port = tracing_services_disagg["frontend_port"]
    rid = str(uuid.uuid4())

    resp = _send_chat_completions(port, request_id=rid)
    assert resp.status_code == 200
    time.sleep(2)

    fe_logs = parse_jsonl_logs(
        read_log_file(tracing_services_disagg["frontend"])
    )
    prefill_logs = parse_jsonl_logs(
        read_log_file(tracing_services_disagg["prefill_worker"])
    )
    decode_logs = parse_jsonl_logs(
        read_log_file(tracing_services_disagg["decode_worker"])
    )

    fe_req = find_logs_by_request_id(fe_logs, rid)

    received = [e for e in fe_req if e.get("message") == "request received"]
    completed = [e for e in fe_req if e.get("message") == "request completed"]
    http_sent = [e for e in fe_req if e.get("message") == "http response sent"]

    assert len(received) >= 1, "Frontend should log 'request received'"
    assert len(completed) >= 1, "Frontend should log 'request completed'"
    assert completed[0].get("status") == "success"
    assert len(http_sent) >= 1, "Frontend should log 'http response sent'"
    assert http_sent[0].get("status") == "200"

    # Both workers should have request logs
    prefill_req = find_logs_by_request_id(prefill_logs, rid)
    decode_req = find_logs_by_request_id(decode_logs, rid)
    assert len(prefill_req) > 0, (
        f"Prefill worker should have logs for this request. "
        f"Total prefill lines: {len(prefill_logs)}"
    )
    assert len(decode_req) > 0, (
        f"Decode worker should have logs for this request. "
        f"Total decode lines: {len(decode_logs)}"
    )


# ---------------------------------------------------------------------------
# Tests — Disaggregated crash scenarios
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def tracing_services_disagg_slow(
    request, runtime_services_dynamic_ports, dynamo_dynamic_ports, predownload_tokenizers
):
    """Start frontend + slow disaggregated workers for crash testing."""
    ports = dynamo_dynamic_ports

    jsonl_env = {
        "DYN_LOGGING_JSONL": "1",
        "DYN_LOG": "info",
    }

    decode_system_port = allocate_port(8200)

    try:
        with DynamoFrontendProcess(
            request,
            frontend_port=ports.frontend_port,
            terminate_all_matching_process_names=False,
            extra_env=jsonl_env,
        ) as frontend_process:
            with JsonlMockerWorkerProcess(
                request,
                model=TEST_MODEL,
                frontend_port=ports.frontend_port,
                system_port=ports.system_ports[0],
                speedup_ratio=0.1,
                extra_args=["--disaggregation-mode", "prefill"],
                worker_id="prefill-worker",
            ) as prefill_worker:
                with JsonlMockerWorkerProcess(
                    request,
                    model=TEST_MODEL,
                    frontend_port=ports.frontend_port,
                    system_port=decode_system_port,
                    speedup_ratio=0.1,
                    extra_args=["--disaggregation-mode", "decode"],
                    worker_id="decode-worker",
                ) as decode_worker:
                    wait_for_http_completions_ready(
                        frontend_port=ports.frontend_port, model=TEST_MODEL
                    )
                    logger.info("JSONL Disagg slow workers ready")

                    yield {
                        "frontend_port": ports.frontend_port,
                        "frontend": frontend_process,
                        "prefill_worker": prefill_worker,
                        "decode_worker": decode_worker,
                    }
    finally:
        deallocate_port(decode_system_port)


def test_disagg_prefill_crash(tracing_services_disagg_slow) -> None:
    """Kill prefill worker during request with large prompt: frontend should log error."""
    import threading

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
                "x-dynamo-request-id": rid,
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
    except Exception:
        pass

    killer.join(timeout=5)
    time.sleep(3)

    fe_logs = parse_jsonl_logs(
        read_log_file(tracing_services_disagg_slow["frontend"])
    )
    req_logs = find_logs_by_request_id(fe_logs, rid)

    assert len(req_logs) >= 1, f"Frontend should have logs for request {rid}"

    # Expect error — either "request completed" with error status,
    # "request cancelled by client", or "http response sent" with error status
    error_logs = [
        e
        for e in req_logs
        if (e.get("message") == "request completed" and e.get("status") == "error")
        or e.get("message") == "request cancelled by client"
        or (
            e.get("message") == "http response sent"
            and e.get("status") not in ("200", "201")
        )
    ]
    assert len(error_logs) >= 1, (
        f"Expected error log after prefill crash, "
        f"got messages: {[(e.get('message'), e.get('status')) for e in req_logs]}"
    )


def test_disagg_decode_crash(tracing_services_disagg_slow) -> None:
    """Kill decode worker mid-stream: frontend should log error."""
    import threading

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
    except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
        pass

    killer.join(timeout=5)
    time.sleep(3)

    fe_logs = parse_jsonl_logs(
        read_log_file(tracing_services_disagg_slow["frontend"])
    )
    req_logs = find_logs_by_request_id(fe_logs, rid)

    received = [e for e in req_logs if e.get("message") == "request received"]
    assert len(received) >= 1, "Frontend should log 'request received'"

    error_logs = [
        e
        for e in req_logs
        if (e.get("message") == "request completed" and e.get("status") == "error")
        or e.get("message") == "request cancelled by client"
    ]
    assert len(error_logs) >= 1, (
        f"Expected error/cancellation log after decode crash, "
        f"got messages: {[e.get('message') for e in req_logs]}"
    )
