# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""E2E tracing tests for the engine-native preprocessor (chat processor) path.

Parallel to ``test_request_tracing_logs.py``, which covers the default Rust
preprocessor. This file covers the ``--dyn-chat-processor vllm`` path, where the
engine's own (vLLM-native) preprocessor runs in the frontend and re-enters Rust
through PyO3 (topology 2). Under that topology the ``KvPushRouter`` runs
in-process in the frontend, so ``kv_router.route_request`` lands in the
frontend's own JSONL log.

The distinct risk here is the Python -> Rust boundary: tracing "current span"
is task-local, and a fresh pyo3 tokio task does not inherit it, so the
downstream ``route_request`` span can be minted on a *new* trace_id, silently
disconnecting the router spans from the request's trace. ``RoutedEngine`` guards
against this by rebuilding a span from the carried context and instrumenting the
re-entry future (``dispatch_span`` in ``routed_engine.rs``). This test is the
regression guard for that reconnection: it sends a request carrying a W3C
``traceparent`` and asserts the inbound trace_id is preserved through
``route_request`` — the exact continuity the default-preprocessor path gets for
free (all-Rust, same async task) and is therefore not exercised by
``test_request_tracing_logs.py``.
"""

from __future__ import annotations

import logging
import time
import uuid

import pytest
import requests

from tests.frontend.test_request_tracing_logs import (
    JSONL_ENV,
    TEST_MODEL,
    parse_jsonl_logs,
    read_log_file,
)
from tests.frontend.test_vllm_prepost_integration import MockVllmPrepostWorkerProcess
from tests.utils.managed_process import DynamoFrontendProcess

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.vllm,
    # gpu_1 not gpu_0: --dyn-chat-processor vllm imports vLLM in the frontend,
    # and vLLM DeviceConfig(device='auto') fails on CPU-only arm64 runners even
    # for this mock-worker test (matches test_vllm_prepost_integration.py).
    pytest.mark.gpu_1,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.e2e,
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
    pytest.mark.timeout(300),
]

# A fixed, sampled W3C traceparent so the assertion pins an exact inbound trace.
INBOUND_TRACE_ID = "0af7651916cd43dd8448eb211c80319c"
INBOUND_PARENT_SPAN = "b7ad6b7169203331"
INBOUND_TRACEPARENT = f"00-{INBOUND_TRACE_ID}-{INBOUND_PARENT_SPAN}-01"


def test_native_preprocessor_route_request_inherits_trace(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
    tmp_path,
) -> None:
    """route_request must stay on the inbound trace across the Python/Rust boundary.

    Frontend runs the vLLM-native chat processor (topology 2) so the KV router —
    and its ``kv_router.route_request`` span — execute in-process in the frontend.
    We send a request with an inbound W3C ``traceparent`` and assert both the
    frontend ``http-request`` span and the downstream ``kv_router.route_request``
    span carry that same trace_id (no fresh, disconnected trace minted at the
    pyo3 re-entry).
    """
    _ = runtime_services_dynamic_ports
    ports = dynamo_dynamic_ports
    capture_path = tmp_path / "captured_request.json"
    rid = str(uuid.uuid4())

    with DynamoFrontendProcess(
        request,
        frontend_port=ports.frontend_port,
        router_mode="kv",  # route_request span lives on the KV path
        extra_args=[
            "--dyn-chat-processor",
            "vllm",
            "--discovery-backend",
            "etcd",
            "--request-plane",
            "tcp",
        ],
        extra_env=JSONL_ENV,
        terminate_all_matching_process_names=False,
    ) as frontend:
        with MockVllmPrepostWorkerProcess(
            request,
            frontend_port=ports.frontend_port,
            capture_path=capture_path,
        ):
            resp = requests.post(
                f"http://localhost:{ports.frontend_port}/v1/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "x-request-id": rid,
                    "traceparent": INBOUND_TRACEPARENT,
                },
                json={
                    "model": TEST_MODEL,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 5,
                    "stream": False,
                },
                timeout=60,
            )
            assert resp.status_code == 200, resp.text
            # Let the trailing spans/logs flush before the frontend is torn down.
            time.sleep(1)

    # Under topology 2 every span for this request runs in-process in the
    # frontend, so its JSONL log holds both http-request and route_request.
    logs = parse_jsonl_logs(read_log_file(frontend))

    # UNFILTERED on purpose: do NOT pre-filter by rid/trace_id. A disconnected
    # route_request span would be on a *different* trace_id (and carry no
    # x_request_id), so filtering first would hide the very bug we guard against.
    span_traces = sorted(
        {(e.get("span_name"), e.get("trace_id")) for e in logs if e.get("span_name")}
    )
    all_span_names = sorted({e.get("span_name") for e in logs if e.get("span_name")})
    logger.info("frontend span_names: %s", all_span_names)
    logger.info("frontend (span_name, trace_id) pairs: %s", span_traces)

    fe_traces = {t for (s, t) in span_traces if s == "http-request" and t}
    route_traces = {t for (s, t) in span_traces if s and "route_request" in s and t}

    assert fe_traces == {INBOUND_TRACE_ID}, (
        f"frontend http-request span should carry the inbound trace; got {fe_traces} "
        f"(full map: {span_traces})"
    )
    assert route_traces, (
        "no kv_router.route_request span found in frontend logs — is the vLLM chat "
        f"processor active and router_mode=kv? (full map: {span_traces})"
    )
    assert route_traces == {INBOUND_TRACE_ID}, (
        "DISCONNECT: kv_router.route_request is on a different trace than the frontend "
        f"({route_traces} != {{{INBOUND_TRACE_ID}}}) — context lost across the "
        f"Python/Rust boundary (full map: {span_traces})"
    )
