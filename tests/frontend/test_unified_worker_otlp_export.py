# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression test for unified-backend workers' OTLP exporter pipeline.

Where the JSONL smoke test only asserts the tracing subscriber
installed, this test asserts spans actually travel over OTLP/gRPC to a
collector — the strongest signal that the export pipeline is wired
end-to-end. Boots an in-process gRPC collector, runs a sample worker,
curls the frontend, and asserts the `engine.generate` span arrived
with its attributes intact.
"""

from __future__ import annotations

import threading
import time
from concurrent import futures

import grpc
import pytest
from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)

from tests.frontend.conftest import (
    SampleUnifiedWorkerProcess,
    wait_for_http_completions_ready,
)
from tests.frontend.test_request_tracing_logs import _send_chat_completions
from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.parallel,
    pytest.mark.model(TEST_MODEL),
    pytest.mark.timeout(180),
]


class InProcOtlpCollector(trace_service_pb2_grpc.TraceServiceServicer):
    """Minimal in-process OTLP/gRPC trace collector.

    Stores every received Span proto for the test to assert on. Thread-safe;
    the gRPC server runs on a worker thread pool.
    """

    def __init__(self):
        self.spans = []
        self._lock = threading.Lock()

    def Export(self, request, context):
        with self._lock:
            for resource_spans in request.resource_spans:
                for scope_spans in resource_spans.scope_spans:
                    self.spans.extend(scope_spans.spans)
        return trace_service_pb2.ExportTraceServiceResponse()

    def engine_generate_spans(self):
        with self._lock:
            return [s for s in self.spans if s.name == "engine.generate"]

    def has_span(self, name):
        with self._lock:
            return any(s.name == name for s in self.spans)

    def snapshot(self):
        """Return a stable copy of `self.spans` for assertions."""
        with self._lock:
            return list(self.spans)


def _get_attr(span, key):
    """Return the attribute value as a string, or None if absent.
    Int/double values are stringified via `str()`."""
    for attr in span.attributes:
        if attr.key == key:
            v = attr.value
            if v.HasField("string_value"):
                return v.string_value
            if v.HasField("int_value"):
                return str(v.int_value)
            if v.HasField("double_value"):
                return str(v.double_value)
    return None


@pytest.fixture
def otlp_collector():
    """Spin up an in-process OTLP gRPC server on a random port. Yields
    (collector, port). Cleans up on test exit.
    """
    collector = InProcOtlpCollector()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(collector, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield collector, port
    finally:
        server.stop(grace=1)


def test_unified_worker_exports_engine_generate_span_over_otlp(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
    otlp_collector,
):
    """Aggregated unified worker must export the `engine.generate` span
    over OTLP to the collector — proves the full export pipeline works,
    not just the subscriber install.
    """
    collector, otlp_port = otlp_collector

    # Only the traces endpoint is wired to our collector. The default
    # logs endpoint is left at localhost:4317; if nothing's listening,
    # the logs batch processor drops silently (no extra noise in the
    # worker log).
    otel_env = {
        "OTEL_EXPORT_ENABLED": "1",
        "DYN_LOGGING_JSONL": "1",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"http://127.0.0.1:{otlp_port}",
        "OTEL_SERVICE_NAME": "dynamo-unified-worker-test",
    }

    ports = dynamo_dynamic_ports
    frontend_port = ports.frontend_port
    system_port = ports.system_ports[0]

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env=otel_env,
        terminate_all_matching_process_names=False,
    ):
        with SampleUnifiedWorkerProcess(
            request,
            frontend_port=frontend_port,
            system_port=system_port,
            model_name=TEST_MODEL,
            component="sample",
            disaggregation_mode="agg",
            extra_env=otel_env,
            worker_id="sample-agg-otlp",
        ):
            wait_for_http_completions_ready(
                frontend_port=frontend_port, model=TEST_MODEL
            )

            resp = _send_chat_completions(frontend_port, model=TEST_MODEL, max_tokens=5)
            assert (
                resp.status_code == 200
            ), f"curl failed: {resp.status_code} {resp.text!r}"

            # Poll until the batch exporter flushes (~5s default delay).
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if collector.engine_generate_spans():
                    break
                time.sleep(0.5)

    eg_spans = collector.engine_generate_spans()
    assert eg_spans, (
        "OTLP collector received zero `engine.generate` spans. The worker "
        "either failed to install the tracing subscriber or the OTLP "
        "exporter is not wired. Check lib/bindings/python/rust/backend.rs."
    )

    # Verify auto-span attributes round-tripped through OTLP.
    span = eg_spans[0]
    assert (
        _get_attr(span, "disagg_role") == "agg"
    ), f"expected disagg_role=agg, got {_get_attr(span, 'disagg_role')!r}"
    assert _get_attr(span, "model") is not None, "missing `model` attribute"
    assert (
        _get_attr(span, "input_tokens") is not None
    ), "missing `input_tokens` attribute"


@pytest.mark.parametrize("num_system_ports", [2], indirect=True)
def test_disagg_decode_span_links_to_prefill_span(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
    otlp_collector,
):
    """Disaggregated mode: the decode-side `engine.generate` span must
    carry an OTel Link pointing at the prefill-side span. This regression-
    tests the typed `prefill_trace_link` round-trip:
        prefill EngineAdapter writes `chunk.prefill_trace_link`
        → runtime carries it across to decode via `PrefillResult`
        → decode EngineAdapter reads it and calls `add_link(...)`.
    """
    collector, otlp_port = otlp_collector

    otel_env = {
        "OTEL_EXPORT_ENABLED": "1",
        "DYN_LOGGING_JSONL": "1",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"http://127.0.0.1:{otlp_port}",
        "OTEL_SERVICE_NAME": "dynamo-unified-disagg-test",
    }

    ports = dynamo_dynamic_ports
    frontend_port = ports.frontend_port
    prefill_system_port, decode_system_port = (
        ports.system_ports[0],
        ports.system_ports[1],
    )

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env=otel_env,
        terminate_all_matching_process_names=False,
    ):
        with SampleUnifiedWorkerProcess(
            request,
            frontend_port=frontend_port,
            system_port=prefill_system_port,
            model_name=TEST_MODEL,
            component="sample-prefill",
            disaggregation_mode="prefill",
            extra_env=otel_env,
            worker_id="sample-prefill",
        ):
            with SampleUnifiedWorkerProcess(
                request,
                frontend_port=frontend_port,
                system_port=decode_system_port,
                model_name=TEST_MODEL,
                component="sample-decode",
                disaggregation_mode="decode",
                extra_env=otel_env,
                worker_id="sample-decode",
            ):
                wait_for_http_completions_ready(
                    frontend_port=frontend_port, model=TEST_MODEL
                )

                resp = _send_chat_completions(
                    frontend_port, model=TEST_MODEL, max_tokens=5
                )
                assert (
                    resp.status_code == 200
                ), f"curl failed: {resp.status_code} {resp.text!r}"

                # Wait for prefill/decode engine.generate AND at least one
                # `sample.tokens` child span — the child is exported in a
                # separate batch and can lag the parent.
                deadline = time.monotonic() + 30.0
                while time.monotonic() < deadline:
                    roles = {
                        _get_attr(s, "disagg_role")
                        for s in collector.engine_generate_spans()
                    }
                    if {"prefill", "decode"}.issubset(roles) and collector.has_span(
                        "sample.tokens"
                    ):
                        break
                    time.sleep(0.5)

    eg_spans = collector.engine_generate_spans()
    # Single curl ⇒ at most one span per role; if there were retries the
    # last would win, which is fine for this regression test.
    by_role = {_get_attr(s, "disagg_role"): s for s in eg_spans}
    assert (
        "prefill" in by_role
    ), f"no prefill engine.generate span; got roles {set(by_role)}"
    assert (
        "decode" in by_role
    ), f"no decode engine.generate span; got roles {set(by_role)}"

    prefill_span = by_role["prefill"]
    decode_span = by_role["decode"]

    assert decode_span.links, (
        "decode-side engine.generate span has no Links — the typed "
        "`prefill_trace_link` round-trip is broken. Check EngineAdapter "
        "decode-read at lib/backend-common/src/adapter.rs."
    )
    link_span_ids = {link.span_id for link in decode_span.links}
    assert prefill_span.span_id in link_span_ids, (
        f"decode Link span_ids {[link.span_id.hex() for link in decode_span.links]} "
        f"don't include prefill span_id {prefill_span.span_id.hex()}"
    )

    # Engine-author child spans MUST nest under engine.generate, not be
    # siblings. The sample engine opens `sample.tokens` via
    # telemetry.start_span() — its parent_span_id must equal the decode
    # engine.generate span_id.
    sample_token_spans = [
        s
        for s in collector.snapshot()
        if s.name == "sample.tokens"
        and s.trace_id == decode_span.trace_id
        and s.parent_span_id == decode_span.span_id
    ]
    assert sample_token_spans, (
        "no `sample.tokens` child span nesting under decode engine.generate "
        f"(decode trace_id={decode_span.trace_id.hex()} "
        f"span_id={decode_span.span_id.hex()}). Likely causes: "
        "OTel global TracerProvider not registered "
        "(see lib/runtime/src/logging.rs), or Context.start_span hit the "
        "NoOp path (bridge not installed)."
    )
