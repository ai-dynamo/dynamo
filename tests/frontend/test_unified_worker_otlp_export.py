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
import requests
from opentelemetry.proto.collector.trace.v1 import (
    trace_service_pb2,
    trace_service_pb2_grpc,
)
from opentelemetry.proto.trace.v1 import trace_pb2

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

    def spans_for_trace_id(self, trace_id_hex):
        trace_id = bytes.fromhex(trace_id_hex)
        with self._lock:
            return [s for s in self.spans if s.trace_id == trace_id]

    def has_span(self, name):
        with self._lock:
            return any(s.name == name for s in self.spans)

    def clear(self):
        with self._lock:
            self.spans.clear()

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
            if v.HasField("bool_value"):
                return str(v.bool_value).lower()
    return None


def _send_chat_completions_with_headers(
    port: int,
    *,
    headers: dict[str, str],
    model: str = TEST_MODEL,
    max_tokens: int = 5,
) -> requests.Response:
    request_headers = {"Content-Type": "application/json", **headers}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": max_tokens,
    }
    return requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        headers=request_headers,
        json=payload,
        timeout=60,
    )


def _wait_for_engine_generate_count(
    collector: InProcOtlpCollector,
    *,
    min_count: int,
    timeout: float = 15.0,
) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        count = len(collector.engine_generate_spans())
        if count >= min_count:
            return count
        time.sleep(0.5)
    return len(collector.engine_generate_spans())


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
        "DYN_LOG": "warn",
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

            # Poll until both worker and full-lifetime route spans flush
            # (~5s default batch delay).
            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                spans = collector.snapshot()
                if any(s.name == "engine.generate" for s in spans) and any(
                    s.name == "router.route_request" for s in spans
                ):
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

    same_trace = [s for s in collector.snapshot() if s.trace_id == span.trace_id]
    route_spans = [s for s in same_trace if s.name == "router.route_request"]
    assert route_spans, "missing frontend `router.route_request` span"
    route_span = route_spans[0]
    assert route_span.kind == trace_pb2.Span.SPAN_KIND_CLIENT
    assert _get_attr(route_span, "request.attempt") == "0"
    assert _get_attr(route_span, "migration.is_retry") == "false"
    assert _get_attr(route_span, "request.outcome") == "success"

    worker_spans = [
        s
        for s in same_trace
        if s.name == "handle_payload" and s.parent_span_id == route_span.span_id
    ]
    assert (
        worker_spans
    ), "worker `handle_payload` must be a remote child of the frontend route span"
    worker_span = worker_spans[0]
    assert worker_span.kind == trace_pb2.Span.SPAN_KIND_SERVER
    assert span.parent_span_id == worker_span.span_id
    assert (
        route_span.end_time_unix_nano >= worker_span.end_time_unix_nano
    ), "route span ended before worker request handling completed"
    assert (
        route_span.end_time_unix_nano >= span.end_time_unix_nano
    ), "route span ended before worker generation completed"


def test_unsampled_traceparent_does_not_export_spans_over_otlp(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
    otlp_collector,
):
    collector, otlp_port = otlp_collector
    trace_id = "11111111111111111111111111111111"
    traceparent = f"00-{trace_id}-2222222222222222-00"

    otel_env = {
        "OTEL_EXPORT_ENABLED": "1",
        "DYN_LOGGING_JSONL": "1",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"http://127.0.0.1:{otlp_port}",
        "OTEL_SERVICE_NAME": "dynamo-unified-worker-unsampled-test",
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
            worker_id="sample-agg-otlp-unsampled",
        ):
            wait_for_http_completions_ready(
                frontend_port=frontend_port, model=TEST_MODEL
            )
            collector.clear()

            resp = _send_chat_completions_with_headers(
                frontend_port,
                headers={"traceparent": traceparent},
                model=TEST_MODEL,
                max_tokens=5,
            )
            assert (
                resp.status_code == 200
            ), f"curl failed: {resp.status_code} {resp.text!r}"

            deadline = time.monotonic() + 15.0
            while time.monotonic() < deadline:
                if collector.spans_for_trace_id(trace_id):
                    break
                time.sleep(0.5)

    spans = collector.spans_for_trace_id(trace_id)
    assert not spans, (
        "unsampled traceparent exported spans: " f"{[span.name for span in spans]}"
    )


@pytest.mark.parametrize(
    ("sampler_arg", "request_count", "expected_min", "expected_max"),
    [
        ("0", 20, 0, 0),
        ("0.1", 200, 5, 45),
        ("1", 20, 20, None),
    ],
    ids=["ratio-0", "ratio-0.1", "ratio-1"],
)
def test_traceidratio_sampler_controls_otlp_exports(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports,
    predownload_tokenizers,
    otlp_collector,
    sampler_arg,
    request_count,
    expected_min,
    expected_max,
):
    collector, otlp_port = otlp_collector

    otel_env = {
        "OTEL_EXPORT_ENABLED": "1",
        "DYN_LOGGING_JSONL": "1",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": f"http://127.0.0.1:{otlp_port}",
        "OTEL_SERVICE_NAME": f"dynamo-unified-worker-sampler-{sampler_arg}",
        "OTEL_TRACES_SAMPLER": "parentbased_traceidratio",
        "OTEL_TRACES_SAMPLER_ARG": sampler_arg,
        "OTEL_BSP_SCHEDULE_DELAY": "1000",
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
            worker_id=f"sample-agg-otlp-sampler-{sampler_arg}",
        ):
            wait_for_http_completions_ready(
                frontend_port=frontend_port, model=TEST_MODEL
            )
            collector.clear()

            for _ in range(request_count):
                resp = _send_chat_completions(
                    frontend_port, model=TEST_MODEL, max_tokens=1
                )
                assert (
                    resp.status_code == 200
                ), f"curl failed: {resp.status_code} {resp.text!r}"

            count = _wait_for_engine_generate_count(
                collector,
                min_count=expected_min if expected_max is None else expected_max + 1,
            )

    assert count >= expected_min
    if expected_max is not None:
        assert count <= expected_max


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
    tests the typed `worker_trace_link` round-trip:
        prefill EngineAdapter writes `chunk.worker_trace_link`
        → PrefillRouter copies it onto `PreprocessedRequest.migration_link`
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
        "`worker_trace_link` round-trip is broken. Check EngineAdapter "
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
