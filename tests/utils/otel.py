# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OpenTelemetry test collector utilities."""

from __future__ import annotations

import threading
import time
from concurrent import futures

import pytest


class InProcOtlpCollector:
    """Minimal thread-safe in-process OTLP/gRPC trace collector."""

    def __init__(self):
        self.spans = []
        self._lock = threading.Lock()

    def Export(self, request, context):
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

        with self._lock:
            for resource_spans in request.resource_spans:
                for scope_spans in resource_spans.scope_spans:
                    self.spans.extend(scope_spans.spans)
        return trace_service_pb2.ExportTraceServiceResponse()

    def engine_generate_spans(self):
        with self._lock:
            return [span for span in self.spans if span.name == "engine.generate"]

    def spans_for_trace_id(self, trace_id_hex):
        trace_id = bytes.fromhex(trace_id_hex)
        with self._lock:
            return [span for span in self.spans if span.trace_id == trace_id]

    def has_span(self, name):
        with self._lock:
            return any(span.name == name for span in self.spans)

    def clear(self):
        with self._lock:
            self.spans.clear()

    def snapshot(self):
        """Return a stable copy of the received spans for assertions."""
        with self._lock:
            return list(self.spans)


def wait_for_engine_generate_count(
    collector: InProcOtlpCollector,
    *,
    min_count: int,
    timeout: float = 15.0,
) -> int:
    """Wait until the collector receives the requested number of engine spans."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        count = len(collector.engine_generate_spans())
        if count >= min_count:
            return count
        time.sleep(0.5)
    return len(collector.engine_generate_spans())


@pytest.fixture
def otlp_collector():
    """Run an in-process OTLP/gRPC server on an ephemeral port."""
    import grpc
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2_grpc

    collector = InProcOtlpCollector()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    trace_service_pb2_grpc.add_TraceServiceServicer_to_server(collector, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield collector, port
    finally:
        server.stop(grace=1)
