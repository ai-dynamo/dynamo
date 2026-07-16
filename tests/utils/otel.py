# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared OpenTelemetry test collector utilities."""

from __future__ import annotations

import threading
import time


def _get_attribute(attributes, key):
    """Return an OTLP attribute value as a string, or ``None`` if absent."""
    for attr in attributes:
        if attr.key != key:
            continue
        value = attr.value
        if value.HasField("string_value"):
            return value.string_value
        if value.HasField("int_value"):
            return str(value.int_value)
        if value.HasField("double_value"):
            return str(value.double_value)
        if value.HasField("bool_value"):
            return str(value.bool_value).lower()
    return None


def get_span_attribute(span, key):
    """Return a span attribute value as a string, or ``None`` if absent."""
    return _get_attribute(span.attributes, key)


class InProcOtlpCollector:
    """Minimal thread-safe in-process OTLP/gRPC trace collector."""

    def __init__(self):
        self.spans = []
        self._spans_with_service_names = []
        self._lock = threading.Lock()

    def Export(self, request, context):
        from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

        with self._lock:
            for resource_spans in request.resource_spans:
                service_name = _get_attribute(
                    resource_spans.resource.attributes, "service.name"
                )
                for scope_spans in resource_spans.scope_spans:
                    self.spans.extend(scope_spans.spans)
                    self._spans_with_service_names.extend(
                        (service_name, span) for span in scope_spans.spans
                    )
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
            self._spans_with_service_names.clear()

    def snapshot(self):
        """Return a stable copy of the received spans for assertions."""
        with self._lock:
            return list(self.spans)

    def snapshot_with_service_names(self):
        """Return stable ``(service.name, span)`` pairs for assertions."""
        with self._lock:
            return list(self._spans_with_service_names)


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


def get_engine_generate_roles(collector: InProcOtlpCollector) -> set[str]:
    """Return disaggregated roles present on collected engine spans."""
    return {
        role
        for span in collector.engine_generate_spans()
        if (role := get_span_attribute(span, "disagg_role")) is not None
    }


def wait_for_engine_generate_roles(
    collector: InProcOtlpCollector,
    *,
    expected_roles: set[str],
    timeout: float = 30.0,
) -> set[str]:
    """Wait until ``engine.generate`` spans cover all disaggregated roles."""
    deadline = time.monotonic() + timeout
    roles = set()
    while time.monotonic() < deadline:
        roles = get_engine_generate_roles(collector)
        if expected_roles.issubset(roles):
            return roles
        time.sleep(0.5)
    return get_engine_generate_roles(collector)
