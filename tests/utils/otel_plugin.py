# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest fixtures for OpenTelemetry collector tests."""

from concurrent import futures

import pytest

from tests.utils.otel import InProcOtlpCollector


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
