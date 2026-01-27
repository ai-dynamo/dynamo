# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
OpenTelemetry tracing utilities for the streaming ASR pipeline.

Environment variables:
    OTEL_EXPORT_ENABLED: Set to "true" to enable trace export (default: false)
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: OTLP gRPC endpoint (default: http://localhost:4317)
    OTEL_SERVICE_NAME: Service name for this component (default: streaming-asr)

Usage:
    from tracing import setup_tracing, get_tracer

    # Call once at startup
    setup_tracing("my-service-name")

    # Get a tracer for your module
    tracer = get_tracer(__name__)

    # Create spans
    with tracer.start_as_current_span("operation_name") as span:
        span.set_attribute("key", "value")
        # ... do work ...
"""

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Global flag to track if tracing is enabled
_tracing_enabled = False
_tracer_provider = None


def is_tracing_enabled() -> bool:
    """Check if OTEL tracing export is enabled."""
    return os.environ.get("OTEL_EXPORT_ENABLED", "false").lower() == "true"


def setup_tracing(service_name: Optional[str] = None) -> bool:
    """
    Initialize OpenTelemetry tracing if enabled.

    Args:
        service_name: Override for OTEL_SERVICE_NAME env var

    Returns:
        True if tracing was enabled and configured, False otherwise
    """
    global _tracing_enabled, _tracer_provider

    if not is_tracing_enabled():
        logger.info("OTEL tracing disabled (set OTEL_EXPORT_ENABLED=true to enable)")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "OpenTelemetry packages not installed. "
            "Install with: pip install opentelemetry-api opentelemetry-sdk "
            "opentelemetry-exporter-otlp-proto-grpc"
        )
        return False

    # Get service name from param or env
    svc_name = service_name or os.environ.get("OTEL_SERVICE_NAME", "streaming-asr")

    # Get OTLP endpoint
    otlp_endpoint = os.environ.get(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317"
    )

    # Create resource with service name
    resource = Resource(attributes={SERVICE_NAME: svc_name})

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Create OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)

    # Add batch processor for efficient export
    span_processor = BatchSpanProcessor(otlp_exporter)
    _tracer_provider.add_span_processor(span_processor)

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    _tracing_enabled = True
    logger.info(f"OTEL tracing enabled: service={svc_name}, endpoint={otlp_endpoint}")
    return True


def get_tracer(name: str):
    """
    Get a tracer instance for the given module name.

    If tracing is not enabled, returns a NoOpTracer.
    """
    if _tracing_enabled:
        from opentelemetry import trace

        return trace.get_tracer(name)
    else:
        return NoOpTracer()


class NoOpSpan:
    """No-op span that does nothing when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: dict) -> None:
        pass

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def set_status(self, status: Any) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class NoOpTracer:
    """No-op tracer that returns no-op spans when tracing is disabled."""

    def start_as_current_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()

    def start_span(self, name: str, **kwargs) -> NoOpSpan:
        return NoOpSpan()


def shutdown_tracing():
    """Shutdown the tracer provider and flush any pending spans."""
    global _tracer_provider
    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        logger.info("OTEL tracing shutdown complete")
