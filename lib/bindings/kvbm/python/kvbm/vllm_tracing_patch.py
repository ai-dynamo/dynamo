# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Monkey-patches vLLM's Scheduler to emit per-request OTEL spans at key
lifecycle points (scheduled, first_token, finished). Each span is linked
to the request's HTTP trace via trace_headers["traceparent"].

Import this module AFTER vLLM is imported but BEFORE serving starts.
Usage: import kvbm.vllm_tracing_patch  (or call apply_patches())
"""

import logging
import os
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any

logger = logging.getLogger("kvbm.vllm_tracing")

_tracer = None
_patched = False


def _get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    if os.environ.get("OTEL_EXPORT_ENABLED", "").lower() not in ("1", "true", "on", "yes"):
        return None

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource

    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        endpoint = os.environ.get(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://localhost:4317"
        )
        resource = Resource.create({
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "dynamo") + ".vllm",
        })
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("vllm.scheduler", "0.1.0")
    return _tracer


@contextmanager
def _linked_span(name, traceparent, attributes=None):
    """Create a span linked to an existing trace via traceparent string."""
    tracer = _get_tracer()
    assert tracer is not None, f"_linked_span({name}): tracer is None but OTEL_EXPORT_ENABLED is set"
    assert traceparent, f"_linked_span({name}): traceparent is falsy: {traceparent!r}"

    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    ctx = TraceContextTextMapPropagator().extract({"traceparent": traceparent})
    with tracer.start_as_current_span(
        name, context=ctx, attributes=attributes or {}
    ) as span:
        yield span


def _get_request_traceparent(request):
    """Extract traceparent from a vLLM Request's trace_headers."""
    headers = getattr(request, "trace_headers", None)
    if headers:
        return headers.get("traceparent")
    return None


def apply_patches():
    """Apply monkey-patches to vLLM's Scheduler for OTEL tracing."""
    global _patched
    if _patched:
        return
    _patched = True

    if _get_tracer() is None:
        logger.warning("OTEL tracing not enabled (OTEL_EXPORT_ENABLED != 1), skipping vLLM scheduler patches")
        return

    from vllm.v1.core.sched.scheduler import Scheduler

    logger.warning("Applying OTEL tracing patches to vLLM Scheduler")

    _original_schedule = Scheduler.schedule
    _original_update_from_output = Scheduler.update_from_output
    _original_finish_requests = Scheduler.finish_requests

    @wraps(_original_schedule)
    def patched_schedule(self):
        output = _original_schedule(self)

        for new_req in output.scheduled_new_reqs:
            req = self.requests.get(new_req.req_id)
            assert req is not None, (
                f"patched_schedule: req {new_req.req_id} not found in "
                f"self.requests (size={len(self.requests)}, keys={list(self.requests.keys())[:5]})"
            )
            tp = _get_request_traceparent(req)
            if not tp:
                logger.warning(
                    "patched_schedule: req %s has NO traceparent — "
                    "trace_headers=%r, type=%s, has_attr=%s. "
                    "Span will NOT be emitted.",
                    new_req.req_id,
                    getattr(req, "trace_headers", "ATTR_MISSING"),
                    type(getattr(req, "trace_headers", None)),
                    hasattr(req, "trace_headers"),
                )
                continue

            num_scheduled = output.num_scheduled_tokens.get(new_req.req_id, 0)
            with _linked_span("vllm.prefill_scheduled", tp, {
                "request_id": new_req.req_id,
                "num_prompt_tokens": len(new_req.prompt_token_ids or []),
                "num_computed_tokens": new_req.num_computed_tokens,
                "num_scheduled_tokens": num_scheduled,
            }):
                logger.warning(
                    "patched_schedule: EMITTED vllm.prefill_scheduled for %s (tp=%s...)",
                    new_req.req_id, tp[:30],
                )

        return output

    @wraps(_original_update_from_output)
    def patched_update_from_output(self, scheduler_output, model_runner_output):
        result = _original_update_from_output(self, scheduler_output, model_runner_output)

        for req_id in scheduler_output.num_scheduled_tokens:
            req = self.requests.get(req_id)
            if req is None:
                continue
            tp = _get_request_traceparent(req)
            if not tp:
                continue

            num_output = getattr(req, "num_output_tokens", 0)
            if num_output == 1:
                ttft = time.time() - req.arrival_time if hasattr(req, "arrival_time") else 0
                with _linked_span("vllm.first_token", tp, {
                    "request_id": req_id,
                    "ttft_ms": round(ttft * 1000, 2),
                    "num_prompt_tokens": len(getattr(req, "prompt_token_ids", []) or []),
                    "num_computed_tokens": getattr(req, "num_computed_tokens", 0),
                }):
                    logger.warning(
                        "patched_update_from_output: EMITTED vllm.first_token for %s ttft=%.1fms",
                        req_id, ttft * 1000,
                    )

            if req.is_finished():
                e2e = time.time() - req.arrival_time if hasattr(req, "arrival_time") else 0
                with _linked_span("vllm.finished", tp, {
                    "request_id": req_id,
                    "e2e_ms": round(e2e * 1000, 2),
                    "num_output_tokens": num_output,
                    "num_prompt_tokens": len(getattr(req, "prompt_token_ids", []) or []),
                    "finish_reason": str(getattr(req, "status", "")),
                }):
                    logger.warning(
                        "patched_update_from_output: EMITTED vllm.finished for %s e2e=%.1fms",
                        req_id, e2e * 1000,
                    )

        return result

    @wraps(_original_finish_requests)
    def patched_finish_requests(self, request_ids, finished_status):
        if isinstance(request_ids, str):
            ids = [request_ids]
        else:
            ids = list(request_ids)

        for req_id in ids:
            req = self.requests.get(req_id)
            if req is None:
                continue
            tp = _get_request_traceparent(req)
            if tp:
                e2e = time.time() - req.arrival_time if hasattr(req, "arrival_time") else 0
                with _linked_span("vllm.aborted", tp, {
                    "request_id": req_id,
                    "e2e_ms": round(e2e * 1000, 2),
                    "finish_status": str(finished_status),
                }):
                    logger.warning("patched_finish_requests: EMITTED vllm.aborted for %s", req_id)

        return _original_finish_requests(self, request_ids, finished_status)

    Scheduler.schedule = patched_schedule
    Scheduler.update_from_output = patched_update_from_output
    Scheduler.finish_requests = patched_finish_requests

    logger.warning("vLLM Scheduler OTEL tracing patches applied successfully")


# Auto-apply when imported
apply_patches()
