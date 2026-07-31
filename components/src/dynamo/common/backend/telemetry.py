# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-author telemetry facade.

OTel-shaped surface for adding attributes, events, and child spans to the
framework's ``engine.generate`` span. Both functions return a unified
``SpanProxy`` whose ``set_attribute`` / ``add_event`` / ``set_status``
mirror the OpenTelemetry ``Span`` API — no Dynamo-specific vocabulary to
learn.

Example::

    from dynamo.common.backend import telemetry

    span = telemetry.current_span(context)
    span.set_attribute("ttft_ms", 42.0)
    span.add_event("first_token", token_id=15339)

    with telemetry.start_span(context, "kv_load") as child:
        child.set_attribute("blocks", 8)

When no parent span is plumbed in (Python-instantiated test contexts) or
the OTel bridge isn't installed, the returned ``SpanProxy`` is a silent
no-op — calls do not raise. A once-per-process WARN log fires the first
time a no-op is exercised so the misconfiguration is discoverable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dynamo._core import Context

if TYPE_CHECKING:
    from dynamo._core import SpanProxy


def current_span(context: Context) -> "SpanProxy":
    """Return a handle on the framework's ``engine.generate`` span."""
    return context.current_span()


def start_span(context: Context, name: str, **attrs: Any) -> "SpanProxy":
    """Open a child span under ``engine.generate`` with the given dynamic
    name. The returned ``SpanProxy`` is a context manager — use ``with``
    so the span ends on exit.

    Example::

        with telemetry.start_span(context, "tokenize", batch_size=8) as s:
            tokens = tokenizer.encode(prompt)
            s.add_event("encoder_warmup_complete")
    """
    return context.start_span(name, dict(attrs) if attrs else None)


def trace_headers(context: Context) -> dict[str, str] | None:
    """W3C trace headers to forward to the underlying inference engine.

    Thin wrapper over ``Context.trace_headers()``; see that method's
    docstring for return semantics. Most engine code should prefer
    :func:`engine_trace_kwargs` instead — it builds the splat-ready kwargs
    dict with the right name and gate per backend.
    """
    return context.trace_headers()


def engine_trace_kwargs(
    context: Context,
    *,
    kwarg_name: str = "trace_headers",
    enabled: bool = True,
) -> dict[str, dict[str, str]]:
    """Splat-ready ``{kwarg_name: headers}`` for the inference-engine call,
    or ``{}`` when no traceparent / ``enabled=False``. See CLAUDE.md for
    per-backend kwarg names + gates."""
    if not enabled:
        return {}
    headers = context.trace_headers()
    if headers is None:
        return {}
    return {kwarg_name: headers}
