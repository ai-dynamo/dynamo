# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
OpenTelemetry tracing header utilities for Dynamo components.
"""


from dynamo._core import Context


def build_trace_headers(context: Context) -> dict[str, str] | None:
    """
    Build trace headers from context for propagation.
    """
    trace_id = context.trace_id
    span_id = context.span_id
    if not trace_id or not span_id:
        return None

    # W3C Trace Context format: {version}-{trace_id}-{parent_id}-{trace_flags}
    # version: 00, trace_flags: 01 (sampled)
    # TODO: properly propagate the trace-flags from current span.
    return {"traceparent": f"00-{trace_id}-{span_id}-01"}
