# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-author telemetry facade.

Thin wrapper around the PyO3 ``Context`` methods so engine code adds
attributes and events without importing ``opentelemetry`` directly.
Records land on the framework's ``engine.generate`` span (created by
``EngineAdapter`` in ``lib/backend-common``); when no span is plumbed
in (Python-instantiated test contexts), the calls are silent no-ops.

Span attributes must be **declared** on the ``engine.generate`` span to
take effect — the canonical set is ``model``, ``input_tokens``,
``output_tokens``, ``ttft_ms``, ``finish_reason``, ``cancelled``,
``disagg_role``, ``error_kind``. Unknown attribute names are dropped
by ``tracing::Span::record`` (matching the underlying Rust API).
Events have no such restriction.
"""

from __future__ import annotations

from typing import Any

from dynamo._core import Context


def record(context: Context, **attrs: Any) -> None:
    """Record one or more attributes on the current ``engine.generate`` span.

    Example::

        telemetry.record(context, kv_cache_hit_blocks=8, scheduler_wait_ms=12.3)
    """
    for key, value in attrs.items():
        context.record_attribute(key, value)


def event(context: Context, name: str, **attrs: Any) -> None:
    """Emit a span event with optional structured attributes.

    Example::

        telemetry.event(context, "nixl_transfer_complete", bytes=1048576)
    """
    context.record_event(name, dict(attrs) if attrs else None)
