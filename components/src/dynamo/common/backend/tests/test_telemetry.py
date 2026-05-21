# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the engine-author telemetry facade.

The PyO3 ``Context`` / ``SpanProxy`` are exercised indirectly: we duck-type
fakes that record what the facade called, then assert the facade routes to
the right method with the right args.

Test doubles are structurally validated against ``Protocol`` definitions
so they stay in sync with the real PyO3 surface. If ``Context.current_span``
or ``SpanProxy.set_attribute`` ever changes signature, ``isinstance``
checks here will fail at test setup — catching drift before silent
misbehavior in CI.
"""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import pytest

from dynamo.common.backend import telemetry

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@runtime_checkable
class _SpanProxyProtocol(Protocol):
    """Structural contract for ``dynamo._core.SpanProxy``. The fake must
    match — if the real proxy adds or renames a method engines call, update
    this protocol AND the fake."""

    def set_attribute(self, key: str, value: Any) -> None:
        ...

    def add_event(self, name: str, attrs: Optional[dict] = None) -> None:
        ...

    def set_status(self, status: str, description: Optional[str] = None) -> None:
        ...

    def close(self) -> None:
        ...

    def __enter__(self) -> "_SpanProxyProtocol":
        ...

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        ...


@runtime_checkable
class _ContextProtocol(Protocol):
    """Structural contract for the telemetry-relevant slice of
    ``dynamo._core.Context``. Cancellation / identity methods are not part
    of this protocol — they're on Context but not used by the facade."""

    def current_span(self) -> _SpanProxyProtocol:
        ...

    def start_span(self, name: str, attrs: Optional[dict] = None) -> _SpanProxyProtocol:
        ...

    def trace_headers(self) -> Optional[dict]:
        ...


class _FakeSpan:
    def __init__(self):
        self.attrs: list[tuple[str, Any]] = []
        self.events: list[tuple[str, Optional[dict]]] = []
        self.statuses: list[tuple[str, Optional[str]]] = []
        self.closed = False

    def set_attribute(self, key, value):
        self.attrs.append((key, value))

    def add_event(self, name, attrs=None):
        self.events.append((name, attrs))

    def set_status(self, status, description=None):
        self.statuses.append((status, description))

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class _FakeContext:
    def __init__(self, trace_headers_value: Optional[dict] = None):
        self._auto_span = _FakeSpan()
        self._child_spans: list[tuple[str, Optional[dict], _FakeSpan]] = []
        self._trace_headers_value = trace_headers_value

    def current_span(self):
        return self._auto_span

    def start_span(self, name, attrs=None):
        child = _FakeSpan()
        self._child_spans.append((name, attrs, child))
        return child

    def trace_headers(self):
        return self._trace_headers_value


def test_fakes_match_pyo3_protocols():
    """Catches drift between the test doubles and the real PyO3 surface.
    If this fails, either the protocol or the fake needs updating."""
    assert isinstance(_FakeSpan(), _SpanProxyProtocol)
    assert isinstance(_FakeContext(), _ContextProtocol)


def test_current_span_returns_auto_span():
    ctx = _FakeContext()
    span = telemetry.current_span(ctx)
    span.set_attribute("ttft_ms", 42.0)
    span.add_event("first_token", {"token_id": 15339})
    span.set_status("ok")
    assert ctx._auto_span.attrs == [("ttft_ms", 42.0)]
    assert ctx._auto_span.events == [("first_token", {"token_id": 15339})]
    assert ctx._auto_span.statuses == [("ok", None)]


def test_start_span_passes_attrs_dict_or_none():
    ctx = _FakeContext()
    with telemetry.start_span(ctx, "kv_load", blocks=8) as s:
        s.set_attribute("rank", 0)
    with telemetry.start_span(ctx, "phase") as _:
        pass
    names_and_attrs = [(n, a) for n, a, _ in ctx._child_spans]
    assert names_and_attrs == [
        ("kv_load", {"blocks": 8}),
        ("phase", None),
    ]


def test_start_span_closes_on_exit():
    ctx = _FakeContext()
    with telemetry.start_span(ctx, "phase") as s:
        s.set_attribute("k", "v")
    _, _, child = ctx._child_spans[-1]
    assert child.attrs == [("k", "v")]
    assert child.closed is True


def test_start_span_closes_on_exception():
    ctx = _FakeContext()
    with pytest.raises(RuntimeError):
        with telemetry.start_span(ctx, "phase"):
            raise RuntimeError("boom")
    _, _, child = ctx._child_spans[-1]
    assert child.closed is True


def test_set_status_accepts_error_with_description():
    ctx = _FakeContext()
    span = telemetry.current_span(ctx)
    span.set_status("error", "kv_transfer_timeout")
    assert ctx._auto_span.statuses == [("error", "kv_transfer_timeout")]


def test_trace_headers_delegates_to_context():
    headers = {"traceparent": "00-1234-5678-01"}
    ctx = _FakeContext(trace_headers_value=headers)
    assert telemetry.trace_headers(ctx) is headers
    assert telemetry.trace_headers(_FakeContext()) is None


def test_engine_trace_kwargs_default_kwarg_name():
    headers = {"traceparent": "00-1234-5678-01"}
    ctx = _FakeContext(trace_headers_value=headers)
    assert telemetry.engine_trace_kwargs(ctx) == {"trace_headers": headers}


def test_engine_trace_kwargs_custom_kwarg_name_for_sglang():
    headers = {"traceparent": "00-1234-5678-01"}
    ctx = _FakeContext(trace_headers_value=headers)
    assert telemetry.engine_trace_kwargs(ctx, kwarg_name="external_trace_header") == {
        "external_trace_header": headers
    }


def test_engine_trace_kwargs_returns_empty_when_disabled_or_no_headers():
    # SGLang's --enable-trace=False gate: omit the kwarg entirely.
    assert (
        telemetry.engine_trace_kwargs(
            _FakeContext(trace_headers_value={"traceparent": "x"}), enabled=False
        )
        == {}
    )
    # No upstream traceparent → omit the kwarg entirely.
    assert telemetry.engine_trace_kwargs(_FakeContext()) == {}
