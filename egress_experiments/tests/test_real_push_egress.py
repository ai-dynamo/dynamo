# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests against the **real**
``components/src/dynamo/trtllm/request_handlers/push_egress.py``.

Not a lookalike: :mod:`egress_experiments.dynamo_sim.realcode` loads the file
that ships. The invariants below are the ones its module docstring calls
load-bearing, each of which has already broken a run at least once:

* **Shape.** ``push_egress_capable`` must return an async **generator**, not a
  coroutine. Runs 339221/339222 died with ``AttributeError: 'coroutine' object
  has no attribute '__anext__'`` because Rust does ``getattr("__anext__")`` on
  whatever comes back. Fixed in 0fb02c2ea6.
* **Opt-in by signature.** Rust enables push mode only if
  ``inspect.signature(handler).parameters`` shows ``response_sender``. Since
  ``inspect.signature`` follows ``__wrapped__``, ``functools.wraps``'s link is
  deleted; restoring it would silently drop every worker back to the pull path.
* **Termination.** ``send`` * N then exactly one ``close()``; on failure
  ``close_with_error`` instead, and both idempotent.
* **Passthrough.** With no sender the decorator is transparent -- the async
  generator is returned untouched, so the pull path is bit-identical to before
  the change.
"""

from __future__ import annotations

import asyncio
import inspect
from typing import Any, List

import pytest

from egress_experiments.dynamo_sim import realcode

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.none,
]

_pe = realcode.load_push_egress()
if _pe is None:  # pragma: no cover - only without the checkout
    pytest.skip(
        f"real push_egress.py unavailable: {realcode.load_failure()}",
        allow_module_level=True,
    )


class RecordingSender:
    """The ``ResponseSender`` contract from ``push_egress.rs``, recorded."""

    def __init__(self) -> None:
        self.calls: List[str] = []
        self.items: List[Any] = []
        self.error: str | None = None

    def send(self, obj: Any) -> None:
        self.calls.append("send")
        self.items.append(obj)

    def close(self) -> None:
        self.calls.append("close")

    def close_with_error(self, message: str) -> None:
        self.calls.append("close_with_error")
        self.error = message


class Handler:
    """Stands in for ``AggregatedHandler``: the decorator is the real one."""

    def __init__(self, count: int = 3, fail_at: int | None = None) -> None:
        self.count = count
        self.fail_at = fail_at

    @_pe.push_egress_capable
    async def generate(self, request, context):
        for i in range(self.count):
            if self.fail_at is not None and i == self.fail_at:
                raise ValueError("engine exploded")
            yield {"token_ids": [i], "index": 0}


async def _drain(obj) -> List[Any]:
    """What Rust does: getattr('__anext__') and advance until StopAsyncIteration."""
    anext = obj.__anext__
    out = []
    while True:
        try:
            out.append(await anext())
        except StopAsyncIteration:
            return out


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def test_push_mode_returns_an_async_generator_not_a_coroutine():
    """The 0fb02c2ea6 regression, pinned."""
    obj = Handler().generate({}, None, response_sender=RecordingSender())
    try:
        assert not asyncio.iscoroutine(obj), (
            "push mode returned a coroutine; Rust's demand_driven_python_stream "
            "does getattr('__anext__') and every request would die with "
            "AttributeError"
        )
        assert hasattr(obj, "__anext__")
        assert inspect.isasyncgen(obj)
    finally:
        asyncio.run(obj.aclose())


def test_push_generator_is_advanced_exactly_once_per_request():
    """One ``__anext__`` runs the whole request, then StopAsyncIteration.

    Anything yielded here would take the Rust driver's fallback arm
    (``pybridge.push_forward_yield``) and put the per-response GIL acquisition
    straight back.
    """
    sender = RecordingSender()
    obj = Handler(count=5).generate({}, None, response_sender=sender)

    advances = 0

    async def main():
        nonlocal advances
        anext = obj.__anext__
        while True:
            try:
                item = await anext()
            except StopAsyncIteration:
                return
            advances += 1
            assert item is None, f"push mode yielded {item!r}"

    asyncio.run(main())

    assert advances == 0, "the push generator must yield nothing"
    assert sender.calls == ["send"] * 5 + ["close"]
    assert [i["token_ids"][0] for i in sender.items] == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------
# Opt-in by signature
# --------------------------------------------------------------------------


def test_response_sender_is_visible_to_the_rust_opt_in_sniff():
    params = inspect.signature(Handler.generate).parameters
    assert "response_sender" in params, (
        "handler_supports_push() would return False and the worker would "
        "silently stay on the pull path"
    )
    assert "context" in params


def test_wrapped_link_is_dropped():
    """``functools.wraps`` sets ``__wrapped__`` and ``inspect.signature``
    follows it, which would hide ``response_sender`` again."""
    assert not hasattr(Handler.generate, "__wrapped__")
    # ...while still looking like the function it wraps.
    assert Handler.generate.__name__ == "generate"


# --------------------------------------------------------------------------
# Termination
# --------------------------------------------------------------------------


def test_error_terminates_with_close_with_error_after_partial_sends():
    sender = RecordingSender()
    obj = Handler(count=5, fail_at=2).generate({}, None, response_sender=sender)
    asyncio.run(_drain(obj))

    assert sender.calls == ["send", "send", "close_with_error"]
    assert sender.error is not None and "ValueError" in sender.error
    # No close() after close_with_error: termination happens exactly once.
    assert sender.calls.count("close") == 0


def test_cancellation_closes_the_stream_and_reraises():
    """``_cancellation_monitor`` has already aborted the engine request; the
    pull path's cancelled generator simply stops producing, so push closes
    normally -- and must still honour asyncio's cancellation contract."""

    class Cancelling:
        @_pe.push_egress_capable
        async def generate(self, request, context):
            yield {"token_ids": [0]}
            raise asyncio.CancelledError

    sender = RecordingSender()
    obj = Cancelling().generate({}, None, response_sender=sender)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_drain(obj))

    assert sender.calls == ["send", "close"]


# --------------------------------------------------------------------------
# Passthrough
# --------------------------------------------------------------------------


def test_without_a_sender_the_decorator_is_transparent():
    obj = Handler(count=3).generate({}, None)
    assert inspect.isasyncgen(obj)
    items = asyncio.run(_drain(obj))
    assert [i["token_ids"][0] for i in items] == [0, 1, 2]


def test_sender_on_the_context_is_honoured():
    """Rust delivers the sender BOTH as a kwarg and on ``context``; the
    decorator's safety net reads the context so losing one delivery route
    degrades to pull instead of failing every request."""

    class Ctx:
        def __init__(self, sender):
            self.response_sender = sender

    sender = RecordingSender()
    obj = Handler(count=2).generate({}, Ctx(sender))
    asyncio.run(_drain(obj))
    assert sender.calls == ["send", "send", "close"]


def test_the_simulation_uses_this_module():
    """Guard against the sim quietly falling back to its own stand-in."""
    from egress_experiments.dynamo_sim.worker import USING_REAL_PUSH_EGRESS

    assert USING_REAL_PUSH_EGRESS is True
