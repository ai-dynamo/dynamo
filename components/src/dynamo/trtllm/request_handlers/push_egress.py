# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inverted (push-based) Python -> Rust response egress for the TRT-LLM workers.

Why
---
On the pull path the Rust bridge drives the handler's async generator
(``lib/bindings/python/rust/engine.rs``, ``demand_driven_python_stream``). Per
response a tokio worker takes the GIL to call ``__anext__()``, schedules the
resulting coroutine onto the Python event loop via ``call_soon_threadsafe``,
parks, is woken, and then takes the GIL a *second* time to depythonize the
item. Three GIL acquisitions per response, two of them on arbitrary tokio
threads. On the decode worker -- 45 GIL-capable threads in the app interpreter
versus ``trtllm-serve``'s 3, and a GIL wait/hold ratio of 23.4 versus serve's
0.3 -- those cross-thread acquisitions are the expensive ones.

Inverting the direction removes them. The handler is ALREADY running on the
event loop holding the GIL at the moment it produces a response, so it hands
that response straight to a Rust sender: one GIL acquisition, an existing one,
and no tokio thread ever touches Python. Rust advances the handler once per
REQUEST instead of once per RESPONSE.

Rust-side contract
------------------
Implemented in ``lib/bindings/python/rust/push_egress.rs`` (``ResponseSender``,
``PythonPushEngine``) and selected in ``lib.rs::serve_endpoint``. Three things
matter to this file, and all three are load-bearing:

1. **Shape.** Rust drives BOTH paths through the same helper --
   ``PythonPushEngine::generate`` calls ``engine::invoke_generator``, which ends
   in ``demand_driven_python_stream``::

       let anext = generator.getattr("__anext__")?.unbind();   // engine.rs:126

   So **both** paths must return an **async generator**; the difference is only
   how often it is advanced. Pull yields once per response. Push yields nothing
   and is advanced ONCE per request: that single ``__anext__`` runs the whole
   request (draining into ``response_sender``) and then raises
   ``StopAsyncIteration``. Responses travel out of band on the sender.

   .. warning::

      Returning a **coroutine** here does not work and fails loudly: a coroutine
      has no ``__anext__``, so every request dies with ``AttributeError:
      'coroutine' object has no attribute '__anext__'`` plus a
      ``RuntimeWarning: coroutine 'drive_push_egress' was never awaited``. This
      is exactly how runs 339221/339222 failed -- this docstring previously
      described an ``into_future_with_locals`` call on the Rust side that was
      never written. Keep :func:`drive_push_egress_stream` (an async generator)
      as the returned object, not :func:`drive_push_egress` (a coroutine).

   Keeping the generator shape on both paths is deliberate on the Rust side
   (``push_egress.rs:427-437``): registration, cancellation, and the graceful
   fallback for a handler that yields anyway all stay untouched.

2. **Opt-in by signature.** Rust enables push mode only when
   ``DYN_TRTLLM_PUSH_EGRESS=1`` *and* ``handler_supports_push()`` says the
   handler declares a ``response_sender`` parameter, which it tests with
   ``inspect.signature(handler).parameters``. ``inspect.signature`` follows
   ``__wrapped__``, so any decorator stacked on ``generate`` that uses
   ``functools.wraps`` would hide the parameter and silently drop the worker
   back to the pull path. :func:`push_egress_capable` therefore drops its own
   ``__wrapped__`` link. Keep it that way, and keep this decorator OUTERMOST.
   (``context`` is detected the same way, in both engines.)

3. **The sender.** ``response_sender`` and ``context`` arrive as keyword
   arguments. The object exposes exactly::

       send(obj)              one call per response; converts under our GIL and
                              enqueues. Blocks when the consumer is behind, but
                              releases the GIL across the wait.
       close()                normal end of stream. Idempotent. Replaces the
                              StopAsyncIteration the pull path relied on.
       close_with_error(msg)  error termination. Idempotent; a later close() is
                              a no-op.

   Rust also installs a safety net that closes the sink when the coroutine
   finishes, so a missed ``close()`` cannot hang a stream -- but we close
   explicitly anyway, because our error message is better than the generic one.

Where the push happens
----------------------
At the OUTERMOST handler boundary (``{Decode,Prefill,Encode,Aggregated}
Handler.generate``), not at the innermost ``yield out`` inside
``handler_base._generate_locally_impl``. Two reasons:

1. The expensive crossing is the outermost one -- the one that reaches Rust.
   The intermediate ``yield``s (``_generate_locally_impl`` ->
   ``generate_locally`` -> ``Handler.generate``) are pure-Python generator
   delegation on one thread: no GIL release, no bridge, no tokio thread. Moving
   the push inward would buy nothing measurable and would fork a 300-line
   function.
2. Semantics are preserved *by construction*. The whole existing generator
   stack runs untouched, so termination, ``is_final``/``finish_reason``
   handling, the ``trtllm:first_response`` mark, error-frame mapping,
   cancellation via ``_cancellation_monitor``, PrefillHandler's "exactly one
   response" guard and its ``context.is_stopped()`` check all behave exactly as
   they do on the pull path. Only the last hop changes: ``yield res`` becomes
   ``sender.send(res)``.

Flag
----
``DYN_TRTLLM_PUSH_EGRESS=1`` enables it; default OFF. Both paths ship in one
image so they can be A/B tested. With the flag unset, Rust never hands over a
sender and :func:`push_egress_capable` is a transparent passthrough, so the
generator path behaves exactly as before.
"""

import asyncio
import functools
import logging
import os
from typing import Any, AsyncGenerator, Optional

from dynamo.common.utils import nvtx_utils as _nvtx

logger = logging.getLogger(__name__)

PUSH_EGRESS_ENV_VAR = "DYN_TRTLLM_PUSH_EGRESS"

# Read once at import, and parsed EXACTLY as push_egress.rs parses it
# (`value == "1"`). A looser parse here would let the two halves disagree about
# which path is active, which is the one disagreement that actually breaks: the
# Rust side would await an async generator, or drive a coroutine with
# __anext__.
_PUSH_EGRESS_ENABLED: bool = os.environ.get(PUSH_EGRESS_ENV_VAR) == "1"

# One-shot guard so a misconfiguration is loud without a line per request.
_warned_no_sender = False


def push_egress_enabled() -> bool:
    """True when ``DYN_TRTLLM_PUSH_EGRESS`` selects the push (inverted) path.

    Informational only. The authority on which path a given request takes is
    whether Rust handed us a ``response_sender``: Rust decides once, at
    ``serve_endpoint`` time, and a sender is only ever passed in push mode.
    """
    return _PUSH_EGRESS_ENABLED


async def drive_push_egress(
    stream: AsyncGenerator[Any, None], response_sender: Any
) -> None:
    """Drain ``stream`` into ``response_sender``, then terminate the stream.

    Call sequence, per request:

        send(res) * N  ->  close()                  normal completion
        send(res) * N  ->  close()                  cancellation
        send(res) * N  ->  close_with_error(msg)    failure

    ``close()``/``close_with_error()`` are issued exactly once from here (and
    are idempotent on the Rust side regardless).

    On errors: in push mode the sender is the only channel Rust is listening
    on for response data, so a failure is reported through
    ``close_with_error`` and is not re-raised. Most per-request errors never
    reach here anyway -- ``_generate_locally_impl`` already converts
    ``RequestError`` and generic exceptions into
    ``{"finish_reason": {"error": ...}}`` frames, which travel as ordinary
    ``send()`` calls exactly as they do on the pull path.
    """
    # Spans awaits, so start/end range rather than the annotate context manager:
    # annotate uses the thread's nested push/pop stack, which interleaves
    # incorrectly when another coroutine resumes on the same event loop.
    egress_rng = _nvtx.start_range("trtllm:push_egress", color="cyan")
    closed = False

    def _terminate(error: Optional[str]) -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            if error is None:
                _nvtx.mark("trtllm:push_close", color="cyan")
                response_sender.close()
            else:
                _nvtx.mark("trtllm:push_error", color="red")
                response_sender.close_with_error(error)
        except Exception:
            logger.exception("push egress: failed to close the Rust response stream")

    try:
        async for response in stream:
            # The actual Python -> Rust crossing: pythonize/enqueue happens
            # under the GIL we are already holding. Kept as a tight range so
            # nsys shows bridge cost only, with engine time falling in the gaps
            # between consecutive ranges.
            send_rng = _nvtx.start_range("trtllm:push_send", color="cyan")
            try:
                response_sender.send(response)
            finally:
                _nvtx.end_range(send_rng)
    except (asyncio.CancelledError, GeneratorExit):
        # Client/connection cancellation. `_cancellation_monitor` has already
        # abort()ed the engine request and `_generate_locally_impl` swallows its
        # own CancelledError, so getting here means the enclosing task was
        # cancelled. End the stream normally -- the pull path's cancelled
        # generator likewise just stops producing -- then re-raise so asyncio's
        # cancellation contract is honored.
        _terminate(None)
        raise
    except Exception as exc:
        logger.exception("push egress: request failed")
        _terminate(f"{type(exc).__name__}: {exc}")
    except BaseException:
        # KeyboardInterrupt / SystemExit: still close the stream, then let it go.
        _terminate("worker interrupted")
        raise
    else:
        _terminate(None)
    finally:
        _nvtx.end_range(egress_rng)


async def drive_push_egress_stream(
    stream: AsyncGenerator[Any, None], response_sender: Any
) -> AsyncGenerator[Any, None]:
    """Async-**generator** wrapper around :func:`drive_push_egress`.

    This is what push mode returns to Rust. It exists purely for shape: Rust
    drives the push path with ``demand_driven_python_stream``, which does
    ``getattr("__anext__")``, so the object it receives must be an async
    generator and not a coroutine (see "Shape" in the module docstring).

    The unreachable ``yield`` below is load-bearing -- it is what makes Python
    compile this as an async-generator function. Do not remove it, and do not
    add a reachable one: yielding here would push the response back onto the
    pull path via the Rust driver's fallback arm
    (``pybridge.push_forward_yield``), reintroducing the per-response GIL
    acquisition this whole change exists to remove.

    Advanced exactly once per request: that call runs the request to completion,
    then falls off the end and raises ``StopAsyncIteration``.
    """
    await drive_push_egress(stream, response_sender)
    if False:  # pragma: no cover - never runs; makes this an async generator
        yield


def push_egress_capable(func):
    """Let an async-generator ``generate`` be driven by push OR by pull.

    Turns ``async def generate(self, request, context)`` into a plain ``def``
    that returns whichever object the active Rust engine expects:

    * no ``response_sender`` (pull, the default) -> the **async generator**
      itself, completely untouched. With ``DYN_TRTLLM_PUSH_EGRESS`` unset this
      is the only path, and nothing about existing behavior changes.
    * ``response_sender=<ResponseSender>`` (push) -> a **coroutine** that
      drains that same generator into the sender via
      :func:`drive_push_egress`.

    A sender is honored whenever one is supplied, without consulting the env
    var: Rust only ever supplies one when it has already committed to awaiting
    a coroutine, so second-guessing it here could only produce a shape
    mismatch.

    Must stay the OUTERMOST decorator on ``generate``:

    * the Rust ``handler_supports_push()`` sniff runs
      ``inspect.signature()`` on the registered callable, so ``response_sender``
      has to be visible in the outermost signature (hence the ``__wrapped__``
      link is dropped below -- ``inspect.signature`` follows it);
    * ``nvtx_utils.range_decorator`` must sit *inside*, where it still sees a
      real async-generator function. Applied to this plain ``def`` it would
      take its synchronous branch and close the range as soon as the
      generator/coroutine object was constructed, measuring nothing.
    """

    @functools.wraps(func)
    def dispatch(self, request, context=None, response_sender=None, **kwargs):
        global _warned_no_sender

        # Lazy either way: creating an async generator runs none of its body,
        # and drive_push_egress() is a coroutine function.
        stream = func(self, request, context, **kwargs)

        # Safety net. Rust delivers the sender BOTH as this keyword argument
        # and on `context.response_sender` (push_egress.rs, the kwarg-list
        # closure) -- they are the same Py object. Reading the context too means
        # a future Rust change that drops one delivery route degrades to the
        # pull path instead of failing every request: without a sender we return
        # the async generator, which is only correct if Rust also stayed on the
        # pull path. That is why the warning below is loud.
        if response_sender is None:
            response_sender = getattr(context, "response_sender", None)

        if response_sender is None:
            if _PUSH_EGRESS_ENABLED and not _warned_no_sender:
                _warned_no_sender = True
                logger.warning(
                    "%s=1 but the Rust bridge did not provide a response_sender "
                    "for %s; falling back to the generator (pull) path.",
                    PUSH_EGRESS_ENV_VAR,
                    getattr(func, "__qualname__", func),
                )
            return stream

        # An async GENERATOR, not the `drive_push_egress` coroutine: Rust does
        # `getattr("__anext__")` on whatever comes back. See the module
        # docstring's "Shape" section.
        return drive_push_egress_stream(stream, response_sender)

    # functools.wraps sets __wrapped__, and inspect.signature() follows it --
    # which would report the undecorated `generate(self, request, context)` and
    # hide `response_sender` from the Rust opt-in check, silently disabling push
    # egress. Keep the copied __name__/__doc__/__qualname__, drop the link.
    del dispatch.__wrapped__

    return dispatch
