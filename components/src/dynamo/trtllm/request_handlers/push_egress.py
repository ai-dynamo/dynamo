# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inverted (push-based) Python -> Rust response egress for the TRT-LLM workers.

On the pull path Rust drives the handler's async generator, taking the GIL on
tokio threads once per response. Here the handler -- already on the event loop
holding the GIL when it produces a response -- hands it straight to a Rust
``response_sender`` instead, and Rust advances the generator once per REQUEST
rather than once per RESPONSE. Rationale and measurements: DYN-3703.

Three invariants are load-bearing; breaking any of them is silent:

1. **Shape.** Rust drives both paths with ``demand_driven_python_stream``,
   which does ``getattr("__anext__")``, so push mode must return an **async
   generator** -- never a coroutine. It yields nothing and is advanced exactly
   once: that single ``__anext__`` runs the whole request, draining into the
   sender, then raises ``StopAsyncIteration``.

2. **Opt-in by signature.** ``handler_supports_push`` (Rust) tests
   ``"response_sender" in inspect.signature(handler).parameters``. Applying
   :func:`push_egress_capable` is what puts it there, so that decorator IS the
   switch -- there is no environment variable. It must stay OUTERMOST, and it
   must keep deleting its own ``__wrapped__``, which ``inspect.signature``
   would otherwise follow to the undecorated function and hide the parameter,
   reverting every endpoint to the pull path with nothing logged.

3. **The sender.** ``send(obj)`` once per response; ``close()`` on normal end;
   ``close_with_error(msg)`` on failure. Both closes are idempotent, and Rust
   closes the sink when the generator finishes as a safety net.
"""

import asyncio
import functools
import logging
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# One-shot guard so a bridge mismatch is loud without a line per request.
_warned_no_sender = False


async def drive_push_egress(
    stream: AsyncGenerator[Any, None], response_sender: Any
) -> None:
    """Drain ``stream`` into ``response_sender``, then terminate it exactly once.

    ``send()`` per response, then ``close()`` on normal completion or
    cancellation and ``close_with_error()`` on failure. Errors are reported
    through the sender rather than re-raised, since it is the only channel Rust
    is listening on for response data.
    """
    closed = False

    def _terminate(error: Optional[str]) -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            if error is None:
                response_sender.close()
            else:
                response_sender.close_with_error(error)
        except Exception:
            logger.exception("push egress: failed to close the Rust response stream")

    try:
        async for response in stream:
            # The actual Python -> Rust crossing: depythonize and enqueue,
            # both under the GIL we are already holding.
            response_sender.send(response)
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


async def drive_push_egress_stream(
    stream: AsyncGenerator[Any, None], response_sender: Any
) -> AsyncGenerator[Any, None]:
    """Async-**generator** wrapper around :func:`drive_push_egress`.

    Exists purely for shape (invariant 1 in the module docstring): Rust does
    ``getattr("__anext__")`` on whatever push mode returns, so it must be an
    async generator, not a coroutine.

    The unreachable ``yield`` is what makes Python compile this as an
    async-generator function. Do not remove it, and do not add a reachable one
    -- a push-mode handler that yields now fails its request outright.
    """
    await drive_push_egress(stream, response_sender)
    if False:  # pragma: no cover - never runs; makes this an async generator
        yield


def push_egress_capable(func):
    """Let an async-generator ``generate`` be driven by push OR by pull.

    Turns ``async def generate(self, request, context)`` into a plain ``def``
    returning whichever object the active Rust engine expects: an async
    generator draining into the sender when one is supplied (push, what a
    current binding always does), or the handler's own async generator
    untouched when none is (pull, reached only against a binding predating this
    path). The choice is made purely on whether a sender arrived -- Rust decides
    once at ``serve_endpoint`` time, and second-guessing it here could only
    produce a shape mismatch.

    Must stay the OUTERMOST decorator (invariant 2 in the module docstring).
    Any decorator that inspects what it wraps must sit *inside*, where it still
    sees a real async-generator function rather than this plain ``def``.
    """

    @functools.wraps(func)
    def dispatch(self, request, context=None, response_sender=None, **kwargs):
        global _warned_no_sender

        # Lazy either way: creating an async generator runs none of its body.
        stream = func(self, request, context, **kwargs)

        if response_sender is None:
            if not _warned_no_sender:
                _warned_no_sender = True
                logger.warning(
                    "the Rust bridge did not provide a response_sender for %s; "
                    "falling back to the generator (pull) path. Expect the "
                    "decode-worker GIL cost this decorator exists to remove -- "
                    "the dynamo._core binding is probably older than the "
                    "push-egress path.",
                    getattr(func, "__qualname__", func),
                )
            return stream

        # An async GENERATOR, not the `drive_push_egress` coroutine.
        return drive_push_egress_stream(stream, response_sender)

    # inspect.signature() follows __wrapped__ and would report the undecorated
    # `generate(self, request, context)`, hiding `response_sender` from the Rust
    # opt-in check. Keep the copied __name__/__doc__/__qualname__, drop the link.
    del dispatch.__wrapped__

    return dispatch
