# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inverted (push-based) Python -> Rust response egress for Python workers.

On the pull path Rust drives the handler's async generator, taking the GIL on
tokio threads once per response. Here the handler -- already on the event loop
holding the GIL when it produces a response -- hands it straight to a Rust
``response_sender`` instead, and Rust advances the generator once per REQUEST
rather than once per RESPONSE. Rationale and measurements: #12845.

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

3. **The sender.** ``send(obj)`` once per response and ``close()`` on normal
   end -- and nothing on failure. Exceptions propagate out of the handler, so
   Rust classifies them with ``map_python_exception`` and terminates the
   stream with a *typed* backend error. Reporting them here instead, through
   ``close_with_error``, would flatten every failure to an untyped string:
   ``EngineShutdown`` would stop triggering request migration and worker
   inhibition, and a client's bad input would surface as an unknown server
   error. ``close()`` is idempotent, and Rust closes the sink when the
   generator finishes as a safety net.
"""

import functools
import logging
from typing import Any, AsyncGenerator

logger = logging.getLogger(__name__)

# One-shot latch: say it once, not once per call.
_logged_no_sender = False


async def drive_push_egress(
    stream: AsyncGenerator[Any, None], response_sender: Any
) -> None:
    """Drain ``stream`` into ``response_sender``, then close it.

    Nothing is caught. Every way this can fail -- engine shutdown, a rejected
    request, cancellation, the consumer dropping the stream -- is an exception,
    and letting it out is what lets Rust's ``map_python_exception`` classify it
    and terminate the stream with a *typed* backend error, exactly as on the
    pull path. See invariant 3 in the module docstring for what catching it
    would cost.
    """
    async for response in stream:
        # The actual Python -> Rust crossing: encode to request-plane bytes
        # and enqueue, both under the GIL we are already holding.
        response_sender.send(response)
    response_sender.close()


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
    returning whichever object the calling Rust engine expects: an async
    generator draining into the sender when one is supplied (push), or the
    handler's own async generator untouched when none is (pull). The choice is
    made purely on whether a sender arrived -- Rust decides per call, and
    second-guessing it here could only produce a shape mismatch.

    Both arms are live in normal operation. Network requests arrive through the
    push ingress; in-process callers and the canary health check go through the
    pull engine registered in the local endpoint registry, which passes no
    sender. That is why the pull arm must keep working rather than being
    treated as a legacy fallback.

    Must stay the OUTERMOST decorator (invariant 2 in the module docstring).
    Any decorator that inspects what it wraps must sit *inside*, where it still
    sees a real async-generator function rather than this plain ``def``.
    """

    @functools.wraps(func)
    def dispatch(self, request, context=None, response_sender=None, **kwargs):
        global _logged_no_sender

        # Lazy either way: creating an async generator runs none of its body.
        stream = func(self, request, context, **kwargs)

        if response_sender is None:
            if not _logged_no_sender:
                _logged_no_sender = True
                # Normal for in-process and canary calls. Only a problem if it
                # is EVERY call, which would mean the signature probe in
                # `serve_endpoint` found no `response_sender` and network
                # traffic is on the pull path too.
                logger.info(
                    "no response_sender for %s; serving this call on the pull "
                    "path (expected for in-process and health-check calls)",
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
