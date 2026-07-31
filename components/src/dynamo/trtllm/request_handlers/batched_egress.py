# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flag-gated cross-request batched egress for the TRT-LLM decode worker.

Root cause (DYN-3703): every response token is pulled out of a *per-request*
Python async generator and crosses the Python->Rust boundary individually,
through the pyo3-async bridge, on ONE GIL-bound event loop (~37 us/token on a
96-core node). Per decode iteration at batch B that is B * 37 us on a single
lane, which eventually can't keep up with the engine and back-pressures prefill.

This module batches egress ACROSS requests: all of a step's ready responses are
handed to Rust in ONE bridge crossing, then Rust demuxes them back to each
request's response stream by id. It is the analogue of TRT-LLM's
``handle_for_ipc_batched`` (base_worker.py:860), which is exactly how
``trtllm serve`` keeps its egress cost from scaling with batch.

Design goal: reuse ``handler.generate`` UNCHANGED, so cancellation, error
mapping, disagg prefill/decode, and metrics behave identically to the
per-request path. The only new work is (a) a per-request feeder that tags each
chunk with the request id and pushes it onto a shared queue, and (b) a single
``drain`` generator that coalesces whatever is ready into one yield.

Enable with ``DYN_TRTLLM_BATCHED_EGRESS=1`` (default OFF -> no behavior change).

Pairs with the Rust ``Endpoint.serve_endpoint_batched(submit, drain, ...)`` +
``PythonBatchedEgressEngine`` added in the same patch.
"""

import asyncio
import logging
import os
from typing import Any, AsyncGenerator, Optional


def batched_egress_enabled() -> bool:
    """True when DYN_TRTLLM_BATCHED_EGRESS selects the batched path."""
    return os.environ.get("DYN_TRTLLM_BATCHED_EGRESS", "0").lower() not in (
        "0",
        "",
        "false",
        "no",
    )


# Sentinel pushed after a request's generator completes so the Rust demux can
# close that request's response stream. Sent over the queue as the chunk value;
# the drain converts it to ``None`` on the wire so Rust does not have to know
# about this Python object.
_DONE = object()


class BatchedEgress:
    """Owns the shared egress queue, the per-request feeders, and the drain.

    One instance per worker. ``submit`` is scheduled once per request by the
    Rust engine; ``drain`` is driven once by the Rust engine's single forwarder.
    """

    def __init__(self, handler: Any) -> None:
        self._handler = handler
        # Unbounded: back-pressure is applied downstream by each request's Rust
        # mpsc channel (RESPONSE_CHANNEL_DEPTH). Keeping the shared queue
        # unbounded avoids one slow client stalling the whole worker's drain.
        self._q: "asyncio.Queue[tuple[str, Any]]" = asyncio.Queue()

    async def submit(self, request: dict, context: Any) -> None:
        """Run one request's existing generator, tag each chunk with the Dynamo
        request id, and push it onto the shared queue.

        Reuses ``handler.generate`` verbatim, so all per-request semantics are
        preserved. Any exception is surfaced as a tagged error frame so the Rust
        side can map it exactly like the per-request path does.
        """
        rid = context.id()
        try:
            async for chunk in self._handler.generate(request, context):
                await self._q.put((rid, chunk))
        except asyncio.CancelledError:
            # Client/connection cancellation: let the sentinel close the stream.
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a tagged frame
            logging.exception("batched egress feeder failed for request %s", rid)
            await self._q.put(
                (rid, {"finish_reason": {"error": repr(exc)}, "token_ids": []})
            )
        finally:
            await self._q.put((rid, _DONE))

    async def drain(self, context: Optional[Any] = None) -> AsyncGenerator[list, None]:
        """Single multiplexed generator for the whole worker.

        Each ``__anext__`` returns ALL currently-ready ``(rid, chunk)`` pairs as
        one list, so a whole engine step's responses cross the bridge in a single
        crossing. Emits ``list[tuple[str, Optional[dict]]]``; a chunk of ``None``
        marks that request finished (Rust closes its stream and drops the route).
        """
        q = self._q
        while True:
            # Block for the first item so we never spin, then greedily drain
            # everything else already queued into the same batch.
            first_rid, first_chunk = await q.get()
            batch = [(first_rid, None if first_chunk is _DONE else first_chunk)]
            while not q.empty():
                rid, chunk = q.get_nowait()
                batch.append((rid, None if chunk is _DONE else chunk))
            yield batch


async def serve_batched_endpoint(
    endpoint: Any, handler: Any, **serve_kwargs: Any
) -> None:
    """Register the batched-egress engine on ``endpoint``.

    Mirrors ``endpoint.serve_endpoint(handler.generate, ...)`` but routes through
    the Rust ``PythonBatchedEgressEngine`` via ``serve_endpoint_batched``.
    """
    be = BatchedEgress(handler)
    await endpoint.serve_endpoint_batched(be.submit, be.drain, **serve_kwargs)
