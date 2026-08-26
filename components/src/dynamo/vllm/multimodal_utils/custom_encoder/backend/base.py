# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The author-written contract for a pluggable vision encoder.

``VisionEncoderBackend`` is the **single surface an encoder author implements**.
It is a pure policy + compute backend: no threads, no futures, no event loop.
Dynamo owns all the *driving* — the dedicated actor thread, cross-request
coalescing, engine adaptation, and the lifecycle — via ``ThreadedMicroBatcher``
(the generic cross-request batcher) and ``AsyncVisionEncoder`` (the async
request-API glue). This module defines only the contract those drivers call.

The driver may run beside an aggregated vLLM worker or in an inline workflow
stage. It turns image inputs into ordered, producer-defined artifacts. An
in-process decoder adapter or ``EncoderStage`` validates those artifacts for its
downstream consumer.

Division of labour (author vs. Dynamo):

- ``build(model_id)`` — **actor thread, once.** Load weights / tokenizer; warm up
  to peak; if ``buckets`` is set (once CUDA-graph batching is supported), capture
  one CUDA graph per rung here so it is bound to the thread that later replays it
  in ``forward_batch``. Pick the device yourself (``"cuda"`` / the current device).
- ``preprocess(raw) -> Preprocessed{item, cost}`` — **off the actor thread,
  concurrent.** Deterministic, thread-safe, CUDA-free (fetch / resize / patchify
  on CPU/pinned memory). ``cost`` is a **scalar** — how much the item adds toward
  ``max_batch_cost`` (e.g. its visual-token count). Raise to reject a bad input —
  it fails only that image, before any GPU work. **Off by default:** override
  ``preprocess`` *and* set ``preprocess_concurrency > 0`` together to enable this
  pool. With the defaults (identity passthrough, ``preprocess_concurrency = 0``)
  there is no preprocess phase — ``preprocess`` is never called and raws go
  straight to ``forward_batch``. A mismatch (overridden ``preprocess`` with
  ``preprocess_concurrency`` left at ``0``) fails fast at startup.
- ``forward_batch(items, target_bucket=None) -> list[ArtifactT]`` — **actor
  thread, serialized.** ``items`` are a cost-bounded batch (summed ``cost`` within
  the budget). Fence (stream event + sync) and **copy outputs to CPU** before
  returning, so results are safe to consume from another thread. Returns one
  artifact per item, in input order. ``target_bucket`` is reserved for CUDA-graph
  batching, once supported (the ladder rung to pad to); it is ``None`` until then.
- ``close()`` — actor thread, on teardown. Release any thread-affine resources.

Attributes read **once at setup** (never per-request):

- ``max_batch_cost`` — the scalar dispatch ceiling the batcher packs up to; a
  *chosen* budget (a token budget when ``cost`` is a token count). ``None`` (the
  default) ⇒ **pass-through**: no cap (the author owns sizing).
- ``buckets`` — sorted graph ladder, forward-compatible (unused until CUDA-graph
  batching is supported). ``None``/empty ⇒ eager.
- ``preprocess_concurrency`` — size of the off-thread pool Dynamo runs
  ``preprocess`` on. ``0`` (the **default**) ⇒ no preprocess phase: raws go
  straight to ``forward_batch``. Set ``> 0`` (with an overridden ``preprocess``)
  for off-loop fetch / resize / patchify.

Batching is **one-dimensional**: Dynamo packs by scalar ``cost`` up to
``max_batch_cost`` and never inspects item shape — the author owns any
shape/padding concerns inside ``forward_batch``.

Raising errors
--------------

An exception raised anywhere in this contract reaches the HTTP client, and its
**type** — not its message — decides how:

- ``ValueError`` / ``TypeError`` ⇒ the caller's input is at fault. Dynamo maps
  these to ``Backend(InvalidArgument)``, answering **HTTP 400 with the message
  forwarded to the client verbatim**.
- **any other type** (``RuntimeError``, ``TimeoutError``, ``torch`` errors, …)
  ⇒ the engine is at fault. The caller gets a sanitized 5xx and the message
  survives in the server log only.

Pick the type deliberately. Reporting an out-of-memory or a driver fault as
``ValueError`` tells the caller its request was malformed and suppresses the
retry that would have succeeded; reporting a genuinely bad image as
``RuntimeError`` costs the caller the one message that would let them fix it.

.. warning::
   **A ``ValueError``/``TypeError`` message is published to the client.** Do not
   interpolate file paths, tracebacks, tensor dumps, model or weight
   identifiers, or any other server-internal state into one. Describe the fault
   in terms of the request ("image 2 is 1-D; expected a 2-D embedding"), and
   keep the diagnosis in a log line. Non-validation exception types are
   sanitized before they leave the process, so they may say anything.

Note the blast radius differs by method. A ``preprocess`` failure is scoped to
the one image that caused it, but ``forward_batch`` runs a batch coalesced
across *concurrent, unrelated requests*, and an exception there is delivered to
**every request in that batch**. A per-item fault therefore belongs in
``preprocess`` — raised from ``forward_batch`` it would tell unrelated callers
their requests were invalid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, TypeVar

RawT = TypeVar("RawT")  # raw input the author preprocesses (e.g. an image URL)
ItemT = TypeVar("ItemT")  # opaque payload preprocess() hands to forward_batch()
ArtifactT = TypeVar("ArtifactT")  # opaque result consumed by a decoder adapter


@dataclass(frozen=True)
class Preprocessed(Generic[ItemT]):
    """The result of ``preprocess(raw)``: an opaque item plus its batching cost.

    ``cost`` is computed **once, off the actor thread**, so the batcher never
    evaluates model policy (it stays torch-free) and packs purely by this scalar.

    Attributes:
        item: Opaque payload passed verbatim to ``forward_batch``.
        cost: Scalar size of this item (``>= 1``); packs toward ``max_batch_cost``.
            Read only in **budgeted mode** (``max_batch_cost`` set). In
            **pass-through mode** (``max_batch_cost`` is ``None``) the batcher
            never reads it, so a pass-through author can leave it at the default
            ``1``.
    """

    item: ItemT
    cost: int = 1


class VisionEncoderBackend(ABC, Generic[RawT, ItemT, ArtifactT]):
    """Author-written vision encoder contract.

    A pure policy + compute backend — no threads, no futures. Dynamo drives it
    on a dedicated actor thread (``ThreadedMicroBatcher``) and exposes the async
    request API (``AsyncVisionEncoder``). Subclasses implement ``build`` and
    ``forward_batch``; ``preprocess`` (default identity passthrough),
    ``max_batch_cost``, ``buckets``, and ``preprocess_concurrency`` are overridden
    only as needed. Artifact interpretation belongs to the selected adapter.
    """

    #: Scalar dispatch ceiling: the batcher packs items up to this summed ``cost``
    #: per ``forward_batch`` call. ``None`` (the default) ⇒ **pass-through**: no cap
    #: — every drained item in one iteration is handed to a single ``forward_batch``
    #: (the author owns sizing; ``cost`` is ignored).
    max_batch_cost: Optional[int] = None

    #: Sorted graph ladder (the captured rungs), **forward-compatible** — unused
    #: until CUDA-graph batching is supported. ``None``/empty ⇒ eager.
    buckets: Optional[Sequence[int]] = None

    #: Off-loop preprocess pool size Dynamo runs ``preprocess`` on. Not just a
    #: pool size — it gates the preprocess **phase**: ``0`` (the **default**) ⇒
    #: ``preprocess`` is never called and raws go straight to ``forward_batch``
    #: (``raw`` is the item; do any prep there). Set ``> 0`` (with an overridden
    #: ``preprocess``) to fetch / resize / patchify off the actor thread; overriding
    #: ``preprocess`` while leaving this at ``0`` fails fast at startup. Whether an
    #: encoder needs off-loop prep is a property of the encoder, so it lives here;
    #: the driver takes an optional override for tuning.
    preprocess_concurrency: int = 0

    # ---- subclass contract -------------------------------------------------

    @abstractmethod
    def build(self, model_id: str) -> None:
        """Load weights / tokenizer, warm up, capture graphs (actor thread, once).

        Any CUDA graph captured here is bound to the thread that later replays it.
        Pick the device yourself (``"cuda"`` / the current device). All CUDA init
        happens here.
        """
        ...

    def preprocess(self, raw: RawT) -> Preprocessed[ItemT]:
        """Turn a raw input into a ``Preprocessed`` item (off the actor thread).

        The default is an **identity passthrough** (``raw`` is the item, ``cost``
        ``1``), so by default there is no preprocessing. Override it for off-loop
        fetch + HF processing **and** set ``preprocess_concurrency > 0`` to run it
        on the pool — it must then be deterministic, thread-safe, and CUDA-free.
        Raise to reject a bad input — it fails only that image, before submit.
        With ``preprocess_concurrency == 0`` this method is **never called**;
        overriding it without raising the concurrency fails fast at startup.

        This is the right place to reject a malformed image: the failure is
        scoped to the one raw that caused it, so a ``ValueError``/``TypeError``
        here reaches exactly the caller who sent it, as an HTTP 400 carrying the
        message. Keep that message free of server-internal detail (see
        *Raising errors* in the module docstring).
        """
        return Preprocessed(item=raw)  # type: ignore[arg-type]  # ItemT == RawT

    @abstractmethod
    def forward_batch(
        self, items: List[ItemT], target_bucket: Optional[int] = None
    ) -> List[ArtifactT]:
        """Encode one cost-bounded batch; one artifact per item, in input order.

        Artifacts are opaque, producer-defined values. Dynamo preserves their
        order and passes them unchanged to the selected adapter, which owns the
        concrete artifact contract and validation.

        Fence (stream event + sync) and **copy outputs to CPU** before returning,
        so results are safe to consume from another thread. ``target_bucket`` is
        reserved for CUDA-graph batching, once supported (the ladder rung to pad
        to), and is ``None`` until then.

        Raises:
            Exception: fails **every request in the batch**, not just the item
                that caused it — ``items`` is coalesced across concurrent,
                unrelated requests, and the exception is delivered to all of
                them. So prefer a non-validation type here (the engine, not any
                one caller, is what failed), and push per-item rejection up into
                ``preprocess`` where it is scoped to a single image. A
                ``ValueError``/``TypeError`` raised here answers HTTP 400 with
                its message forwarded verbatim to callers who may have sent
                perfectly valid input. See *Raising errors* in the module
                docstring.
        """
        ...

    def close(self) -> None:
        """Release thread-affine resources on teardown (actor thread). No-op by
        default; override to free graphs / pools / weights."""
        return None
