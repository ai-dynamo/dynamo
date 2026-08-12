# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The author-written contract for a pluggable in-process vision encoder.

``VisionEncoderBackend`` is the **single surface an encoder author implements**.
It is a pure policy + compute backend: no threads, no futures, no event loop.
Dynamo owns all the *driving* — the dedicated actor thread, cross-request
coalescing, engine adaptation, and the lifecycle — via ``ThreadedMicroBatcher``
(the generic cross-request batcher) and ``AsyncVisionEncoder`` (the async
request-API glue). This module defines only the contract those drivers call.

The encoder runs in the **same process** as the aggregated vLLM worker (no
separate encode worker, no NIXL transfer): it turns image inputs into ordered,
producer-defined artifacts. The resolved downstream decoder selects an adapter
that validates those artifacts and constructs the final engine prompt.

Division of labour (author vs. Dynamo):

- ``build(model_id)`` — **actor thread, once.** Load weights / tokenizer; warm up
  to peak; if ``buckets`` is set (once CUDA-graph batching is supported), capture
  one CUDA graph per rung here so it is bound to the thread that later replays it
  in ``forward_batch``. Pick the device yourself (``"cuda"`` / the current device).
- ``preprocess(raw) -> Preprocessed{item, cost, bucket_key}`` — **off the actor thread,
  concurrent.** Deterministic, thread-safe, CUDA-free (fetch / resize / patchify
  on CPU/pinned memory). ``cost`` is a **scalar** — how much the item adds toward
  ``max_batch_cost`` (e.g. its patch count). ``bucket_key`` is an opaque
  compatibility partition; items with different keys are never mixed. Raise to reject a bad input —
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
- ``max_batch_items`` — an optional independent item-count ceiling for backends
  with finite graph-batch capacity. ``None`` leaves the count unconstrained.
- ``buckets`` — sorted graph ladder, forward-compatible (unused until CUDA-graph
  batching is supported). ``None``/empty ⇒ eager.
- ``preprocess_concurrency`` — size of the off-thread pool Dynamo runs
  ``preprocess`` on. ``0`` (the **default**) ⇒ no preprocess phase: raws go
  straight to ``forward_batch``. Set ``> 0`` (with an overridden ``preprocess``)
  for off-loop fetch / resize / patchify.

Inside each opaque compatibility partition, Dynamo packs by scalar ``cost`` and
item count. It never inspects item shape; the author owns the key and padding.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Hashable, List, Optional, Sequence, TypeVar

RawT = TypeVar("RawT")  # raw input the author preprocesses (e.g. an image URL)
ItemT = TypeVar("ItemT")  # opaque payload preprocess() hands to forward_batch()
ArtifactT = TypeVar("ArtifactT")  # opaque result consumed by a decoder adapter


@dataclass(frozen=True)
class Preprocessed(Generic[ItemT]):
    """The result of ``preprocess(raw)``: an opaque item plus batching policy.

    ``cost`` is computed **once, off the actor thread**, so the batcher never
    evaluates model policy (it stays torch-free) and packs purely by this scalar.

    Attributes:
        item: Opaque payload passed verbatim to ``forward_batch``.
        cost: Scalar size of this item (``>= 1``); packs toward ``max_batch_cost``.
            Read only in **budgeted mode** (``max_batch_cost`` set). In
            **pass-through mode** (``max_batch_cost`` is ``None``) the batcher
            never reads it, so a pass-through author can leave it at the default
            ``1``.
        bucket_key: Optional opaque compatibility partition. Items with different
            keys are never passed to the same ``forward_batch`` call. ``None``
            preserves the single-partition behavior.
    """

    item: ItemT
    cost: int = 1
    bucket_key: Hashable | None = None


class VisionEncoderBackend(ABC, Generic[RawT, ItemT, ArtifactT]):
    """Author-written, in-process vision encoder contract.

    A pure policy + compute backend — no threads, no futures. Dynamo drives it
    on a dedicated actor thread (``ThreadedMicroBatcher``) and exposes the async
    request API (``AsyncVisionEncoder``). Subclasses implement ``build`` and
    ``forward_batch``; ``preprocess`` (default identity passthrough),
    ``max_batch_cost``, ``max_batch_items``, ``buckets``, and
    ``preprocess_concurrency`` are overridden only as needed. Artifact
    interpretation belongs to the selected adapter.
    """

    #: Scalar dispatch ceiling: the batcher packs items up to this summed ``cost``
    #: per ``forward_batch`` call. ``None`` (the default) ⇒ **pass-through**: no cap
    #: — every drained item in one iteration is handed to a single ``forward_batch``
    #: (the author owns sizing; ``cost`` is ignored).
    max_batch_cost: Optional[int] = None

    #: Optional independent item-count ceiling per ``forward_batch`` call. Use
    #: this for a finite graph-count capacity; keep scalar compute/resource work
    #: in ``max_batch_cost``.
    max_batch_items: Optional[int] = None

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
        """
        ...

    def close(self) -> None:
        """Release thread-affine resources on teardown (actor thread). No-op by
        default; override to free graphs / pools / weights."""
        return None
