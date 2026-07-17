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
separate encode worker, no NIXL transfer). It declares the semantic shape of its
artifact through ``output_format``; the resolved downstream decoder selects the
engine adapter. Text-only decoders use a mixed ``EmbedsPrompt``. Qwen2/2.5-VL
decoders use vLLM's native external multimodal ``TokensPrompt`` route.

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
- ``forward_batch(items, target_bucket=None) -> EncoderBatchOutput`` — **actor
  thread, serialized.** ``items`` are a cost-bounded batch (summed ``cost`` within
  the budget). Fence (stream event + sync) and **copy outputs to CPU** before
  returning, so results are safe to consume from another thread. Returns one
  artifact per item, in input order. ``"tensor"`` producers return CPU tensors;
  ``"qwen2_vl_projected_grid"`` producers return ``Qwen2VLImageEncoding``.
  ``target_bucket`` is reserved for CUDA-graph batching, once supported (the
  ladder rung to pad to); it is ``None`` until then.
- ``close()`` — actor thread, on teardown. Release any thread-affine resources.

Attributes read **once at setup** (never per-request):

- ``output_format`` — the semantic artifact kind. It does not choose the engine
  prompt route; Dynamo selects that from the resolved decoder.
- ``image_token_id`` — for ``"tensor"`` output, the token id marking image positions;
  **hardcode it for your model** (e.g. ``151655`` for Qwen3-VL's ``<|image_pad|>``).
  Dynamo uses it to locate each image span for the splice.
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
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Literal, Optional, Sequence, TypeAlias, TypeVar

import torch

RawT = TypeVar("RawT")  # raw input the author preprocesses (e.g. an image URL)
ItemT = TypeVar("ItemT")  # opaque payload preprocess() hands to forward_batch()


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


EncoderOutputFormat: TypeAlias = Literal["tensor", "qwen2_vl_projected_grid"]


@dataclass(frozen=True)
class Qwen2VLImageEncoding:
    """Projected Qwen2/2.5-VL rows and their pre-merge patch grid.

    Rows are in canonical ``(t, merged_h, merged_w)`` raster order. A Qwen2.5
    producer must undo the tower's window permutation before returning.
    """

    projected: torch.Tensor
    grid_thw: tuple[int, int, int]


EncoderOutput: TypeAlias = torch.Tensor | Qwen2VLImageEncoding
EncoderBatchOutput: TypeAlias = list[torch.Tensor] | list[Qwen2VLImageEncoding]


class VisionEncoderBackend(ABC, Generic[RawT, ItemT]):
    """Author-written, in-process vision encoder contract.

    A pure policy + compute backend — no threads, no futures. Dynamo drives it
    on a dedicated actor thread (``ThreadedMicroBatcher``) and exposes the async
    request API (``AsyncVisionEncoder``). Subclasses implement ``build`` and
    ``forward_batch`` and declare ``output_format``. Tensor producers also set
    ``image_token_id``. ``preprocess`` (default identity passthrough),
    ``max_batch_cost``, ``buckets``, and ``preprocess_concurrency`` are overridden
    only as needed.
    """

    #: Tensor-output placeholder token id — **hardcode it for your model** (e.g. ``151655``
    #: for Qwen3-VL's ``<|image_pad|>``; resolve it from your tokenizer offline if
    #: unsure). Dynamo uses it to locate each image span for the splice. Declared
    #: without a default so a backend that forgets to set it fails fast at startup.
    image_token_id: int

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

    #: Producer-semantic artifact declaration. The downstream decoder still
    #: selects the concrete engine adapter; this only validates compatibility.
    output_format: EncoderOutputFormat = "tensor"

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
    ) -> EncoderBatchOutput:
        """Encode one cost-bounded batch; one artifact per item, in order.

        Fence (stream event + sync) and **copy outputs to CPU** before returning,
        so results are safe to consume from another thread. Tensor backends return
        ``(n_visual_tokens, lm_hidden_dim)`` rows. Native Qwen backends return
        ``Qwen2VLImageEncoding``. ``target_bucket`` is reserved for CUDA-graph
        batching and is ``None`` until then.
        """
        ...

    def close(self) -> None:
        """Release thread-affine resources on teardown (actor thread). No-op by
        default; override to free graphs / pools / weights."""
        return None
