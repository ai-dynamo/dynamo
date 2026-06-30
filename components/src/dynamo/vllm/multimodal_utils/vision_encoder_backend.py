# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The author-written contract for a pluggable in-process vision encoder.

``VisionEncoderBackend`` is the **single surface an encoder author implements**.
It is a pure policy + compute backend: no threads, no futures, no event loop.
Dynamo owns all the *driving* — the dedicated actor thread, cross-request
coalescing, the embeds splice, and the lifecycle — via ``ThreadedMicroBatcher``
(the generic cross-request batcher) and ``AsyncVisionEncoder`` (the async
request-API glue). This module defines only the contract those drivers call.

The encoder runs in the **same process** as the aggregated vLLM worker (no
separate encode worker, no NIXL transfer): it turns image inputs into the
visual-token embeddings for each image, and Dynamo splices those embeds into a
mixed ``EmbedsPrompt`` at the placeholder positions (see
``embed_assembler.build_mixed_embeds``) for a text-only LM.

Division of labour (author vs. Dynamo):

- ``build(model_id)`` — **actor thread, once.** Load weights / tokenizer; warm up
  to peak; if ``buckets`` is set (once CUDA-graph batching is supported), capture
  one CUDA graph per rung here so it is bound to the thread that later replays it
  in ``forward_batch``. Pick the device yourself (the worker pins it via
  ``CUDA_VISIBLE_DEVICES``, so ``"cuda"`` / the current device is correct).
- ``preprocess(raw) -> Preprocessed{item, cost}`` — **off the actor thread,
  concurrent.** Deterministic, thread-safe, CUDA-free (fetch / resize / patchify
  on CPU/pinned memory). ``cost`` is a **scalar** — how much the item adds toward
  ``max_batch_cost`` (e.g. its visual-token count). Raise to reject a bad input —
  it fails only that image, before any GPU work.
- ``forward_batch(items, target_bucket=None) -> list[torch.Tensor]`` — **actor
  thread, serialized.** ``items`` are a cost-bounded batch (summed ``cost`` within
  the budget). Fence (stream event + sync) and **copy outputs to CPU** before
  returning, so results are safe to consume from another thread and splice
  directly. Returns one ``(n_visual_tokens, lm_hidden_dim)`` **CPU** tensor per
  item, in input order. ``target_bucket`` is reserved for CUDA-graph batching,
  once supported (the ladder rung to pad to); it is ``None`` until then.
- ``close()`` — actor thread, on teardown. Release any thread-affine resources.

Attributes read **once at setup** (never per-request):

- ``image_token_id`` — the token id marking image positions in the prompt;
  **hardcode it for your model** (e.g. ``151655`` for Qwen3-VL's ``<|image_pad|>``).
  Dynamo uses it to locate each image span for the splice.
- ``max_batch_cost`` — the scalar dispatch ceiling the batcher packs up to; a
  *chosen* budget (a token budget when ``cost`` is a token count). ``None`` (the
  default) ⇒ **pass-through**: no cap (the author owns sizing).
- ``buckets`` — sorted graph ladder, forward-compatible (unused until CUDA-graph
  batching is supported). ``None``/empty ⇒ eager.

Batching is **one-dimensional**: Dynamo packs by scalar ``cost`` up to
``max_batch_cost`` and never inspects item shape — the author owns any
shape/padding concerns inside ``forward_batch``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Sequence, TypeVar

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


class VisionEncoderBackend(ABC, Generic[RawT, ItemT]):
    """Author-written, in-process vision encoder contract.

    A pure policy + compute backend — no threads, no futures. Dynamo drives it
    on a dedicated actor thread (``ThreadedMicroBatcher``) and exposes the async
    request API (``AsyncVisionEncoder``). Subclasses implement ``build`` /
    ``preprocess`` / ``forward_batch``, and set ``image_token_id`` (and
    ``max_batch_cost`` / ``buckets`` as needed).
    """

    #: Image placeholder token id — **hardcode it for your model** (e.g. ``151655``
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

    # ---- subclass contract -------------------------------------------------

    @abstractmethod
    def build(self, model_id: str) -> None:
        """Load weights / tokenizer, warm up, capture graphs (actor thread, once).

        Any CUDA graph captured here is bound to the thread that later replays it.
        Pick the device yourself — the worker pins it via ``CUDA_VISIBLE_DEVICES``,
        so ``"cuda"`` / the current device is correct. All CUDA init happens here.
        """
        ...

    @abstractmethod
    def preprocess(self, raw: RawT) -> Preprocessed[ItemT]:
        """Turn a raw input into a ``Preprocessed`` item (off the actor thread).

        Runs concurrently on a preprocess pool, so it is the place for image
        fetch + HF processing. Must be deterministic, thread-safe, and CUDA-free.
        Raise to reject a bad input — it fails only that image, before submit.
        """
        ...

    @abstractmethod
    def forward_batch(
        self, items: List[ItemT], target_bucket: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Encode one cost-bounded batch (actor thread); one tensor per item, in order.

        Fence (stream event + sync) and **copy outputs to CPU** before returning,
        so results are safe to consume from another thread and splice directly.
        Return one ``(n_visual_tokens, lm_hidden_dim)`` **CPU** tensor per item, in
        input order. ``target_bucket`` is reserved for CUDA-graph batching, once
        supported (the ladder rung to pad to), and is ``None`` until then.
        """
        ...

    def close(self) -> None:
        """Release thread-affine resources on teardown (actor thread). No-op by
        default; override to free graphs / pools / weights."""
        return None
