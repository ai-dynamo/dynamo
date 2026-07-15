# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The author-written contract for a pluggable in-process vision encoder.

``VisionEncoderBackend`` is the **single surface an encoder author implements**.
It is a pure policy + compute backend: no threads, no futures, no event loop.
Dynamo owns all the *driving* — the dedicated actor thread, cross-request
coalescing, engine-input adaptation, and the lifecycle — via ``ThreadedMicroBatcher``
(the generic cross-request batcher) and ``AsyncVisionEncoder`` (the async
request-API glue). This module defines only the contract those drivers call.

The encoder runs in the **same process** as the aggregated vLLM worker (no
separate encode worker, no NIXL transfer): it turns image inputs into the
visual-token embeddings for each image. Linear-position models use a mixed
``EmbedsPrompt``; metadata-dependent models such as Qwen2/2.5-VL use vLLM's
native external multimodal-embedding input so grid-driven M-RoPE is preserved.

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
- ``forward_batch(items, target_bucket=None)`` — **actor thread, serialized.**
  Legacy backends receive raw items and return one CPU tensor per item. Backends
  declaring ``encoding_spec`` receive ``ForwardItemV1`` values and return
  correlation-echoing ``EncodedMediaResultV1`` values. The driver validates and
  restores their input order, then freezes tensor storage on the actor thread.
- ``close()`` — actor thread, on teardown, including after a partially failed
  ``build``. Release any initialized thread-affine resources idempotently.

Attributes read **once at setup** (never per-request):

- ``image_token_id`` — for the linear route, the token id marking image positions;
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
- ``encoding_spec`` — optional typed result/adapter handshake. ``None`` keeps the
  legacy tensor route; Qwen2/2.5 backends declare
  ``vllm-qwen2-vl-external-v1`` and return projected rows plus ``grid_thw``.

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


@dataclass(frozen=True)
class BackendEncodingSpecV1:
    """Setup-time declaration of the backend's encoded-media contract.

    Backends that do not declare a spec retain the legacy ``list[Tensor]``
    contract. A backend that declares a spec receives correlation-tagged forward
    items and must return ``EncodedMediaResultV1`` values.
    """

    adapter_abi: Literal["linear-rows-v1", "vllm-qwen2-vl-external-v1"]
    producer_fingerprint: str
    expected_decoder_config_fingerprint: Optional[str]
    output_dtype: Literal["float16", "bfloat16", "float32"]
    hidden_size: int
    spatial_merge_size: Optional[int] = None


@dataclass(frozen=True)
class LinearRowsV1:
    """External media rows with ordinary one-dimensional positions."""

    rows: torch.Tensor


@dataclass(frozen=True)
class Qwen2VLImageEncodingV1:
    """One canonical Qwen2/2.5-VL image artifact.

    ``projected`` is the output *after* the vision projector/patch merger.
    ``grid_thw`` is the positive pre-spatial-merge patch grid. Rows are ordered
    in canonical ``(t, merged_h, merged_w)`` raster order; Qwen2.5 producers must
    apply the model's inverse window permutation before returning. Shape checks
    cannot detect a pre-inverse tensor, which would attach valid M-RoPE values to
    the wrong spatial patches while still producing fluent output.
    """

    projected: torch.Tensor
    grid_thw: tuple[int, int, int]


EncodedMediaV1: TypeAlias = LinearRowsV1 | Qwen2VLImageEncodingV1


@dataclass(frozen=True)
class ForwardItemV1(Generic[ItemT]):
    """One backend item tagged for order-independent result reconciliation."""

    correlation_id: bytes
    item: ItemT


@dataclass(frozen=True)
class EncodedMediaResultV1:
    """A typed media result echoing its input correlation identifier."""

    correlation_id: bytes
    media: EncodedMediaV1


class VisionEncoderBackend(ABC, Generic[RawT, ItemT]):
    """Author-written, in-process vision encoder contract.

    A pure policy + compute backend — no threads, no futures. Dynamo drives it
    on a dedicated actor thread (``ThreadedMicroBatcher``) and exposes the async
    request API (``AsyncVisionEncoder``). Subclasses implement ``build`` and
    ``forward_batch`` and set ``image_token_id``; ``preprocess`` (default identity
    passthrough), ``max_batch_cost``, ``buckets``, ``preprocess_concurrency``, and
    ``encoding_spec`` are overridden only as needed.
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

    #: Off-loop preprocess pool size Dynamo runs ``preprocess`` on. Not just a
    #: pool size — it gates the preprocess **phase**: ``0`` (the **default**) ⇒
    #: ``preprocess`` is never called and raws go straight to ``forward_batch``
    #: (``raw`` is the item; do any prep there). Set ``> 0`` (with an overridden
    #: ``preprocess``) to fetch / resize / patchify off the actor thread; overriding
    #: ``preprocess`` while leaving this at ``0`` fails fast at startup. Whether an
    #: encoder needs off-loop prep is a property of the encoder, so it lives here;
    #: the driver takes an optional override for tuning.
    preprocess_concurrency: int = 0

    #: Optional typed encoded-media contract. ``None`` preserves the original
    #: untagged ``list[Tensor]`` behavior for existing backends.
    encoding_spec: Optional[BackendEncodingSpecV1] = None

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
    ) -> List[torch.Tensor]:
        """Encode one cost-bounded batch on the actor thread.

        With no ``encoding_spec``, return one CPU tensor per input in order. With
        a spec, Dynamo passes ``ForwardItemV1`` values and requires one
        ``EncodedMediaResultV1`` per input; results may be returned in any order
        because correlation IDs are reconciled before positional scatter. Fence
        device work and copy outputs to CPU before returning.
        """
        ...

    def close(self) -> None:
        """Release initialized thread-affine resources on the actor thread.

        Called after normal serving and after a partially failed ``build``;
        overrides must be idempotent and tolerate incomplete initialization.
        """
        return None
