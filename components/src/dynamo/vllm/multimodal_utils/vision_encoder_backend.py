# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The author-written contract for a pluggable in-process vision encoder.

``VisionEncoderBackend`` is the **single surface an encoder author implements**.
It is a pure policy + compute backend: no threads, no futures, no event loop.
Dynamo owns all the *driving* — the dedicated actor thread, cross-request
coalescing, the embeds splice, and the lifecycle — via ``ThreadedMicroBatcher``
(L1, generic) and ``AsyncVisionEncoder`` (L3, glue). This module is L2.

The encoder runs in the **same process** as the aggregated vLLM worker (no
separate encode worker, no NIXL transfer): it turns image inputs into the
visual-token embeddings for each image, and Dynamo splices those embeds into a
mixed ``EmbedsPrompt`` at the placeholder positions (see
``embed_assembler.build_mixed_embeds``) for a text-only LM.

Division of labour (author vs. Dynamo):

- ``build(model_id, device)`` — **actor thread, once.** Load weights / tokenizer;
  warm up to peak; if ``buckets`` is set, capture one CUDA graph per rung here so
  the graph is bound to the thread that later replays it in ``forward_batch``.
- ``preprocess(raw) -> Preprocessed{item, cost, bucket_key}`` — **off the actor
  thread, concurrent.** Deterministic, thread-safe, CUDA-free (fetch / resize /
  patchify on CPU/pinned memory). ``cost`` is how much the item adds toward
  ``max_batch_cost``; ``bucket_key`` partitions shape-compatible items. Raise to
  reject a bad input — it fails only that image, before any GPU work.
- ``forward_batch(items, target_bucket=None) -> list[torch.Tensor]`` — **actor
  thread, serialized.** ``items`` share a ``bucket_key`` and their summed ``cost``
  is within the budget. ``target_bucket=None`` ⇒ run the packed batch eager; else
  **pad** ``sum(cost)`` up to ``target_bucket``, replay that rung's graph, and
  slice the result back per item. Fence + copy outputs to stable storage before
  returning. Returns one ``(n_visual_tokens, lm_hidden_dim)`` tensor per item, in
  input order.
- ``close()`` — actor thread, on teardown. Release any thread-affine resources.
- ``get_image_placeholder_token_id()`` — the token id Dynamo uses to locate image
  spans for the splice. Qwen-family backends mix in ``QwenPlaceholderMixin``.

Attributes read **once at setup** (never per-request):

- ``max_batch_cost`` — the dispatch ceiling the batcher packs up to (a *chosen*
  budget, like vLLM's ``max_num_batched_tokens`` — not discovered).
- ``buckets`` — the sorted graph ladder (the captured rungs). ``None``/empty ⇒
  **eager** (no graphs); the batcher passes ``target_bucket=None``. Non-empty ⇒
  graphed; the batcher rounds the packed cost up to the nearest rung. Adding
  graphs later does not change the signature — the eager author ignores
  ``buckets`` entirely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Hashable, List, Optional, Sequence, TypeVar

import torch

RawT = TypeVar("RawT")  # raw input the author preprocesses (e.g. an image URL)
ItemT = TypeVar("ItemT")  # opaque payload preprocess() hands to forward_batch()

# Image placeholder token *string* for the Qwen family. The numeric id is always
# resolved from the encoder's tokenizer — the same string maps to different ids
# across versions (151655 for Qwen3-VL, 248056 for Qwen3.5), which is why the
# tokenizer, not a static id table, is authoritative.
QWEN_IMAGE_PLACEHOLDER_TOKEN = "<|image_pad|>"


@dataclass(frozen=True)
class Preprocessed(Generic[ItemT]):
    """The result of ``preprocess(raw)``: an opaque item plus its batching labels.

    ``cost`` and ``bucket_key`` are computed **once, off the actor thread**, so
    the batcher never evaluates model policy (it stays torch-free).

    Attributes:
        item: Opaque payload passed verbatim to ``forward_batch``.
        cost: Token/feature size, ``>= 1``; packs toward ``max_batch_cost``.
            Defaults to ``1`` and is **ignored** when ``max_batch_cost`` is
            ``None`` (pass-through), so a pass-through author can omit it.
        bucket_key: Shape-compatibility partition — items with different keys
            never share a ``forward_batch`` call (so a captured graph can replay).
            Defaults to ``None`` (a single shared bucket).
    """

    item: ItemT
    cost: int = 1
    bucket_key: Hashable = None


def placeholder_token_id_from_tokenizer(
    tokenizer: object,
    token: str,
) -> Optional[int]:
    """Resolve a placeholder token string to its id via a loaded tokenizer.

    Returns ``None`` if the tokenizer does not define ``token`` (it maps to the
    unknown-token id), so callers can raise a clear error.

    Args:
        tokenizer: A loaded tokenizer exposing ``convert_tokens_to_ids``.
        token: The placeholder token string, e.g. ``QWEN_IMAGE_PLACEHOLDER_TOKEN``.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    tid = convert(token)
    unk = getattr(tokenizer, "unk_token_id", None)
    if tid is None or (unk is not None and tid == unk):
        return None
    return int(tid)


class QwenPlaceholderMixin:
    """Mixin resolving the image placeholder id from ``self.tokenizer`` (Qwen).

    Orthogonal to execution: mix it into a ``VisionEncoderBackend`` for any
    Qwen-family model (Qwen2-VL / Qwen3-VL / Qwen3.5) so the subclass only needs
    to assign ``self.tokenizer`` in ``build()``. Place it before the backend base
    in the MRO: ``class MyEnc(QwenPlaceholderMixin, VisionEncoderBackend)``.
    """

    def get_image_placeholder_token_id(self) -> int:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "self.tokenizer is not set; assign the model tokenizer in "
                "build() so the Qwen image placeholder id can be resolved."
            )
        tid = placeholder_token_id_from_tokenizer(
            tokenizer, QWEN_IMAGE_PLACEHOLDER_TOKEN
        )
        if tid is None:
            raise ValueError(
                f"tokenizer does not define placeholder token "
                f"{QWEN_IMAGE_PLACEHOLDER_TOKEN!r}; is this a Qwen-family model?"
            )
        return tid


class VisionEncoderBackend(ABC, Generic[RawT, ItemT]):
    """Author-written, in-process vision encoder contract (L2).

    A pure policy + compute backend — no threads, no futures. Dynamo drives it
    on a dedicated actor thread (``ThreadedMicroBatcher``) and exposes the async
    request API (``AsyncVisionEncoder``). Subclasses implement ``build`` /
    ``preprocess`` / ``forward_batch`` / ``get_image_placeholder_token_id`` and
    set ``max_batch_cost`` (and ``buckets`` if graphed).
    """

    #: Dispatch ceiling: the batcher packs items up to this summed ``cost`` per
    #: ``forward_batch`` call (a chosen budget — a token budget when ``cost`` is a
    #: token count). ``None`` (the default) ⇒ **pass-through**: no cap — every
    #: same-``bucket_key`` item the batcher drains in one iteration is handed to a
    #: single ``forward_batch`` (the author owns sizing; ``cost`` is ignored).
    #: When graphed, leave it ``None`` to derive ``max(buckets)``, or set it
    #: explicitly (must be ``<= max(buckets)``).
    max_batch_cost: Optional[int] = None

    #: Sorted graph ladder (the captured rungs). ``None``/empty ⇒ eager (no
    #: graphs); the batcher passes ``target_bucket=None``. Non-empty ⇒ the batcher
    #: rounds the packed ``sum(cost)`` up to the nearest rung.
    buckets: Optional[Sequence[int]] = None

    # ---- subclass contract -------------------------------------------------

    @abstractmethod
    def build(self, model_id: str, device: str) -> None:
        """Load weights / tokenizer, warm up to peak, capture graphs (actor thread).

        Runs once, on the actor thread, before any ``forward_batch``. Any CUDA
        graph captured here is bound to the thread that later replays it. All
        CUDA initialization for the encoder happens here.
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
        """Encode one coalesced batch (actor thread); one tensor per item, in order.

        ``items`` share a ``bucket_key`` and their summed ``cost`` is within the
        budget. ``target_bucket=None`` ⇒ run the packed batch eager. Otherwise
        **pad** ``sum(cost)`` up to ``target_bucket``, replay that rung's captured
        graph, and slice outputs back per item. Fence (stream event + sync) and
        copy outputs to non-overwritten storage **before returning**, so results
        are safe to consume from another thread. Return one
        ``(n_visual_tokens, lm_hidden_dim)`` tensor per item, in input order.
        """
        ...

    @abstractmethod
    def get_image_placeholder_token_id(self) -> int:
        """Return the token id marking image positions in the prompt.

        Dynamo uses it to locate the image span and splice in the encoder
        tensors. Qwen-family backends inherit this from ``QwenPlaceholderMixin``.

        Raises:
            ValueError: if no valid id can be resolved (surfaced at startup by
                ``AsyncVisionEncoder.load`` via ``validate``).
        """
        ...

    def close(self) -> None:
        """Release thread-affine resources on teardown (actor thread). No-op by
        default; override to free graphs / pools / weights."""
        return None
