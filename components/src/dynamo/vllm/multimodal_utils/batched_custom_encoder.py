# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable, in-process vision encoder for the aggregated vLLM worker.

``BatchedCustomEncoder`` is the single base authors subclass to plug a custom
vision encoder into Dynamo. It is the *contract*; the *mechanism* (dedicated
thread + coalescing micro-batcher) lives in ``ThreadedMicroBatcher``, which this
class wires up in ``load()``.

The encoder runs in the same process as the vLLM aggregated worker (no separate
encode worker, no NIXL transfer): it encodes images and projects them to the LM
hidden dim, and Dynamo splices those embeds into a mixed ``EmbedsPrompt`` at the
placeholder positions (see ``embed_assembler.build_mixed_embeds``).

Subclasses implement:
- ``build(model_id, device)`` — load weights / tokenizer and (optionally)
  ``torch.compile`` + warm up. Runs **on the batcher thread**, so a CUDA graph
  captured here is replayed on the same thread.
- ``forward_batch(items)`` — the batched forward; one ``(n_visual_tokens,
  lm_hidden_dim)`` tensor per item. Receives a coalesced, single-``bucket_key``
  batch whose summed ``cost`` is within the budget; pad to a captured CUDA-graph
  shape here (the encoder owns the model's shapes).
- ``get_image_placeholder_token_id()`` — Qwen-family encoders mix in
  ``QwenPlaceholderMixin`` to get this for free.

Optional batching policy (defaults reduce to count-based, single-bucket):
``preprocess(url)`` (URL → item, run off the batcher thread), ``cost(item)``
(e.g. visual-token count), ``bucket_key(item)`` (only same-shape items batch
together), and ``max_batch_cost`` (the budget — a token budget when ``cost`` is
token count, else falls back to ``max_batch_size`` as an item count).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Hashable, List, Optional

import torch

from dynamo.vllm.multimodal_utils.threaded_micro_batcher import ThreadedMicroBatcher

# Image placeholder token *string* for the Qwen family. The numeric id is always
# resolved from the encoder's tokenizer — the same string maps to different ids
# across versions (151655 for Qwen3-VL, 248056 for Qwen3.5), which is why the
# tokenizer, not a static id table, is authoritative.
QWEN_IMAGE_PLACEHOLDER_TOKEN = "<|image_pad|>"


def placeholder_token_id_from_tokenizer(
    tokenizer: object,
    token: str,
) -> Optional[int]:
    """Resolve a placeholder token string to its ID via a loaded tokenizer.

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

    Orthogonal to execution: mix it into a ``BatchedCustomEncoder`` for any
    Qwen-family model (Qwen2-VL / Qwen3-VL / Qwen3.5) so the subclass only needs
    to assign ``self.tokenizer`` in ``build()``. Place it before the encoder base
    in the MRO: ``class MyEnc(QwenPlaceholderMixin, BatchedCustomEncoder)``.
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


class BatchedCustomEncoder(ABC):
    """In-process image encoder contract, backed by a ``ThreadedMicroBatcher``.

    The worker calls ``load`` once at startup and ``await``s ``encode`` per
    request; ``shutdown`` on teardown. Subclasses implement ``build`` /
    ``forward_batch`` / ``get_image_placeholder_token_id`` (and may override the
    optional batching policy below).
    """

    #: Item-count budget per batch when ``cost`` is the default (1 per item).
    max_batch_size: int = 8
    #: Window to wait for more items after the first arrives, in ms.
    max_wait_ms: float = 5.0
    #: Explicit cost budget per batch (e.g. a token budget). ``None`` →
    #: ``max_batch_size`` (count-based).
    max_batch_cost: Optional[int] = None

    # ---- subclass contract -------------------------------------------------

    @abstractmethod
    def build(self, model_id: str, device: str) -> None:
        """Load weights / tokenizer and (optionally) compile + warm up.

        Runs on the batcher thread, so any CUDA graph captured here is bound to
        the thread that later replays it in ``forward_batch``.
        """
        ...

    @abstractmethod
    def forward_batch(self, items: List[object]) -> List[torch.Tensor]:
        """Encode one coalesced batch, returning one tensor per item, in order.

        Runs on the batcher thread, serialized. ``items`` share a ``bucket_key``
        and their summed ``cost`` is within the budget. Return one
        ``(n_visual_tokens, lm_hidden_dim)`` tensor per item; pad to a captured
        CUDA-graph shape here if using graphs.
        """
        ...

    @abstractmethod
    def get_image_placeholder_token_id(self) -> int:
        """Return the token ID marking image positions in the prompt.

        Dynamo uses it to locate the image span and splice in the encoder
        tensors. Qwen-family encoders inherit this from ``QwenPlaceholderMixin``.

        Raises:
            ValueError: if no valid id can be resolved (surfaced at startup by
                ``load`` via ``validate``).
        """
        ...

    # ---- optional batching policy (defaults → count-based, single bucket) --

    def preprocess(self, image_url: str) -> object:
        """Turn an image URL into the item ``forward_batch`` consumes.

        Runs off the batcher thread (concurrently), so it is the place for image
        fetch + HF processing. Default: identity (the URL is the item).
        """
        return image_url

    def cost(self, item: object) -> int:
        """Batching cost of an item (default 1 → count). Override with e.g. the
        item's visual-token count for a token budget."""
        return 1

    def bucket_key(self, item: object) -> Hashable:
        """Items with different keys never share a batch (default: one bucket).
        Override to group by shape so a CUDA graph can replay."""
        return None

    # ---- runtime (provided) ------------------------------------------------

    def load(self, model_id: str, device: str) -> None:
        """Start the batcher (running ``build`` on its thread) and fail fast.

        Re-raises any build error, then ``validate``s the placeholder id so a
        misconfigured encoder errors at startup instead of on the first request.
        """
        self._batcher: ThreadedMicroBatcher = ThreadedMicroBatcher(
            self.forward_batch,
            max_batch_cost=(
                self.max_batch_cost
                if self.max_batch_cost is not None
                else self.max_batch_size
            ),
            max_wait_ms=self.max_wait_ms,
            cost=self.cost,
            bucket_key=self.bucket_key,
            on_start=lambda: self.build(model_id, device),
            name="batched-custom-encoder",
        )
        self._batcher.start()
        try:
            self.validate()
        except BaseException:
            self._batcher.shutdown()
            raise

    def validate(self) -> None:
        """Fail-fast checks run by ``load`` after ``build`` (resolve the
        placeholder id once). Subclasses may override (call ``super().validate()``)."""
        self.get_image_placeholder_token_id()

    async def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Preprocess each URL (off the batcher thread) and submit for batched encode."""
        batcher = getattr(self, "_batcher", None)
        if batcher is None:
            raise RuntimeError("BatchedCustomEncoder.encode() called before load()")
        items = await asyncio.gather(
            *(asyncio.to_thread(self.preprocess, url) for url in image_urls)
        )
        return await batcher.submit(list(items))

    def shutdown(self) -> None:
        """Stop the batcher thread. Safe before ``load`` and idempotent."""
        batcher = getattr(self, "_batcher", None)
        if batcher is not None:
            batcher.shutdown()
