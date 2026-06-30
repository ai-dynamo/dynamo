# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async glue between the worker's event loop and a ``VisionEncoderBackend``.

``AsyncVisionEncoder`` is the **Dynamo-owned** layer the worker talks to. It
turns the author's synchronous, thread-affine backend into an awaitable
``encode(raws) -> list[tensor]`` by:

- running ``backend.preprocess`` **off the event loop** on a bounded
  ``ThreadPoolExecutor`` (CPU-heavy fetch / resize / patchify must not serialize
  on the GPU actor thread);
- enforcing **request-level atomicity**: a gather-barrier between preprocess
  and submit — ``encode`` waits for *every* image's preprocess to settle and only
  submits if **all** succeed; on any failure it submits nothing (zero GPU work)
  and raises the request-level error, so a text-only LM never sees a partial
  result;
- handing the preprocessed items (with their off-thread-computed scalar ``cost``)
  to a ``ThreadedMicroBatcher``, which coalesces across concurrent ``encode`` calls
  by cost and runs ``backend.forward_batch`` on the single actor thread.

The backend's ``build`` runs on the batcher's actor thread (so a CUDA graph it
captures is replayed on the same thread) and its ``close`` runs there at
teardown. ``load`` fails fast: it re-raises a build error and resolves the image
placeholder id once, so a misconfigured encoder errors at startup, not on the
first request.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, List, Optional

import torch

from dynamo.vllm.multimodal_utils.threaded_micro_batcher import ThreadedMicroBatcher
from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    ItemT,
    Preprocessed,
    RawT,
    VisionEncoderBackend,
)

logger = logging.getLogger(__name__)


class AsyncVisionEncoder(Generic[RawT, ItemT]):
    """Drive a ``VisionEncoderBackend`` from the worker's async request path.

    The worker calls ``load`` once at startup and ``await``s ``encode`` per
    request; ``shutdown`` on teardown. All model knowledge lives in ``backend``;
    this class owns the preprocess pool, the A5 barrier, and the micro-batcher.

    Args:
        backend: The author-written ``VisionEncoderBackend``.
        preprocess_concurrency: Worker threads for off-loop ``preprocess``.
        name: Base name for the actor thread / preprocess pool.
    """

    def __init__(
        self,
        backend: VisionEncoderBackend[RawT, ItemT],
        *,
        preprocess_concurrency: int = 4,
        name: str = "vision-encoder",
    ) -> None:
        if preprocess_concurrency < 1:
            raise ValueError("preprocess_concurrency must be >= 1")
        self._backend = backend
        self._preprocess_concurrency = preprocess_concurrency
        self._name = name
        self._batcher: Optional[ThreadedMicroBatcher] = None
        self._pool: Optional[ThreadPoolExecutor] = None

    # ---- lifecycle ---------------------------------------------------------

    def load(self, model_id: str) -> None:
        """Start the actor thread (running ``backend.build`` on it) and fail fast.

        Re-raises any build error, then ``validate``s the placeholder id so a
        misconfigured encoder errors at startup instead of on the first request.
        Single-shot: a second ``load()`` raises rather than orphaning the first
        batcher's (non-daemon) worker thread and model.
        """
        if self._batcher is not None or self._pool is not None:
            raise RuntimeError("AsyncVisionEncoder.load() called twice")
        # Construct the pool + batcher INSIDE the try so a constructor failure
        # (e.g. a backend exposing a max_batch_cost the batcher rejects) still
        # reaps the pool via shutdown() instead of leaking it. shutdown() is
        # None-safe on the not-yet-assigned member.
        try:
            self._pool = ThreadPoolExecutor(
                max_workers=self._preprocess_concurrency,
                thread_name_prefix=f"{self._name}-pre",
            )
            self._batcher = ThreadedMicroBatcher(
                self._backend.forward_batch,
                max_batch_cost=self._backend.max_batch_cost,
                on_start=lambda: self._backend.build(model_id),
                on_stop=self._backend.close,
                name=self._name,
            )
            self._batcher.start()  # runs backend.build() on the actor thread
            self.validate()
        except BaseException:
            self.shutdown()
            raise

    def validate(self) -> None:
        """Fail-fast check run by ``load`` after ``build``: the author hardcoded a
        usable ``image_token_id``."""
        tid = getattr(self._backend, "image_token_id", None)
        if not isinstance(tid, int) or isinstance(tid, bool):
            raise ValueError(
                "VisionEncoderBackend.image_token_id must be a hardcoded int (the "
                f"image placeholder token id); got {tid!r}"
            )

    def get_image_placeholder_token_id(self) -> int:
        """The token id marking image positions (the backend's hardcoded value)."""
        return self._backend.image_token_id

    # ---- request path ------------------------------------------------------

    async def encode(self, raws: List[RawT]) -> List[torch.Tensor]:
        """Preprocess (off-loop, A5 barrier) then batched-encode; all-or-nothing.

        Returns one ``(n_visual_tokens, lm_hidden_dim)`` tensor per raw input, in
        order. Raises if any image's preprocess fails (submitting nothing) or if
        the batched forward fails.
        """
        if self._batcher is None or self._pool is None:
            raise RuntimeError("AsyncVisionEncoder.encode() called before load()")
        if not raws:
            return []
        loop = asyncio.get_running_loop()
        # A5 barrier: preprocess all images concurrently, wait for EVERY one to
        # settle, and submit only if all succeeded. return_exceptions=True makes
        # the gather a true barrier (it never short-circuits), so a failed sibling
        # cannot leave a half-submitted request — we submit nothing on any error.
        tasks = [
            loop.run_in_executor(self._pool, self._backend.preprocess, raw)
            for raw in raws
        ]
        settled = await asyncio.gather(*tasks, return_exceptions=True)
        errors = [r for r in settled if isinstance(r, BaseException)]
        if errors:
            # Fail the whole request atomically; no item was submitted (no GPU
            # work). Surface the first failure.
            raise errors[0]
        preprocessed: List[Preprocessed] = list(settled)  # type: ignore[arg-type]
        items = [p.item for p in preprocessed]
        costs = [p.cost for p in preprocessed]
        return await self._batcher.submit(items, costs)

    def shutdown(self) -> None:
        """Stop the actor thread (running ``backend.close`` on it) and the
        preprocess pool. Safe before ``load`` and idempotent."""
        if self._batcher is not None:
            self._batcher.shutdown()  # runs backend.close() on the actor thread
        if self._pool is not None:
            self._pool.shutdown(wait=False)
