# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, List

import torch

from dynamo.vllm.multimodal_utils.threaded_micro_batcher import ThreadedMicroBatcher
from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    ItemT,
    Preprocessed,
    RawT,
    VisionEncoderBackend,
)


class AsyncVisionEncoder(Generic[RawT, ItemT]):
    """Drive a ``VisionEncoderBackend`` from the worker's async request path.

    The worker calls ``load`` once at startup and ``await``s ``encode`` per
    request; ``shutdown`` on teardown. All model knowledge lives in ``backend``;
    this class owns the optional preprocess pool, the request-atomicity barrier,
    and the micro-batcher.

    ``preprocess_concurrency`` (backend-declared, constructor-overridable) is not
    just a pool size — it gates the whole preprocess **phase**: ``0`` means
    ``preprocess`` is never called and raws go straight to ``forward_batch``. A
    backend that overrides ``preprocess`` while the effective concurrency is ``0``
    is rejected at construction (the override would silently never run).

    Args:
        backend: The author-written ``VisionEncoderBackend``.
        preprocess_concurrency: Off-loop ``preprocess`` pool size; ``None`` ⇒ use
            the backend's value (default 0 ⇒ no pool / passthrough).
        name: Base name for the actor thread / preprocess pool.
    """

    def __init__(
        self,
        backend: VisionEncoderBackend[RawT, ItemT],
        *,
        preprocess_concurrency: int | None = None,
        name: str = "vision-encoder",
    ) -> None:
        # The backend declares whether it needs off-loop preprocessing; the
        # explicit arg is an override for tuning. Not just a pool size: 0 ⇒ no
        # preprocess phase at all — preprocess() is never called and raws pass
        # straight to forward_batch.
        conc = (
            backend.preprocess_concurrency
            if preprocess_concurrency is None
            else preprocess_concurrency
        )
        if conc < 0:
            raise ValueError("preprocess_concurrency must be >= 0")
        # Fail fast on the silent mismatch: an overridden preprocess that the
        # effective concurrency of 0 would skip forever.
        overrides_preprocess = (
            type(backend).preprocess is not VisionEncoderBackend.preprocess
        )
        if conc == 0 and overrides_preprocess:
            raise ValueError(
                f"{type(backend).__name__} overrides preprocess() but the "
                "effective preprocess_concurrency is 0, so preprocess would never "
                "run (raws go straight to forward_batch). Set "
                "preprocess_concurrency > 0 to enable the preprocess phase, or "
                "drop the override and do the prep inside forward_batch."
            )
        self._backend = backend
        self._preprocess_concurrency = conc
        self._name = name
        self._batcher: ThreadedMicroBatcher | None = None
        self._pool: ThreadPoolExecutor | None = None

    # ---- lifecycle ---------------------------------------------------------

    def load(self, model_id: str) -> None:
        """Start the actor thread (running ``backend.build`` on it) and fail fast.

        Re-raises any build error, then ``validate``s the placeholder id so a
        misconfigured encoder errors at startup instead of on the first request.
        Single-shot: a second ``load()`` raises rather than orphaning the first
        batcher's (non-daemon) worker thread and model.
        """
        if self._batcher is not None:
            raise RuntimeError("AsyncVisionEncoder.load() called twice")
        # Construct INSIDE the try so a later start()/build failure reaps the pool
        # via shutdown() (None-safe on the not-yet-assigned members). The batcher
        # is built before the pool so that a config it rejects (e.g. a backend
        # max_batch_cost < 1) raises from its ctor before any pool is spawned —
        # nothing to reap in that case.
        try:
            self._batcher = ThreadedMicroBatcher(
                self._backend.forward_batch,
                max_batch_cost=self._backend.max_batch_cost,
                on_start=lambda: self._backend.build(model_id),
                on_stop=self._backend.close,
                name=self._name,
            )
            # No pool when concurrency is 0 — preprocess is skipped (passthrough).
            self._pool = (
                ThreadPoolExecutor(
                    max_workers=self._preprocess_concurrency,
                    thread_name_prefix=f"{self._name}-pre",
                )
                if self._preprocess_concurrency > 0
                else None
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
        """Optionally preprocess (off-loop, with a request-atomicity barrier) then
        batched-encode.

        With no preprocess pool (``preprocess_concurrency == 0``) raws go straight
        to the batcher (the backend folds any prep into ``forward_batch``). Returns
        one ``(n_visual_tokens, lm_hidden_dim)`` tensor per raw input, in order.
        Raises if any image's preprocess fails (submitting nothing) or if the
        batched forward fails.
        """
        if self._batcher is None:
            raise RuntimeError("AsyncVisionEncoder.encode() called before load()")
        if not raws:
            return []
        if self._pool is None:
            # No preprocess phase: raw IS the item (cost defaults to 1). No
            # barrier needed — the batched forward is all-or-nothing per request.
            return await self._batcher.submit(list(raws))  # type: ignore[arg-type]
        loop = asyncio.get_running_loop()
        # Request-atomicity barrier: preprocess all images concurrently, wait for
        # EVERY one to settle, and submit only if all succeeded. return_exceptions=True makes
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
        # Detach both resources before teardown so repeated cleanup is a no-op,
        # including if teardown itself raises.
        batcher = self._batcher
        pool = self._pool
        self._batcher = None
        self._pool = None

        if batcher is not None:
            batcher.shutdown()  # runs backend.close() on the actor thread
        if pool is not None:
            pool.shutdown(wait=False)
