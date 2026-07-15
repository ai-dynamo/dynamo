# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Async glue between the worker's event loop and a ``VisionEncoderBackend``.

``AsyncVisionEncoder`` is the **Dynamo-owned** layer the worker talks to. It
turns the author's synchronous, thread-affine backend into an awaitable
``encode(raws) -> list[encoded_media]`` by:

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
  by cost and runs ``backend.forward_batch`` on the single actor thread; and
- for typed backends, correlation-tagging each physical batch, restoring input
  order, validating the declared schema, and cloning owned CPU results before
  positional scatter.

The backend's ``build`` runs on the batcher's actor thread (so a CUDA graph it
captures is replayed on the same thread) and its ``close`` runs there at
teardown. ``load`` fails fast: it re-raises a build error and resolves the image
placeholder id once, so a misconfigured encoder errors at startup, not on the
first request.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic, List
from uuid import uuid4

import torch

from dynamo.vllm.multimodal_utils.custom_encoder_adapter import (
    reconcile_and_canonicalize,
)
from dynamo.vllm.multimodal_utils.threaded_micro_batcher import ThreadedMicroBatcher
from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    BackendEncodingSpecV1,
    EncodedMediaV1,
    ForwardItemV1,
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
        encoding_spec = getattr(backend, "encoding_spec", None)
        if encoding_spec is not None and not isinstance(
            encoding_spec, BackendEncodingSpecV1
        ):
            raise TypeError(
                "VisionEncoderBackend.encoding_spec must be a "
                f"BackendEncodingSpecV1 or None, got {encoding_spec!r}"
            )
        self._encoding_spec = encoding_spec
        self._preprocess_concurrency = conc
        self._name = name
        self._batcher: ThreadedMicroBatcher | None = None
        self._pool: ThreadPoolExecutor | None = None
        self._closing = False

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
                self._forward_batch,
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
        """Fail fast on the legacy linear placeholder contract."""
        if (
            self._encoding_spec is not None
            and self._encoding_spec.adapter_abi != "linear-rows-v1"
        ):
            return
        tid = getattr(self._backend, "image_token_id", None)
        if not isinstance(tid, int) or isinstance(tid, bool):
            raise ValueError(
                "VisionEncoderBackend.image_token_id must be a hardcoded int (the "
                f"image placeholder token id); got {tid!r}"
            )

    def get_image_placeholder_token_id(self) -> int:
        """The token id marking image positions (the backend's hardcoded value)."""
        if (
            self._encoding_spec is not None
            and self._encoding_spec.adapter_abi != "linear-rows-v1"
        ):
            raise RuntimeError(
                "native multimodal CustomEncoder output does not use the legacy "
                "image placeholder splice"
            )
        return self._backend.image_token_id

    @property
    def encoding_spec(self) -> BackendEncodingSpecV1 | None:
        """The backend's setup-time encoded-media contract, if declared."""
        return self._encoding_spec

    def _forward_batch(self, items: list[ItemT]) -> list[torch.Tensor | EncodedMediaV1]:
        """Run one physical batch and reconcile typed results on the actor."""
        spec = self._encoding_spec
        if spec is None:
            return self._backend.forward_batch(items)

        tagged_items = [
            ForwardItemV1(correlation_id=uuid4().bytes, item=item) for item in items
        ]
        returned: Any = self._backend.forward_batch(tagged_items)  # type: ignore[arg-type]
        if not isinstance(returned, list):
            raise TypeError("typed CustomEncoder forward_batch must return a list")
        return reconcile_and_canonicalize(spec, tagged_items, returned)

    # ---- request path ------------------------------------------------------

    async def encode(self, raws: List[RawT]) -> List[torch.Tensor | EncodedMediaV1]:
        """Optionally preprocess (off-loop, with a request-atomicity barrier) then
        batched-encode.

        With no preprocess pool (``preprocess_concurrency == 0``) raws go straight
        to the batcher (the backend folds any prep into ``forward_batch``). Returns
        one legacy tensor or typed encoded-media value per raw input, in order.
        Raises if any image's preprocess fails (submitting nothing), correlation
        reconciliation fails, or the batched forward fails.
        """
        batcher = self._batcher
        if batcher is None or self._closing:
            raise RuntimeError("AsyncVisionEncoder.encode() called before load()")
        if not raws:
            return []
        if self._pool is None:
            # No preprocess phase: raw IS the item (cost defaults to 1). No
            # barrier needed — the batched forward is all-or-nothing per request.
            return await batcher.submit(list(raws))  # type: ignore[arg-type]
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
        for result in settled:
            if isinstance(result, BaseException):
                # Fail the whole request atomically; no item was submitted (no GPU
                # work). Surface the first failure, in order.
                raise result
        if self._closing:
            raise RuntimeError("AsyncVisionEncoder shut down during preprocess")
        # No exception above ⇒ every settled entry is a Preprocessed. Alias the
        # list gather() already returned rather than copying it.
        preprocessed: List[Preprocessed] = settled  # type: ignore[assignment]
        items = [p.item for p in preprocessed]
        costs = [p.cost for p in preprocessed]
        return await batcher.submit(items, costs)

    def shutdown(self) -> None:
        """Stop the actor thread (running ``backend.close`` on it) and the
        preprocess pool. Safe before ``load`` and idempotent."""
        # Close admission before waiting for preprocessing. An encode call that
        # was already in the pool rechecks this flag before submitting GPU work.
        self._closing = True
        # Detach both resources before teardown so repeated cleanup is a no-op,
        # including if teardown itself raises.
        batcher = self._batcher
        pool = self._pool
        self._batcher = None
        self._pool = None

        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        if batcher is not None:
            batcher.shutdown()  # runs backend.close() on the actor thread
