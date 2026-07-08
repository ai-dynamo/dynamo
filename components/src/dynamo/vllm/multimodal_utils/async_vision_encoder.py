# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serial async glue between the worker's event loop and a ``VisionEncoderBackend``.

This is the **eager** glue: it proves the splice path end to end with a
**direct call — no micro-batcher**. Per request it preprocesses the images off the
event loop, enforces request-level atomicity, and runs the author's
``forward_batch`` on a single dedicated **actor thread**, serialized — there is no
cross-request coalescing. A follow-up swaps this body for a
``ThreadedMicroBatcher`` (cross-request batching); the public surface
(``load`` / ``encode`` / ``get_image_placeholder_token_id`` / ``shutdown``) is
identical, so the worker integration does not change.

Why a single actor thread (not ``asyncio.to_thread``): build and every
``forward_batch`` run on the **same** thread, so an author that captures a CUDA
graph in ``build`` can replay it from ``forward_batch`` — the affinity the batched
version also guarantees. ``max_workers=1`` serializes forwards (FIFO).

Request-level atomicity: a gather-barrier sits between preprocess and
the forward — ``encode`` waits for *every* image's preprocess to settle and runs
the forward only if **all** succeed; on any failure it does no GPU work and raises
the request-level error, so a text-only LM never sees a partial result.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Generic, List

import torch

from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    ItemT,
    Preprocessed,
    RawT,
    VisionEncoderBackend,
)

logger = logging.getLogger(__name__)


class AsyncVisionEncoder(Generic[RawT, ItemT]):
    """Drive a ``VisionEncoderBackend`` from the async request path, serially.

    The worker calls ``load`` once at startup and ``await``s ``encode`` per
    request; ``shutdown`` on teardown. All model knowledge lives in ``backend``;
    this class owns the optional preprocess pool, the request-atomicity barrier,
    and the single actor thread that runs ``build`` / ``forward_batch`` / ``close``.

    ``preprocess_concurrency`` (backend-declared, constructor-overridable) is not
    just a pool size — it gates the whole preprocess **phase**: ``0`` means
    ``preprocess`` is never called and raws go straight to ``forward_batch``. A
    backend that overrides ``preprocess`` while the effective concurrency is ``0``
    is rejected at construction (the override would silently never run).
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
        self._actor: ThreadPoolExecutor | None = None  # build + every forward
        self._pool: ThreadPoolExecutor | None = None  # off-loop preprocess (or None)

    # ---- lifecycle ---------------------------------------------------------

    def load(self, model_id: str) -> None:
        """Run ``backend.build`` on the actor thread and fail fast.

        Re-raises any build error, then ``validate``s the hardcoded image token id
        so a misconfigured encoder errors at startup. Single-shot: a second
        ``load()`` raises rather than orphaning the first actor thread and model.
        """
        if self._actor is not None:
            raise RuntimeError("AsyncVisionEncoder.load() called twice")
        try:
            # One actor thread so build + every forward share a thread; a single
            # worker also serializes forwards (FIFO) — no cross-request batching.
            self._actor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=f"{self._name}-actor"
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
            self._actor.submit(self._backend.build, model_id).result()
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
        run a single serial forward.

        With no preprocess pool (``preprocess_concurrency == 0``) raws go straight
        to ``forward_batch`` (the backend folds any prep in there). Returns one
        ``(n_visual_tokens, lm_hidden_dim)`` CPU tensor per raw input, in order.
        Raises if any image's preprocess fails (no GPU work) or if the forward
        fails.
        """
        if self._actor is None:
            raise RuntimeError("AsyncVisionEncoder.encode() called before load()")
        if not raws:
            return []
        loop = asyncio.get_running_loop()
        if self._pool is None:
            # No preprocess phase: raw IS the item. No barrier needed — the
            # single forward is already all-or-nothing for the request.
            items: List[ItemT] = list(raws)  # type: ignore[arg-type]  # ItemT==RawT
        else:
            # Request-atomicity barrier: preprocess all images concurrently, wait
            # for EVERY one to settle, run the forward only if all succeeded.
            # return_exceptions=True makes the gather a true barrier (no short-circuit).
            tasks = [
                loop.run_in_executor(self._pool, self._backend.preprocess, raw)
                for raw in raws
            ]
            settled = await asyncio.gather(*tasks, return_exceptions=True)
            errors = [r for r in settled if isinstance(r, BaseException)]
            if errors:
                raise errors[0]
            preprocessed: List[Preprocessed] = list(settled)  # type: ignore[arg-type]
            items = [p.item for p in preprocessed]
        # Direct, serialized forward on the actor thread (eager; target_bucket
        # defaults to None — there is no graph ladder until CUDA-graph batching
        # is supported).
        return await loop.run_in_executor(
            self._actor, self._backend.forward_batch, items
        )

    def shutdown(self) -> None:
        """Run ``backend.close`` on the actor thread, then stop both pools. Safe
        before ``load`` and idempotent."""
        if self._actor is not None:
            try:
                self._actor.submit(self._backend.close).result(timeout=10)
            except BaseException:  # noqa: BLE001 — teardown best-effort
                logger.exception(
                    "AsyncVisionEncoder(%s): backend.close raised during teardown",
                    self._name,
                )
            self._actor.shutdown(wait=False)
        if self._pool is not None:
            self._pool.shutdown(wait=False)
