# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pluggable, in-process vision encoder for the aggregated vLLM worker.

``BatchedCustomEncoder`` is the single base authors subclass to plug a custom
vision encoder into Dynamo. It owns the execution runtime so the encoder author
does not have to:

- a **dedicated worker thread** that builds the model and runs every forward.
  Build and all forwards happen on that one thread, so a ``torch.compile(...,
  mode="reduce-overhead")`` CUDA graph (whose capture is bound to thread-local
  storage) is captured and replayed on the same thread — the affinity that an
  ``asyncio.to_thread`` thread pool cannot guarantee.
- a **coalescing micro-batcher**: concurrent ``encode()`` calls are merged
  (up to ``max_batch_size`` images, within a ``max_wait_ms`` window) into one
  ``forward_batch`` call, then results are scattered back to each caller —
  cross-request batching without the caller holding any lock.

The encoder runs in the same process as the vLLM aggregated worker (no separate
encode worker, no NIXL transfer): it encodes images and projects them to the LM
hidden dim, and Dynamo splices those embeds into a mixed ``EmbedsPrompt`` at the
placeholder positions (see ``embed_assembler.build_mixed_embeds``).

Subclasses implement three methods — ``build`` (load weights / compile on the
batcher thread), ``forward_batch`` (the batched forward), and
``get_image_placeholder_token_id`` (Qwen-family encoders mix in
``QwenPlaceholderMixin`` to get it for free). Everything else — thread lifecycle,
coalescing, the async-to-thread bridge, and error fan-out — is provided here.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

# Image placeholder token *string* for the Qwen family. The numeric id is always
# resolved from the encoder's tokenizer — the same string maps to different ids
# across versions (151655 for Qwen3-VL, 248056 for Qwen3.5), which is why the
# tokenizer, not a static id table, is authoritative.
QWEN_IMAGE_PLACEHOLDER_TOKEN = "<|image_pad|>"

# Sentinel pushed onto the request queue to stop the batcher thread.
_SHUTDOWN = object()


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


@dataclass
class _Pending:
    """One in-flight ``encode()`` call queued for the batcher thread."""

    image_urls: List[str]
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future


class BatchedCustomEncoder(ABC):
    """In-process image encoder with a dedicated thread + coalescing batcher.

    Subclasses implement ``build``, ``forward_batch``, and
    ``get_image_placeholder_token_id``; the worker calls ``load`` once at startup
    and ``await``s ``encode`` per request. See the module docstring for the
    execution model.
    """

    #: Max images coalesced into one ``forward_batch`` call.
    max_batch_size: int = 8
    #: Window to wait for more requests after the first arrives, in ms.
    max_wait_ms: float = 5.0
    #: Pad each ``forward_batch`` to exactly ``max_batch_size`` images (repeat the
    #: last, drop the extra outputs) so the batch dimension is static — required
    #: for a CUDA-graphed forward. Resolution/shape bucketing remains the
    #: encoder's responsibility.
    pad_to_max_batch: bool = False
    #: Seconds ``shutdown()`` waits for an in-flight forward before deferring the
    #: reap to a later call.
    _join_timeout_s: float = 10.0

    # ---- subclass contract -------------------------------------------------

    @abstractmethod
    def build(self, model_id: str, device: str) -> None:
        """Load weights / tokenizer and (optionally) compile + warm up.

        Runs **on the batcher thread**, so any CUDA graph captured here is bound
        to the thread that later replays it in ``forward_batch``.

        Args:
            model_id: The LM checkpoint — local dir or HF id, passed verbatim
                from ``--model``.
            device: Target device string, e.g. ``"cuda"``.
        """
        ...

    @abstractmethod
    def forward_batch(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Encode a coalesced batch of images, one tensor per URL, in order.

        Runs on the batcher thread, serialized (one call at a time). ``image_urls``
        is a coalesced batch of at most ``max_batch_size`` images; with
        ``pad_to_max_batch`` it is padded to exactly ``max_batch_size`` (the
        runtime drops the padded outputs). Return one ``(n_visual_tokens,
        lm_hidden_dim)`` tensor per URL.
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

    # ---- runtime (provided) ------------------------------------------------

    def load(self, model_id: str, device: str) -> None:
        """Start the batcher thread, run ``build`` on it, and fail fast.

        Blocks until ``build`` completes on the batcher thread, re-raising any
        build error, then ``validate``s the placeholder id so a misconfigured
        encoder errors at startup instead of on the first request.
        """
        # Runtime state is set up here (not __init__) so subclasses need not call
        # super().__init__(); the worker always calls load() before encode().
        self._queue: queue.Queue = queue.Queue()
        self._ready = threading.Event()
        self._build_error: Optional[BaseException] = None
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            args=(model_id, device),
            name="batched-custom-encoder",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()
        if self._build_error is not None:
            # build() returned, so the thread has already exited. Mark closed
            # (via shutdown) so a later encode() raises instead of queueing to a
            # dead consumer and hanging.
            self.shutdown()
            raise self._build_error
        try:
            self.validate()
        except BaseException:
            # build() succeeded, so the batcher thread is alive waiting on the
            # queue (holding the model). Stop it before propagating so a failed
            # handler init does not leak the thread / GPU memory.
            self.shutdown()
            raise

    def validate(self) -> None:
        """Fail-fast checks run by ``load`` after ``build``.

        Resolves the placeholder id once (it is not request-dependent). Subclasses
        may override to add post-build checks (call ``super().validate()`` first).
        """
        self.get_image_placeholder_token_id()

    async def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Submit a request's images to the batcher and await its embeddings.

        The worker ``await``s this directly; coalescing and the forward run on the
        batcher thread, so the event loop is never blocked by the forward.
        """
        lock = getattr(self, "_lifecycle_lock", None)
        if lock is None:
            raise RuntimeError("BatchedCustomEncoder.encode() called before load()")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        # Hold the lock across the closed-check and the put so a concurrent
        # shutdown() cannot slip the sentinel between them and strand this request
        # in a queue with no consumer.
        with lock:
            if self._closed:
                raise RuntimeError(
                    "BatchedCustomEncoder.encode() called after shutdown()"
                )
            self._queue.put(_Pending(list(image_urls), loop, future))
        return await future

    def shutdown(self) -> None:
        """Stop the batcher thread, failing any still-queued requests.

        Idempotent and safe to call before ``load``. If a slow forward is still
        in flight when the join times out, the thread is left to drain on its own
        and a later ``shutdown()`` call reaps it (we must not remove the sentinel
        while the thread is alive, or it would block forever on an empty queue).
        """
        lock = getattr(self, "_lifecycle_lock", None)
        if lock is None:
            return  # never loaded
        with lock:
            if not self._closed:
                # Mark closed and enqueue the stop sentinel exactly once; repeat
                # calls fall through to retry the join below.
                self._closed = True
                self._queue.put(_SHUTDOWN)
        self._thread.join(timeout=self._join_timeout_s)
        if self._thread.is_alive():
            # A forward is still running: leave the sentinel queued for the thread
            # to consume and do NOT drain (removing it would strand the thread).
            # A later shutdown() call retries the join.
            logger.warning(
                "BatchedCustomEncoder.shutdown: batcher thread still running after "
                "%gs; will reap on a later call",
                self._join_timeout_s,
            )
            return
        # The consumer has exited; resolve anything still queued so an awaiter that
        # raced shutdown does not hang forever.
        self._drain_pending()

    def _drain_pending(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _SHUTDOWN:
                continue
            self._resolve(item, exc=RuntimeError("BatchedCustomEncoder shut down"))

    # ---- batcher thread ----------------------------------------------------

    def _run(self, model_id: str, device: str) -> None:
        try:
            self.build(model_id, device)  # capture CUDA graphs HERE
        except (
            BaseException
        ) as exc:  # noqa: BLE001 — surface any build failure to load()
            self._build_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            batch = self._collect()
            if batch is None:
                return
            self._dispatch(batch)

    def _collect(self) -> Optional[List[_Pending]]:
        """Block for one request, then coalesce more within the wait window."""
        first = self._queue.get()
        if first is _SHUTDOWN:
            return None
        pending: List[_Pending] = [first]
        n_images = len(first.image_urls)
        deadline = time.monotonic() + self.max_wait_ms / 1000.0
        while n_images < self.max_batch_size:
            timeout = deadline - time.monotonic()
            if timeout <= 0:
                break
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                break
            if item is _SHUTDOWN:
                # Drain the batch we have, then stop on the next loop.
                self._queue.put(_SHUTDOWN)
                break
            pending.append(item)
            n_images += len(item.image_urls)
        return pending

    def _dispatch(self, pending: List[_Pending]) -> None:
        """Run the coalesced batch through ``forward_batch`` and fan results out."""
        flat: List[str] = []
        spans: List[tuple[_Pending, int, int]] = []  # (req, start, count)
        for req in pending:
            spans.append((req, len(flat), len(req.image_urls)))
            flat.extend(req.image_urls)
        try:
            outputs = self._forward_all(flat)
        except (
            BaseException
        ) as exc:  # noqa: BLE001 — one bad batch must not hang awaiters
            for req, _, _ in spans:
                self._resolve(req, exc=exc)
            return
        if len(outputs) != len(flat):
            err = RuntimeError(
                f"forward_batch returned {len(outputs)} tensors for {len(flat)} "
                "images; it must return one tensor per URL"
            )
            for req, _, _ in spans:
                self._resolve(req, exc=err)
            return
        for req, start, count in spans:
            self._resolve(req, result=outputs[start : start + count])

    def _forward_all(self, flat: List[str]) -> List[torch.Tensor]:
        """Forward the flattened batch in chunks of at most ``max_batch_size``.

        Chunking caps the per-forward image count even when a single request (or
        coalescing overshoot) exceeds ``max_batch_size``. With ``pad_to_max_batch``
        each short chunk is padded to exactly ``max_batch_size`` (repeat the last
        URL, drop the padded outputs) so the batch dimension is static for a CUDA
        graph.
        """
        outputs: List[torch.Tensor] = []
        for start in range(0, len(flat), self.max_batch_size):
            chunk = flat[start : start + self.max_batch_size]
            if self.pad_to_max_batch and len(chunk) < self.max_batch_size:
                padded = chunk + [chunk[-1]] * (self.max_batch_size - len(chunk))
                outputs.extend(self.forward_batch(padded)[: len(chunk)])
            else:
                outputs.extend(self.forward_batch(chunk))
        return outputs

    def _resolve(
        self,
        req: _Pending,
        result: Optional[List[torch.Tensor]] = None,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Resolve a request's future on its own event loop (cross-thread safe)."""

        def _set() -> None:
            if req.future.done():
                return
            if exc is not None:
                req.future.set_exception(exc)
            else:
                req.future.set_result(result)

        req.loop.call_soon_threadsafe(_set)
