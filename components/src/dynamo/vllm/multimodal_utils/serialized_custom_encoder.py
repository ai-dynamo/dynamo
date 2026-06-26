# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serial, off-loop default base for the pluggable CustomEncoder path.

The safe default for the 90% case: implement a *synchronous* forward in
``_encode_blocking`` and this base provides ``encode`` for you. ``encode`` runs
the forward via ``asyncio.to_thread`` (so the event loop stays responsive) under
an internal ``asyncio.Lock`` (so concurrent requests are serialized — one
forward at a time, bounded memory, and ``_encode_blocking`` need not be
re-entrant).

Subclass this unless you intend to run your own batching scheduler; in that
case subclass ``CustomEncoder`` and implement async ``encode`` directly.
"""

from __future__ import annotations

import asyncio
from abc import abstractmethod
from typing import List, Optional

import torch

from dynamo.vllm.multimodal_utils.custom_encoder import CustomEncoder


class SerializedCustomEncoder(CustomEncoder):
    """CustomEncoder base that runs a synchronous forward serially, off-loop.

    Implements ``encode`` by running ``_encode_blocking`` in a worker thread
    (``asyncio.to_thread``) under an internal lock, so:

    - the event loop is never blocked by the forward, and
    - concurrent ``encode`` calls are serialized (one forward at a time), so
      ``_encode_blocking`` is plain single-threaded code — no re-entrancy
      needed — and peak memory is bounded to a single forward.

    Subclasses implement ``load``, ``get_image_placeholder_token_id``, and the
    synchronous ``_encode_blocking``.
    """

    def _serialize_lock(self) -> asyncio.Lock:
        # Lazily created so subclasses need not call super().__init__(). The
        # check-and-set has no await between the two statements, so it is atomic
        # with respect to the single-threaded event loop (no double-create race).
        lock: Optional[asyncio.Lock] = getattr(self, "_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._lock = lock
        return lock

    async def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Run ``_encode_blocking`` off the event loop, serialized."""
        async with self._serialize_lock():
            return await asyncio.to_thread(self._encode_blocking, image_urls)

    @abstractmethod
    def _encode_blocking(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Synchronous forward: encode images to per-image visual embeddings.

        Runs in a worker thread, serialized by ``encode``, so it need not be
        re-entrant. Return one ``(n_visual_tokens, lm_hidden_dim)`` tensor per
        URL, in prompt order.
        """
        ...
