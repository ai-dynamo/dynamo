# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interface for pluggable vision encoders in Dynamo.

The encoder runs **in the same process** as the vLLM aggregated worker — no
separate encode worker, no NIXL transfer.  It handles image encoding and
projection to the LM hidden dim; vLLM embeds the text tokens itself while
Dynamo splices the encoder's image embeds at the placeholder positions,
submitting the result as a mixed ``EmbedsPrompt`` (``prompt_token_ids`` +
``prompt_is_token_ids`` + ``prompt_embeds``).

The base class defines the *contract* only — ``load``, async ``encode``, and
``get_image_placeholder_token_id``.  ``encode`` is async and the worker simply
``await``s it; the encoder owns its own execution model (serial, or its own
batching scheduler).  Most authors should subclass ``SerializedCustomEncoder``
(in ``serialized_custom_encoder``), which implements ``encode`` for them —
running a synchronous forward off the event loop, serialized — so they only
write a blocking forward.  Qwen-family authors subclass
``QwenSerializedCustomEncoder`` (in ``qwen_custom_encoder``), which additionally
resolves the placeholder id from the model tokenizer.

Usage::

    # Qwen-family model: subclass QwenSerializedCustomEncoder
    from dynamo.vllm.multimodal_utils.qwen_custom_encoder import (
        QwenSerializedCustomEncoder,
    )

    class MyEncoder(QwenSerializedCustomEncoder):
        def load(self, model_id, device):
            # load ViT + projector + tokenizer; set self.tokenizer
            ...
        def _encode_blocking(self, image_urls):
            # synchronous forward: one (n_tokens, lm_hidden_dim) tensor per URL
            ...

    # launch
    python -m dynamo.vllm \\
        --model /weights/my_lm \\
        --custom-encoder-class customer_encoder.MyEncoder \\
        --enable-multimodal
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch


def placeholder_token_id_from_tokenizer(
    tokenizer: object,
    token: str,
) -> Optional[int]:
    """Resolve the image placeholder token ID from a loaded tokenizer.

    The authoritative source for a tokenizer-based encoder: the encoder loads
    the model's tokenizer, which defines the special token. Returns ``None`` if
    the tokenizer does not define ``token`` (it maps to the unknown-token ID),
    so callers can raise a clear error.

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


class CustomEncoder(ABC):
    """Pluggable image encoder for aggregated serving.

    Runs **in-process** inside the vLLM aggregated worker — no separate encode
    worker needed.  Handles image encoding + projection to the LM hidden dim.
    vLLM embeds the text tokens itself; Dynamo splices the returned image
    tensors at the placeholder positions and submits a mixed ``EmbedsPrompt``.
    The encoder never needs the LM's text embedding table.

    Returned tensor shape per image: ``(n_visual_tokens, lm_hidden_dim)``
    where ``n_visual_tokens`` is determined by the encoder (e.g. number of
    ViT patches after merging).

    Subclasses implement ``load``, async ``encode``, and
    ``get_image_placeholder_token_id``.  For the common case, subclass
    ``SerializedCustomEncoder`` instead of implementing ``encode`` directly: it
    runs a synchronous forward off the event loop, serialized, so you only write
    ``_encode_blocking``.  Qwen-family models subclass
    ``QwenSerializedCustomEncoder``, which also resolves the placeholder id from
    the model tokenizer.

    Usage::

        # Common case: subclass SerializedCustomEncoder (write the sync forward)
        from dynamo.vllm.multimodal_utils.serialized_custom_encoder import (
            SerializedCustomEncoder,
        )

        class MyEncoder(SerializedCustomEncoder):
            def load(self, model_id, device): ...
            def get_image_placeholder_token_id(self): ...
            def _encode_blocking(self, image_urls): ...   # sync forward

        # Advanced: implement async encode directly to run your own batching
        class MyBatchedEncoder(CustomEncoder):
            def load(self, model_id, device): ...
            def get_image_placeholder_token_id(self): ...
            async def encode(self, image_urls): ...        # owns its scheduler

        python -m dynamo.vllm \\
            --model /weights/my_lm \\
            --custom-encoder-class my_module.MyEncoder \\
            --enable-multimodal
    """

    @abstractmethod
    def load(self, model_id: str, device: str) -> None:
        """Load vision encoder weights.

        Args:
            model_id: The LM checkpoint — a local dir or HF model id, passed
                verbatim from ``--model``.
            device: Target device string, e.g. ``"cuda"``.
        """
        ...

    @abstractmethod
    async def encode(self, image_urls: List[str]) -> List[torch.Tensor]:
        """Encode images, returning per-image visual token embeddings.

        Async: the worker ``await``s it. The encoder owns its execution model —
        the worker holds no lock and does not offload to a thread, so:

        - It **may be called concurrently** (multiple in-flight requests). The
          encoder owns any serialization or cross-request batching.
        - It **must not block the event loop**: offload the (blocking) forward
          via ``asyncio.to_thread`` or run it on the encoder's own scheduler.

        Most authors should not implement this directly — subclass
        ``SerializedCustomEncoder`` and implement ``_encode_blocking`` to get a
        safe serial, off-loop default. Implement ``encode`` directly only to run
        your own batching scheduler.

        Args:
            image_urls: URLs of images to encode.

        Returns:
            List of tensors, one per URL, each of shape
            ``(n_visual_tokens, lm_hidden_dim)``.
        """
        ...

    @abstractmethod
    def get_image_placeholder_token_id(self) -> int:
        """Return the token ID that marks image positions in the prompt.

        Dynamo uses it to locate the image span and splice the encoder tensors
        into the mixed ``EmbedsPrompt``.  How the id is obtained is the
        encoder's choice: Qwen-family encoders subclass
        ``QwenSerializedCustomEncoder``, which resolves it from the model
        tokenizer; other encoders resolve it however their model defines the
        image placeholder (e.g. via ``placeholder_token_id_from_tokenizer`` with
        their token string, or a known constant).

        Raises:
            ValueError: if the encoder cannot resolve a valid id. Surfaced at
                startup by ``validate()``.
        """
        ...

    def validate(self) -> None:
        """Fail-fast checks to run immediately after ``load()``.

        Resolves the image placeholder token id so a misconfigured encoder
        errors at startup instead of on the first request. The result is not
        request-dependent, so it is safe to check once after ``load()``. The
        worker calls this right after ``load()``; subclasses can override to add
        their own post-load checks (call ``super().validate()`` first).

        Raises:
            ValueError: propagated from ``get_image_placeholder_token_id``.
        """
        self.get_image_placeholder_token_id()
