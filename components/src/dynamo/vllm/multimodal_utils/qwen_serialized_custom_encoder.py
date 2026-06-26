# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-family base class for the pluggable CustomEncoder path.

Combines the serial/off-loop default (``SerializedCustomEncoder``) with Qwen
placeholder-id resolution, so customer encoders for Qwen-family models
(Qwen2-VL / Qwen3-VL / Qwen3-VL-MoE / Qwen3.5) only implement ``load``
(assigning ``self.tokenizer``) and the synchronous ``_encode_blocking``::

    from dynamo.vllm.multimodal_utils.qwen_serialized_custom_encoder import (
        QwenSerializedCustomEncoder,
    )

    class MyEncoder(QwenSerializedCustomEncoder):
        def load(self, model_id, device): ...        # set self.tokenizer
        def _encode_blocking(self, image_urls): ...   # sync forward
"""

from __future__ import annotations

from dynamo.vllm.multimodal_utils.custom_encoder import (
    placeholder_token_id_from_tokenizer,
)
from dynamo.vllm.multimodal_utils.serialized_custom_encoder import (
    SerializedCustomEncoder,
)

# Image placeholder token *string* for the Qwen family. The numeric id is always
# resolved from the encoder's tokenizer (placeholder_token_id_from_tokenizer);
# the family-specific knowledge is just this string. The same string maps to
# different ids across Qwen versions (e.g. 151655 for Qwen3-VL vs 248056 for
# Qwen3.5), which is exactly why the tokenizer — not a static id table — is
# authoritative.
QWEN_IMAGE_PLACEHOLDER_TOKEN = "<|image_pad|>"


class QwenSerializedCustomEncoder(SerializedCustomEncoder):
    """Serial Qwen-family base (Qwen2-VL / Qwen3-VL / Qwen3.5).

    Inherits the serial/off-loop ``encode`` from ``SerializedCustomEncoder`` and
    implements ``get_image_placeholder_token_id`` via the model tokenizer
    (resolving ``QWEN_IMAGE_PLACEHOLDER_TOKEN``). Subclasses implement ``load``
    (assigning ``self.tokenizer``) and the synchronous ``_encode_blocking``;
    both stay abstract, so this class cannot be instantiated directly.
    """

    def get_image_placeholder_token_id(self) -> int:
        """Resolve the Qwen image placeholder id from the model tokenizer.

        The tokenizer is assigned by the subclass in ``load()``. The same
        ``<|image_pad|>`` string resolves to the per-version id (151655 for
        Qwen3-VL, 248056 for Qwen3.5), so the tokenizer is authoritative.

        Raises:
            ValueError: if ``self.tokenizer`` is unset, or the tokenizer does
                not define ``QWEN_IMAGE_PLACEHOLDER_TOKEN``.
        """
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(
                "self.tokenizer is not set; assign the model tokenizer in "
                "load() so the Qwen image placeholder id can be resolved."
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
