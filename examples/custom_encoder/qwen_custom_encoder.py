# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable example base for Qwen-family custom encoders.

Combines the ``BatchedCustomEncoder`` runtime (dedicated thread + coalescing
micro-batcher) with Qwen placeholder-id resolution (``QwenPlaceholderMixin``
resolves ``<|image_pad|>`` from the model tokenizer). A concrete Qwen-family
encoder subclasses this and implements only ``forward_batch``; ``build`` already
loads the model tokenizer and assigns ``self.tokenizer`` so the placeholder id
resolves.

    class MyQwenEncoder(QwenCustomEncoder):
        def build(self, model_id, device):
            super().build(model_id, device)   # loads self.tokenizer
            # ... load ViT + projector ...
        def forward_batch(self, image_urls):
            ...                                 # synchronous batched forward
"""

from __future__ import annotations

from transformers import AutoTokenizer

from dynamo.vllm.multimodal_utils.batched_custom_encoder import (
    BatchedCustomEncoder,
    QwenPlaceholderMixin,
)


class QwenCustomEncoder(QwenPlaceholderMixin, BatchedCustomEncoder):
    """BatchedCustomEncoder base for Qwen-family models (Qwen2-VL / Qwen3-VL /
    Qwen3.5).

    ``build`` loads the model tokenizer (so ``QwenPlaceholderMixin`` can resolve
    the ``<|image_pad|>`` id); ``forward_batch`` stays abstract, so this class
    cannot be instantiated directly — subclass it and implement the forward.
    """

    def build(self, model_id: str, device: str) -> None:
        """Load the model tokenizer. Subclasses extend this (call super) to also
        load their encoder weights.

        Assigns ``self.tokenizer`` — named ``tokenizer`` (not ``_tokenizer``) so
        ``QwenPlaceholderMixin`` can resolve the ``<|image_pad|>`` id from it.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
