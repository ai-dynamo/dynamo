# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable example base for Qwen-family ``VisionEncoderBackend`` authors.

Combines the contract (``VisionEncoderBackend``) with Qwen placeholder-id
resolution (``QwenPlaceholderMixin`` resolves ``<|image_pad|>`` from the model
tokenizer). A concrete Qwen-family encoder subclasses this and implements only
``preprocess`` + ``forward_batch``; ``build`` already loads the model tokenizer
and assigns ``self.tokenizer`` so the placeholder id resolves.

    class MyQwenEncoder(QwenVisionEncoderBackend):
        def build(self, model_id, device):
            super().build(model_id, device)   # loads self.tokenizer
            # ... load ViT + projector ...
        def preprocess(self, raw):
            ...                                 # off-thread, returns Preprocessed
        def forward_batch(self, items, target_bucket=None):
            ...                                 # actor thread, batched forward
"""

from __future__ import annotations

from transformers import AutoTokenizer

from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    QwenPlaceholderMixin,
    VisionEncoderBackend,
)


class QwenVisionEncoderBackend(QwenPlaceholderMixin, VisionEncoderBackend):
    """``VisionEncoderBackend`` base for Qwen-family models (Qwen2-VL / Qwen3-VL /
    Qwen3.5).

    ``build`` loads the model tokenizer (so ``QwenPlaceholderMixin`` can resolve
    the ``<|image_pad|>`` id); ``preprocess`` and ``forward_batch`` stay abstract,
    so this class cannot be instantiated directly — subclass it and implement them.
    """

    def build(self, model_id: str, device: str) -> None:
        """Load the model tokenizer. Subclasses extend this (call super) to also
        load their encoder weights.

        Assigns ``self.tokenizer`` — named ``tokenizer`` (not ``_tokenizer``) so
        ``QwenPlaceholderMixin`` can resolve the ``<|image_pad|>`` id from it.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
