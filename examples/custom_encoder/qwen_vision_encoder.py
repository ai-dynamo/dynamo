# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable example base for Qwen-family ``VisionEncoderBackend`` authors.

Hardcodes the Qwen ``<|image_pad|>`` placeholder id and loads the model tokenizer
(handy for subclasses that tokenize text). A concrete Qwen-family encoder
subclasses this and implements ``forward_batch`` — plus ``preprocess`` (and
``preprocess_concurrency > 0``) only when it needs off-loop fetch/resize.

    class MyQwenEncoder(QwenVisionEncoderBackend):
        preprocess_concurrency = 4              # enable the off-loop pool
        def build(self, model_id):
            super().build(model_id)             # loads self.tokenizer
            # ... load ViT + projector (pick the device yourself) ...
        def preprocess(self, raw):
            ...                                 # off-thread, returns Preprocessed
        def forward_batch(self, items, target_bucket=None):
            ...                                 # actor thread, batched forward (CPU out)
"""

from __future__ import annotations

from transformers import AutoTokenizer

from dynamo.vllm.multimodal_utils.vision_encoder_backend import VisionEncoderBackend


class QwenVisionEncoderBackend(VisionEncoderBackend):
    """``VisionEncoderBackend`` base for Qwen-family models (Qwen2-VL / Qwen3-VL /
    Qwen3.5).

    Hardcodes ``image_token_id`` to Qwen3-VL's ``<|image_pad|>`` (151655) — override
    it for other versions (e.g. 248056 for Qwen3.5). ``build`` loads the model
    tokenizer; ``forward_batch`` stays abstract, so this class cannot be
    instantiated directly — subclass it and implement ``forward_batch`` (and
    ``preprocess`` only if it needs off-loop prep).
    """

    # Qwen3-VL <|image_pad|>; override for other Qwen versions.
    image_token_id = 151655

    def build(self, model_id: str) -> None:
        """Load the model tokenizer. Subclasses extend this (call super) to also
        load their encoder weights (picking the device themselves)."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
