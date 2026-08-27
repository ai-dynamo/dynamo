# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reusable example base for Qwen-family ``EmbedsPrompt`` encoder authors.

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

from dynamo.vllm.multimodal_utils.custom_encoder import VisionEncoderBackend


class QwenVisionEncoderBackend(VisionEncoderBackend):
    """``VisionEncoderBackend`` base for Qwen-family ``EmbedsPrompt`` examples.

    Hardcodes ``image_token_id`` to Qwen3-VL's ``<|image_pad|>`` (151655) — override
    it for other versions. The linear adapter uses this token to locate embedding
    replacement spans; native ``TokensPrompt`` encoders, such as the Qwen3.5
    example, do not need it. ``build`` loads the model tokenizer;
    ``forward_batch`` stays abstract, so this class cannot be instantiated directly.
    """

    # Qwen3-VL <|image_pad|>; override for other Qwen versions.
    image_token_id = 151655

    def build(self, model_id: str) -> None:
        """Load the model tokenizer. Subclasses extend this (call super) to also
        load their encoder weights (picking the device themselves)."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
