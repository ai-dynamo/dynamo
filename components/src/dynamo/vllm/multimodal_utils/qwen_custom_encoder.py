# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen-family base class for the pluggable CustomEncoder path.

Presets the Qwen image placeholder token string so customer encoders for
Qwen-family models (Qwen2-VL / Qwen3-VL / Qwen3-VL-MoE / Qwen3.5) only implement
``load`` (assigning ``self.tokenizer``) and ``encode``::

    from dynamo.vllm.multimodal_utils.qwen_custom_encoder import QwenCustomEncoder

    class MyEncoder(QwenCustomEncoder):
        def load(self, model_id, device): ...   # set self.tokenizer
        def encode(self, image_urls): ...
"""

from __future__ import annotations

from dynamo.vllm.multimodal_utils.custom_encoder import CustomEncoder

# Image placeholder token *string* for the Qwen family. The numeric id is always
# resolved from the encoder's tokenizer (placeholder_token_id_from_tokenizer);
# the family-specific knowledge is just this string. The same string maps to
# different ids across Qwen versions (e.g. 151655 for Qwen3-VL vs 248056 for
# Qwen3.5), which is exactly why the tokenizer — not a static id table — is
# authoritative.
QWEN_IMAGE_PLACEHOLDER_TOKEN = "<|image_pad|>"


class QwenCustomEncoder(CustomEncoder):
    """Semi-abstract base for Qwen-family encoders (Qwen2-VL / Qwen3-VL / Qwen3.5).

    Presets the Qwen image placeholder token string; the numeric id is resolved
    from the model tokenizer at runtime. Subclasses implement ``load``
    (assigning ``self.tokenizer``) and ``encode``; both stay abstract, so this
    class cannot be instantiated directly.
    """

    image_placeholder_token = QWEN_IMAGE_PLACEHOLDER_TOKEN
