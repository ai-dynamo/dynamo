# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang publication of the active image-tokenization contract."""

from typing import Any, Optional

from dynamo.common.image_tokenization import ImageTokenizationSpec, type_identity

_QWEN_PROCESSOR = "sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor"
_KIMI_PROCESSOR = "sglang.srt.multimodal.processors.kimi_k25.KimiK2_5VLImageProcessor"

_QWEN2_MODEL_TYPES = frozenset({"qwen2_vl", "qwen2_5_vl"})
_QWEN3_MODEL_TYPES = frozenset({"qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"})


def resolve_sglang_image_tokenization_spec(
    processor_identity: str,
    model_type: Optional[str],
    *,
    has_image_overrides: bool,
) -> Optional[ImageTokenizationSpec]:
    """Map an exact built-in processor implementation to a semantic spec.

    Any worker-level image override fails closed because the Rust frontend
    sees the model's processor config, not SGLang's runtime-only overrides.
    """

    if has_image_overrides:
        return None
    if processor_identity == _KIMI_PROCESSOR:
        return ImageTokenizationSpec.MOONVIT_V1
    if processor_identity != _QWEN_PROCESSOR:
        return None
    # SGLang deliberately uses one QwenVLImageProcessor class for both
    # generations; this is the same model_type dispatch the live instance
    # uses internally, not a model-id/family guess by the frontend.
    if model_type in _QWEN2_MODEL_TYPES:
        return ImageTokenizationSpec.QWEN2_VL_V1
    if model_type in _QWEN3_MODEL_TYPES:
        return ImageTokenizationSpec.QWEN3_VL_V1
    return None


def get_sglang_image_tokenization_spec(
    engine: Any,
) -> Optional[ImageTokenizationSpec]:
    """Inspect the processor instance selected by the running engine."""

    processor = engine.tokenizer_manager.mm_processor
    if processor is None:
        return None

    processor_identity = type_identity(processor)
    if processor_identity not in {_QWEN_PROCESSOR, _KIMI_PROCESSOR}:
        return None
    model_type = (
        processor.model_type
        if processor_identity == _QWEN_PROCESSOR
        else processor.hf_config.model_type
    )
    return resolve_sglang_image_tokenization_spec(
        processor_identity,
        model_type,
        has_image_overrides=bool(processor.image_config),
    )
