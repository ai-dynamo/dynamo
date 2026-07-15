# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM publication of the active image-tokenization contract."""

from typing import Any, Optional

from dynamo.common.image_tokenization import ImageTokenizationSpec, type_identity

_QWEN2_PROCESSORS = frozenset(
    {
        "vllm.model_executor.models.qwen2_vl.Qwen2VLMultiModalProcessor",
        "vllm.model_executor.models.qwen2_5_vl.Qwen2_5_VLMultiModalProcessor",
    }
)
_QWEN3_PROCESSORS = frozenset(
    {"vllm.model_executor.models.qwen3_vl.Qwen3VLMultiModalProcessor"}
)
_KIMI_PROCESSOR = "vllm.model_executor.models.kimi_k25.KimiK25MultiModalProcessor"
_KIMI_FUSED_IMAGE_PROCESSOR = (
    "vllm.transformers_utils.processors.kimi_k25_vision_fused."
    "KimiK25FusedVisionProcessor"
)


def resolve_vllm_image_tokenization_spec(
    processor_identity: str,
    *,
    image_processor_identity: Optional[str] = None,
    has_processor_overrides: bool,
) -> Optional[ImageTokenizationSpec]:
    """Map exact built-in processor implementations to semantic specs."""

    if has_processor_overrides:
        return None
    if processor_identity in _QWEN2_PROCESSORS:
        return ImageTokenizationSpec.QWEN2_VL_V1
    if processor_identity in _QWEN3_PROCESSORS:
        return ImageTokenizationSpec.QWEN3_VL_V1
    if (
        processor_identity == _KIMI_PROCESSOR
        and image_processor_identity == _KIMI_FUSED_IMAGE_PROCESSOR
    ):
        # Kimi falls back to arbitrary remote HF code when numba is absent.
        # Only vLLM's concrete fused implementation carries this contract.
        return ImageTokenizationSpec.MOONVIT_V1
    return None


def get_vllm_image_tokenization_spec(
    engine_client: Any,
    model_config: Any,
) -> Optional[ImageTokenizationSpec]:
    """Inspect the multimodal processor instance used by the live engine."""

    if not model_config.is_multimodal_model:
        return None

    mm_config = model_config.get_multimodal_config()
    processor = engine_client.input_processor.renderer.mm_processor
    if processor is None:
        return None

    processor_identity = type_identity(processor)
    image_processor_identity = None
    if processor_identity == _KIMI_PROCESSOR:
        image_processor_identity = type_identity(processor.info.image_processor)

    return resolve_vllm_image_tokenization_spec(
        processor_identity,
        image_processor_identity=image_processor_identity,
        has_processor_overrides=bool(mm_config.mm_processor_kwargs),
    )
