# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.common.image_tokenization import ImageTokenizationSpec
from dynamo.vllm.image_tokenization import (
    get_vllm_image_tokenization_spec,
    resolve_vllm_image_tokenization_spec,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_vllm_resolves_only_exact_supported_processors():
    qwen2 = "vllm.model_executor.models.qwen2_vl.Qwen2VLMultiModalProcessor"
    qwen3 = "vllm.model_executor.models.qwen3_vl.Qwen3VLMultiModalProcessor"
    kimi = "vllm.model_executor.models.kimi_k25.KimiK25MultiModalProcessor"
    fused = (
        "vllm.transformers_utils.processors.kimi_k25_vision_fused."
        "KimiK25FusedVisionProcessor"
    )

    assert (
        resolve_vllm_image_tokenization_spec(qwen2, has_processor_overrides=False)
        == ImageTokenizationSpec.QWEN2_VL_V1
    )
    assert (
        resolve_vllm_image_tokenization_spec(qwen3, has_processor_overrides=False)
        == ImageTokenizationSpec.QWEN3_VL_V1
    )
    assert (
        resolve_vllm_image_tokenization_spec(
            kimi,
            image_processor_identity=fused,
            has_processor_overrides=False,
        )
        == ImageTokenizationSpec.MOONVIT_V1
    )
    assert (
        resolve_vllm_image_tokenization_spec(
            kimi,
            image_processor_identity="custom.RemoteProcessor",
            has_processor_overrides=False,
        )
        is None
    )
    assert (
        resolve_vllm_image_tokenization_spec(qwen3, has_processor_overrides=True)
        is None
    )


def test_vllm_inspects_the_live_concrete_processor():
    processor_type = type(
        "Qwen2VLMultiModalProcessor",
        (),
        {"__module__": "vllm.model_executor.models.qwen2_vl"},
    )
    processor = processor_type()
    engine_client = SimpleNamespace(
        input_processor=SimpleNamespace(
            renderer=SimpleNamespace(mm_processor=processor)
        )
    )
    mm_config = SimpleNamespace(mm_processor_kwargs=None)
    model_config = SimpleNamespace(
        is_multimodal_model=True,
        get_multimodal_config=lambda: mm_config,
    )

    assert (
        get_vllm_image_tokenization_spec(engine_client, model_config)
        == ImageTokenizationSpec.QWEN2_VL_V1
    )
