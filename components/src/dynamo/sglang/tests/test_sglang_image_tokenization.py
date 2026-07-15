# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from dynamo.common.image_tokenization import ImageTokenizationSpec
from dynamo.sglang.image_tokenization import (
    get_sglang_image_tokenization_spec,
    resolve_sglang_image_tokenization_spec,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_sglang_resolves_only_exact_supported_processors():
    qwen = "sglang.srt.multimodal.processors.qwen_vl.QwenVLImageProcessor"
    kimi = "sglang.srt.multimodal.processors.kimi_k25.KimiK2_5VLImageProcessor"

    assert (
        resolve_sglang_image_tokenization_spec(
            qwen, "qwen2_5_vl", has_image_overrides=False
        )
        == ImageTokenizationSpec.QWEN2_VL_V1
    )
    assert (
        resolve_sglang_image_tokenization_spec(
            qwen, "qwen3_5", has_image_overrides=False
        )
        == ImageTokenizationSpec.QWEN3_VL_V1
    )
    assert (
        resolve_sglang_image_tokenization_spec(
            kimi, "kimi_k25", has_image_overrides=False
        )
        == ImageTokenizationSpec.MOONVIT_V1
    )
    assert (
        resolve_sglang_image_tokenization_spec(
            qwen, "qwen2_5_vl", has_image_overrides=True
        )
        is None
    )
    assert (
        resolve_sglang_image_tokenization_spec(
            "custom.module.QwenVLImageProcessor",
            "qwen3_vl",
            has_image_overrides=False,
        )
        is None
    )


def test_sglang_inspects_the_live_concrete_processor():
    processor_type = type(
        "QwenVLImageProcessor",
        (),
        {"__module__": "sglang.srt.multimodal.processors.qwen_vl"},
    )
    processor = processor_type()
    processor.model_type = "qwen3_vl"
    processor.image_config = {}
    engine = SimpleNamespace(tokenizer_manager=SimpleNamespace(mm_processor=processor))

    assert (
        get_sglang_image_tokenization_spec(engine) == ImageTokenizationSpec.QWEN3_VL_V1
    )

    unknown_type = type("CustomProcessor", (), {"__module__": "custom.processor"})
    engine.tokenizer_manager.mm_processor = unknown_type()
    assert get_sglang_image_tokenization_spec(engine) is None
