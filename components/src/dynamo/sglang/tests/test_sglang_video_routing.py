# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest

from dynamo.llm import ModelRuntimeConfig
from dynamo.sglang import video_routing

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.multimodal,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]


def _engine(
    *,
    video_config=None,
    model_type="qwen3_vl",
    architecture="Qwen3VLForConditionalGeneration",
    overrides_video_replacement=True,
):
    if overrides_video_replacement:

        class QwenProcessor(video_routing.transformers.ProcessorMixin):
            def replace_video_token(self):
                return None

    else:

        class QwenProcessor(video_routing.transformers.ProcessorMixin):
            pass

    mm_processor = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type=model_type,
            architectures=[architecture],
        ),
        video_config=video_config or {},
        _processor=QwenProcessor.__new__(QwenProcessor),
    )
    return SimpleNamespace(tokenizer_manager=SimpleNamespace(mm_processor=mm_processor))


@pytest.fixture
def qwen_preprocessor(monkeypatch):
    monkeypatch.setattr(
        video_routing,
        "sglang_qwen_vl",
        SimpleNamespace(
            IMAGE_FACTOR=28,
            VIDEO_MIN_PIXELS=100352,
            VIDEO_MAX_PIXELS=602112,
            VIDEO_TOTAL_PIXELS=90316800,
            FRAME_FACTOR=2,
            FPS=2.0,
            FPS_MIN_FRAMES=4,
            FPS_MAX_FRAMES=768,
        ),
    )
    monkeypatch.setattr(
        video_routing,
        "qwen3_smart_resize",
        lambda **_: (1216, 4096),
    )


def test_publishes_sglang_qwen_video_contract(qwen_preprocessor):
    runtime_config = ModelRuntimeConfig()

    video_routing.publish_sglang_qwen_video_processor_contract(
        runtime_config, _engine()
    )

    contract = json.loads(
        runtime_config.runtime_data[
            video_routing.SGLANG_QWEN_VIDEO_PROCESSOR_CONTRACT_RUNTIME_KEY
        ]
    )
    assert contract == {
        "placeholder_target": "bare_video_token",
        "resize_mode": "legacy_ceil",
        "runless_boundary_hash": "tokens_only",
        "sglang_preprocess": {
            "image_factor": 28,
            "video_min_pixels": 100352,
            "video_max_pixels": 602112,
            "video_total_pixels": 90316800,
            "frame_factor": 2,
            "fps": 2.0,
            "min_frames": 4,
            "max_frames": 768,
        },
    }


def test_processor_override_disables_exact_video_contract(qwen_preprocessor):
    assert (
        video_routing._resolve_qwen_video_processor_contract(
            _engine(video_config={"fps": 1.0})
        )
        is None
    )


def test_missing_processor_disables_exact_video_contract(qwen_preprocessor):
    engine = _engine()
    engine.tokenizer_manager.mm_processor._processor = None

    assert video_routing._resolve_qwen_video_processor_contract(engine) is None


def test_inherited_video_replacement_publishes_wrapped_target(qwen_preprocessor):
    contract = video_routing._resolve_qwen_video_processor_contract(
        _engine(overrides_video_replacement=False)
    )

    assert contract is not None
    assert contract["placeholder_target"] == "vision_wrapped_video_token"


@pytest.mark.parametrize(
    ("model_type", "architecture"),
    [
        ("llava", "Qwen3VLForConditionalGeneration"),
        ("qwen3_vl", "Qwen3VLForCausalLM"),
    ],
)
def test_non_qwen_video_model_does_not_publish_contract(
    qwen_preprocessor, model_type, architecture
):
    assert (
        video_routing._resolve_qwen_video_processor_contract(
            _engine(model_type=model_type, architecture=architecture)
        )
        is None
    )
