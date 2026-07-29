# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder import (
    Qwen2VLImageEncoding,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_QWEN_START_TOKEN_ID = 100
_QWEN_IMAGE_TOKEN_ID = 101
_QWEN_END_TOKEN_ID = 102
_QWEN_VIDEO_TOKEN_ID = 103


class _QwenBackend(VisionEncoderBackend):
    def build(self, model_id: str) -> None:
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


def _qwen_model_config(
    architecture: str = "Qwen2_5_VLForConditionalGeneration",
):
    return SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        is_multimodal_model=lambda: True,
        hf_config=SimpleNamespace(
            architectures=[architecture],
            image_token_id=_QWEN_IMAGE_TOKEN_ID,
            vision_start_token_id=_QWEN_START_TOKEN_ID,
            vision_end_token_id=_QWEN_END_TOKEN_ID,
            video_token_id=_QWEN_VIDEO_TOKEN_ID,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
    )


def _qwen_engine_args(**overrides):
    values = {
        "enable_mm_embeds": True,
        "enable_prompt_embeds": False,
        "language_model_only": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "compilation_config": SimpleNamespace(cudagraph_mm_encoder=False),
        "enable_prefix_caching": False,
        "enable_chunked_prefill": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _qwen_adapter(monkeypatch, **engine_overrides):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.custom_encoder.adapter.qwen2_vl.version",
        lambda package: "0.25.1",
    )
    return create_custom_encoder_adapter(
        _QwenBackend(),
        _qwen_model_config(),
        _qwen_engine_args(**engine_overrides),
    )


def _qwen_encoding(
    rows: int = 1,
    grid_thw: tuple[int, int, int] = (1, 2, 2),
    **tensor_kwargs,
):
    values = {"dtype": torch.bfloat16}
    values.update(tensor_kwargs)
    return Qwen2VLImageEncoding(
        projected=torch.zeros((rows, 4), **values),
        grid_thw=grid_thw,
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
    ],
)
def test_qwen_decoder_selects_native_adapter(monkeypatch, architecture):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.custom_encoder.adapter.qwen2_vl.version",
        lambda package: "0.25.1",
    )

    adapter = create_custom_encoder_adapter(
        _QwenBackend(), _qwen_model_config(architecture), _qwen_engine_args()
    )

    assert type(adapter).__name__ == "_Qwen2VLNativeAdapter"


def test_qwen_adapter_builds_final_tokens_prompt_in_image_order(monkeypatch):
    adapter = _qwen_adapter(monkeypatch)
    token_ids = [
        _QWEN_START_TOKEN_ID,
        _QWEN_IMAGE_TOKEN_ID,
        _QWEN_END_TOKEN_ID,
        7,
        _QWEN_START_TOKEN_ID,
        _QWEN_IMAGE_TOKEN_ID,
        _QWEN_END_TOKEN_ID,
    ]
    first = Qwen2VLImageEncoding(torch.full((1, 4), 1, dtype=torch.bfloat16), (1, 2, 2))
    second = Qwen2VLImageEncoding(
        torch.full((2, 4), 2, dtype=torch.bfloat16), (1, 2, 4)
    )

    prompt = adapter.prepare_prompt(token_ids, [first, second])

    assert prompt["prompt_token_ids"] == token_ids
    image = prompt["multi_modal_data"]["image"]
    assert image["image_embeds"].shape == (3, 4)
    assert image["image_embeds"][:, 0].tolist() == [1, 2, 2]
    assert image["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]
    assert set(prompt["multi_modal_data"]) == {"image"}


def test_unknown_multimodal_decoder_is_rejected():
    with pytest.raises(ValueError, match="does not support"):
        create_custom_encoder_adapter(
            _QwenBackend(),
            _qwen_model_config("OtherVisionForConditionalGeneration"),
            _qwen_engine_args(),
        )


@pytest.mark.parametrize(
    "engine_overrides, message",
    [
        ({"enable_mm_embeds": False}, "--enable-mm-embeds"),
        ({"language_model_only": True}, "full registered model wrapper"),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size=1"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size=1"),
        ({"data_parallel_size": 2}, "data_parallel_size=1"),
        (
            {"compilation_config": {"cudagraph_mm_encoder": True}},
            "encoder CUDA graphs",
        ),
    ],
)
def test_qwen_adapter_rejects_unproven_engine_modes(
    monkeypatch, engine_overrides, message
):
    with pytest.raises(ValueError, match=message):
        _qwen_adapter(monkeypatch, **engine_overrides)


@pytest.mark.parametrize(
    "vllm_config, message",
    [
        (
            SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=True),
                scheduler_config=SimpleNamespace(enable_chunked_prefill=False),
            ),
            "no-enable-prefix-caching",
        ),
        (
            SimpleNamespace(
                cache_config=SimpleNamespace(enable_prefix_caching=False),
                scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
            ),
            "no-enable-chunked-prefill",
        ),
    ],
)
def test_qwen_adapter_rejects_unproven_resolved_modes(
    monkeypatch, vllm_config, message
):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.custom_encoder.adapter.qwen2_vl.version",
        lambda package: "0.25.1",
    )
    with pytest.raises(ValueError, match=message):
        create_custom_encoder_adapter(
            _QwenBackend(),
            _qwen_model_config(),
            _qwen_engine_args(),
            vllm_config,
        )


def test_qwen_adapter_rejects_unvalidated_vllm_version(monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.custom_encoder.adapter.qwen2_vl.version",
        lambda package: "99.0.0",
    )
    with pytest.raises(ValueError, match="no validated adapter"):
        create_custom_encoder_adapter(
            _QwenBackend(), _qwen_model_config(), _qwen_engine_args()
        )


@pytest.mark.parametrize(
    "encoding, message",
    [
        (torch.zeros((1, 4), dtype=torch.bfloat16), "Qwen2VLImageEncoding"),
        (_qwen_encoding(rows=2), "grid .* requires 1"),
        (_qwen_encoding(grid_thw=(2, 2, 2), rows=2), "T=1"),
        (_qwen_encoding(grid_thw=(1, 3, 2)), "divisible"),
        (
            Qwen2VLImageEncoding(torch.zeros((1, 3), dtype=torch.bfloat16), (1, 2, 2)),
            "hidden size 4",
        ),
        (
            Qwen2VLImageEncoding(torch.zeros((1, 4), dtype=torch.float16), (1, 2, 2)),
            "expected torch.bfloat16",
        ),
        (
            Qwen2VLImageEncoding(
                torch.full((1, 4), torch.nan, dtype=torch.bfloat16), (1, 2, 2)
            ),
            "NaN or Inf",
        ),
    ],
)
def test_qwen_adapter_validates_artifacts(monkeypatch, encoding, message):
    adapter = _qwen_adapter(monkeypatch)

    with pytest.raises((TypeError, ValueError), match=message):
        adapter.prepare_prompt(
            [_QWEN_START_TOKEN_ID, _QWEN_IMAGE_TOKEN_ID, _QWEN_END_TOKEN_ID],
            [encoding],
        )


@pytest.mark.parametrize(
    "token_ids, message",
    [
        (
            [_QWEN_START_TOKEN_ID, _QWEN_IMAGE_TOKEN_ID, _QWEN_IMAGE_TOKEN_ID],
            "canonical unexpanded",
        ),
        (
            [_QWEN_START_TOKEN_ID, _QWEN_IMAGE_TOKEN_ID, _QWEN_END_TOKEN_ID, 103],
            "video placeholders",
        ),
        ([_QWEN_IMAGE_TOKEN_ID], "canonical vision triple"),
    ],
)
def test_qwen_adapter_requires_canonical_placeholders(monkeypatch, token_ids, message):
    adapter = _qwen_adapter(monkeypatch)

    with pytest.raises(ValueError, match=message):
        adapter.prepare_prompt(token_ids, [_qwen_encoding()])


def test_qwen_adapter_rejects_processor_kwargs(monkeypatch):
    adapter = _qwen_adapter(monkeypatch)

    with pytest.raises(ValueError, match="mm_processor_kwargs"):
        adapter.prepare_prompt(
            [_QWEN_START_TOKEN_ID, _QWEN_IMAGE_TOKEN_ID, _QWEN_END_TOKEN_ID],
            [_qwen_encoding()],
            mm_processor_kwargs={"min_pixels": 1},
        )
