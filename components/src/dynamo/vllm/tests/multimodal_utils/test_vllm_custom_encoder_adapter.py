# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder_adapter import (
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    Qwen2VLImageEncoding,
    VisionEncoderBackend,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_IMAGE_TOKEN_ID = 99
_QWEN_START_TOKEN_ID = 100
_QWEN_IMAGE_TOKEN_ID = 101
_QWEN_END_TOKEN_ID = 102
_QWEN_VIDEO_TOKEN_ID = 103


class _Backend(VisionEncoderBackend):
    image_token_id = _IMAGE_TOKEN_ID

    def build(self, model_id: str) -> None:
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


class _QwenBackend(_Backend):
    output_format = "qwen2_vl_projected_grid"
    image_token_id = None


def _model_config(*, multimodal: bool = False, callable_flag: bool = False):
    return SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        is_multimodal_model=(lambda: multimodal) if callable_flag else multimodal,
    )


def _engine_args(*, enable_prompt_embeds: bool = True):
    return SimpleNamespace(enable_prompt_embeds=enable_prompt_embeds)


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
        "dynamo.vllm.multimodal_utils.custom_encoder_adapter.version",
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


def test_text_decoder_selects_linear_adapter_and_builds_final_prompt():
    adapter = create_custom_encoder_adapter(_Backend(), _model_config(), _engine_args())

    prompt = adapter.prepare_prompt(
        [1, _IMAGE_TOKEN_ID, 2],
        [torch.ones((2, 4), dtype=torch.bfloat16)],
    )

    assert tuple(prompt["prompt_embeds"].shape) == (4, 4)
    assert prompt["prompt_token_ids"] == [1, 99, 99, 2]
    assert prompt["prompt_is_token_ids"] == [True, False, False, True]


def test_linear_adapter_requires_prompt_embeds_flag():
    with pytest.raises(ValueError, match="--enable-prompt-embeds"):
        create_custom_encoder_adapter(
            _Backend(),
            _model_config(),
            _engine_args(enable_prompt_embeds=False),
        )


def test_linear_adapter_rejects_multimodal_decoder():
    with pytest.raises(ValueError, match="multimodal decoder"):
        create_custom_encoder_adapter(
            _Backend(), _model_config(multimodal=True), _engine_args()
        )


def test_linear_adapter_calls_real_model_config_multimodal_method():
    adapter = create_custom_encoder_adapter(
        _Backend(), _model_config(callable_flag=True), _engine_args()
    )

    assert adapter is not None


@pytest.mark.parametrize(
    "encoding, match",
    [
        (torch.ones((2, 3), dtype=torch.bfloat16), "decoder hidden size 4"),
        (torch.ones((2, 4), dtype=torch.float16), "decoder dtype"),
        ("not-a-tensor", "must return tensors"),
    ],
)
def test_linear_adapter_validates_encoder_artifacts(encoding, match):
    adapter = create_custom_encoder_adapter(_Backend(), _model_config(), _engine_args())

    with pytest.raises((TypeError, ValueError), match=match):
        adapter.prepare_prompt([_IMAGE_TOKEN_ID], [encoding])


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
    ],
)
def test_qwen_decoder_selects_native_adapter(monkeypatch, architecture):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.custom_encoder_adapter.version",
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


def test_qwen_decoder_rejects_tensor_output_format(monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.custom_encoder_adapter.version",
        lambda package: "0.25.1",
    )
    with pytest.raises(ValueError, match="qwen2_vl_projected_grid"):
        create_custom_encoder_adapter(
            _Backend(), _qwen_model_config(), _qwen_engine_args()
        )


def test_text_decoder_rejects_native_output_format():
    with pytest.raises(ValueError, match="output_format='tensor'"):
        create_custom_encoder_adapter(_QwenBackend(), _model_config(), _engine_args())


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
        "dynamo.vllm.multimodal_utils.custom_encoder_adapter.version",
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
        "dynamo.vllm.multimodal_utils.custom_encoder_adapter.version",
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
