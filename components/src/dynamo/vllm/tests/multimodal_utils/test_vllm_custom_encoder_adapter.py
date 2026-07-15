# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for model-aware custom-encoder engine adapters."""

from types import SimpleNamespace

import pytest
import torch

from dynamo.vllm.multimodal_utils.custom_encoder_adapter import (
    BoundCustomEncoderAdapter,
    MixedEmbedsPlan,
    NativeMMPlan,
    reconcile_and_canonicalize,
)
from dynamo.vllm.multimodal_utils.vision_encoder_backend import (
    BackendEncodingSpecV1,
    EncodedMediaResultV1,
    ForwardItemV1,
    Qwen2VLImageEncodingV1,
    VisionEncoderBackend,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_START = 100
_IMAGE = 101
_END = 102


def _decoder_fingerprint(
    architecture: str = "Qwen2_5_VLForConditionalGeneration",
) -> str:
    return (
        f"model=None:revision=None:{architecture}:hidden=4:merge=2:"
        "dtype=torch.bfloat16"
    )


def _qwen_spec(
    architecture: str = "Qwen2_5_VLForConditionalGeneration", **overrides
) -> BackendEncodingSpecV1:
    values = {
        "adapter_abi": "vllm-qwen2-vl-external-v1",
        "producer_fingerprint": "encoder-revision-1",
        "expected_decoder_config_fingerprint": _decoder_fingerprint(architecture),
        "output_dtype": "bfloat16",
        "hidden_size": 4,
        "spatial_merge_size": 2,
    }
    values.update(overrides)
    return BackendEncodingSpecV1(**values)


def _model_config(
    architecture: str = "Qwen2_5_VLForConditionalGeneration",
):
    return SimpleNamespace(
        dtype=torch.bfloat16,
        get_hidden_size=lambda: 4,
        hf_config=SimpleNamespace(
            architectures=[architecture],
            image_token_id=_IMAGE,
            vision_start_token_id=_START,
            vision_end_token_id=_END,
            video_token_id=103,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
    )


def _engine_args(**overrides):
    values = {
        "enable_mm_embeds": True,
        "enable_prompt_embeds": False,
        "language_model_only": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "compilation_config": SimpleNamespace(cudagraph_mm_encoder=False),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Backend(VisionEncoderBackend):
    def __init__(self, encoding_spec=None, image_token_id=None):
        self.encoding_spec = encoding_spec
        self.image_token_id = image_token_id

    def build(self, model_id):
        pass

    def forward_batch(self, items, target_bucket=None):
        raise NotImplementedError


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
    ],
)
def test_qwen2_family_architectures_bind_to_native_adapter(architecture):
    adapter = BoundCustomEncoderAdapter(
        _Backend(_qwen_spec(architecture)),
        _model_config(architecture),
        _engine_args(),
    )

    assert adapter.uses_native_multimodal is True


def test_qwen_adapter_builds_native_tokens_prompt_payload_in_image_order():
    backend = _Backend(_qwen_spec())
    adapter = BoundCustomEncoderAdapter(backend, _model_config(), _engine_args())
    token_ids = [_START, _IMAGE, _END, 7, _START, _IMAGE, _END]
    first = torch.full((1, 4), 1, dtype=torch.bfloat16)
    second = torch.full((2, 4), 2, dtype=torch.bfloat16)

    plan = adapter.prepare_prompt_plan(
        token_ids,
        [
            Qwen2VLImageEncodingV1(first, (1, 2, 2)),
            Qwen2VLImageEncodingV1(second, (1, 2, 4)),
        ],
    )

    assert isinstance(plan, NativeMMPlan)
    image = plan.multi_modal_data["image"]
    assert image["image_embeds"].shape == (3, 4)
    assert image["image_embeds"][:, 0].tolist() == [1, 2, 2]
    assert image["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 4]]
    assert token_ids == [_START, _IMAGE, _END, 7, _START, _IMAGE, _END]


def test_typed_results_are_reordered_by_correlation_and_cloned():
    spec = _qwen_spec()
    submitted = [ForwardItemV1(b"first", "a"), ForwardItemV1(b"second", "b")]
    first = torch.full((1, 4), 1, dtype=torch.bfloat16)
    second = torch.full((2, 4), 2, dtype=torch.bfloat16)
    returned = [
        EncodedMediaResultV1(b"second", Qwen2VLImageEncodingV1(second, (1, 2, 4))),
        EncodedMediaResultV1(b"first", Qwen2VLImageEncodingV1(first, (1, 2, 2))),
    ]

    outputs = reconcile_and_canonicalize(spec, submitted, returned)

    assert [output.projected[0, 0].item() for output in outputs] == [1, 2]
    assert outputs[0].projected.data_ptr() != first.data_ptr()
    first.fill_(9)
    assert outputs[0].projected[0, 0].item() == 1


def test_qwen_result_rejects_grid_row_count_mismatch():
    with pytest.raises(ValueError, match="row count"):
        reconcile_and_canonicalize(
            _qwen_spec(),
            [ForwardItemV1(b"id", "item")],
            [
                EncodedMediaResultV1(
                    b"id",
                    Qwen2VLImageEncodingV1(
                        torch.zeros((2, 4), dtype=torch.bfloat16),
                        (1, 2, 2),
                    ),
                )
            ],
        )


def test_typed_results_reject_duplicate_correlation_id():
    submitted = [ForwardItemV1(b"a", 1), ForwardItemV1(b"b", 2)]
    media = Qwen2VLImageEncodingV1(torch.zeros((1, 4), dtype=torch.bfloat16), (1, 2, 2))
    with pytest.raises(ValueError, match="duplicate"):
        reconcile_and_canonicalize(
            _qwen_spec(),
            submitted,
            [EncodedMediaResultV1(b"a", media), EncodedMediaResultV1(b"a", media)],
        )


def test_qwen_adapter_requires_mm_embeds_flag():
    with pytest.raises(ValueError, match="enable-mm-embeds"):
        BoundCustomEncoderAdapter(
            _Backend(_qwen_spec()),
            _model_config(),
            _engine_args(enable_mm_embeds=False),
        )


def test_qwen_adapter_requires_decoder_fingerprint():
    with pytest.raises(ValueError, match="expected_decoder_config_fingerprint"):
        BoundCustomEncoderAdapter(
            _Backend(_qwen_spec(expected_decoder_config_fingerprint=None)),
            _model_config(),
            _engine_args(),
        )


def test_qwen_adapter_rejects_encoder_cuda_graphs():
    with pytest.raises(ValueError, match="encoder CUDA graphs"):
        BoundCustomEncoderAdapter(
            _Backend(_qwen_spec()),
            _model_config(),
            _engine_args(compilation_config=SimpleNamespace(cudagraph_mm_encoder=True)),
        )


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
def test_qwen_adapter_rejects_unproven_resolved_engine_modes(vllm_config, message):
    with pytest.raises(ValueError, match=message):
        BoundCustomEncoderAdapter(
            _Backend(_qwen_spec()),
            _model_config(),
            _engine_args(),
            vllm_config,
        )


def test_qwen_adapter_rejects_unvalidated_vllm_version(monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.custom_encoder_adapter.version",
        lambda package: "99.0.0",
    )
    with pytest.raises(ValueError, match="no validated adapter"):
        BoundCustomEncoderAdapter(
            _Backend(_qwen_spec()),
            _model_config(),
            _engine_args(),
        )


def test_qwen_architecture_rejects_linear_adapter():
    with pytest.raises(ValueError, match="requires the native"):
        BoundCustomEncoderAdapter(
            _Backend(encoding_spec=None, image_token_id=_IMAGE),
            _model_config(),
            _engine_args(enable_prompt_embeds=True),
        )


@pytest.mark.parametrize(
    "token_ids, message",
    [
        ([_START, _IMAGE, _IMAGE, _END], "canonical vision triple"),
        ([_START, _IMAGE, _END, _START], "canonical vision triple"),
        ([_START, _IMAGE, _END, 103], "video placeholders"),
    ],
)
def test_qwen_adapter_rejects_noncanonical_placeholders(token_ids, message):
    adapter = BoundCustomEncoderAdapter(
        _Backend(_qwen_spec()), _model_config(), _engine_args()
    )
    media = Qwen2VLImageEncodingV1(torch.zeros((1, 4), dtype=torch.bfloat16), (1, 2, 2))
    media_items = [media] * token_ids.count(_IMAGE)
    with pytest.raises(ValueError, match=message):
        adapter.prepare_prompt_plan(token_ids, media_items)


def test_legacy_linear_backend_keeps_mixed_embeds_plan():
    adapter = BoundCustomEncoderAdapter(
        _Backend(encoding_spec=None, image_token_id=42),
        _model_config("Qwen2ForCausalLM"),
        _engine_args(enable_prompt_embeds=True),
    )

    plan = adapter.prepare_prompt_plan(
        [1, 42, 2], [torch.zeros((2, 4), dtype=torch.float32)]
    )

    assert isinstance(plan, MixedEmbedsPlan)
    assert plan.prompt_token_ids == [1, 42, 42, 2]
