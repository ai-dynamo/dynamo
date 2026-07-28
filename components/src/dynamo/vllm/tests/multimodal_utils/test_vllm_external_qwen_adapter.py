# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from typing import Sequence

import pytest
import torch

from dynamo.vllm.multimodal_utils import external_qwen_artifact as artifact_module
from dynamo.vllm.multimodal_utils.external_qwen_adapter import (
    build_external_qwen_prompt,
)
from dynamo.vllm.multimodal_utils.external_qwen_artifact import (
    EXTERNAL_QWEN_ARTIFACT_FORMAT,
    ExternalQwenArtifact,
    deserialize_image_embeds,
    serialize_image_embeds,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_VISION_START_TOKEN_ID = 100
_IMAGE_TOKEN_ID = 101
_VISION_END_TOKEN_ID = 102
_VIDEO_TOKEN_ID = 103
_TOKEN_IDS = (_VISION_START_TOKEN_ID, _IMAGE_TOKEN_ID, _VISION_END_TOKEN_ID)


def _model_config(hidden_size: int = 4, dtype: torch.dtype = torch.bfloat16):
    return SimpleNamespace(
        dtype=dtype,
        get_hidden_size=lambda: hidden_size,
        hf_config=SimpleNamespace(
            architectures=["Qwen2_5_VLForConditionalGeneration"],
            image_token_id=_IMAGE_TOKEN_ID,
            vision_start_token_id=_VISION_START_TOKEN_ID,
            vision_end_token_id=_VISION_END_TOKEN_ID,
            video_token_id=_VIDEO_TOKEN_ID,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
    )


def _engine_args(**overrides):
    values = {
        "enable_mm_embeds": True,
        "enforce_eager": True,
        "language_model_only": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _vllm_config(
    *, enable_prefix_caching: bool = False, enable_chunked_prefill: bool = False
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=enable_chunked_prefill),
    )


def _artifact(
    image_embeds: torch.Tensor | None = None,
    *,
    token_ids: Sequence[int] | None = None,
    grid: list[list[int]] | None = None,
):
    return ExternalQwenArtifact.create(
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        prompt_token_ids=token_ids or _TOKEN_IDS,
        image_embeds=(
            image_embeds
            if image_embeds is not None
            else torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)
        ),
        image_grid_thw=grid or [[1, 2, 2]],
    )


def _build_prompt(monkeypatch, artifact: ExternalQwenArtifact, **overrides):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.external_qwen_adapter.package_version",
        lambda package: "0.25.1",
    )
    kwargs = {
        "external_mm_data": artifact.to_dict(),
        "token_ids": list(artifact.prompt_token_ids),
        "model_name": "Qwen/Qwen2.5-VL-3B-Instruct",
        "model_config": _model_config(),
        "engine_args": _engine_args(),
        "vllm_config": _vllm_config(),
        "enable_multimodal": True,
    }
    kwargs.update(overrides)
    return build_external_qwen_prompt(**kwargs)


def test_artifact_round_trip_preserves_tensor_and_metadata():
    image_embeds = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    artifact = _artifact(image_embeds, grid=[[1, 2, 4]])

    restored = ExternalQwenArtifact.from_dict(artifact.to_dict())

    assert restored.model == artifact.model
    assert restored.prompt_token_ids == tuple(_TOKEN_IDS)
    assert restored.image_grid_thw == ((1, 2, 4),)
    torch.testing.assert_close(restored.load_image_embeds(), image_embeds)


@pytest.mark.parametrize(
    ("image_embeds", "match"),
    [
        (torch.zeros((1, 4), dtype=torch.float16), "torch.bfloat16"),
        (torch.zeros((1, 1, 4), dtype=torch.bfloat16), "2D tensor"),
        (torch.full((1, 4), torch.nan, dtype=torch.bfloat16), "NaN or Inf"),
    ],
)
def test_artifact_rejects_invalid_tensors(image_embeds, match):
    with pytest.raises(ValueError, match=match):
        serialize_image_embeds(image_embeds)


def test_artifact_rejects_oversized_encoded_payload(monkeypatch):
    payload = _artifact().to_dict()
    monkeypatch.setattr(artifact_module, "_MAX_BASE64_LENGTH", 4)

    with pytest.raises(ValueError, match="32 MiB"):
        ExternalQwenArtifact.from_dict(payload)

    with pytest.raises(ValueError, match="32 MiB"):
        deserialize_image_embeds("A" * 8)


def test_artifact_rejects_oversized_serialized_payload(monkeypatch):
    monkeypatch.setattr(artifact_module, "MAX_EXTERNAL_QWEN_ARTIFACT_BYTES", 1)

    with pytest.raises(ValueError, match="32 MiB"):
        serialize_image_embeds(torch.zeros((1, 4), dtype=torch.bfloat16))


def test_artifact_rejects_invalid_wire_shape():
    payload = _artifact().to_dict()
    payload["unexpected"] = True

    with pytest.raises(ValueError, match="unsupported fields"):
        ExternalQwenArtifact.from_dict(payload)

    payload.pop("unexpected")
    payload["format"] = "unknown"
    with pytest.raises(ValueError, match=EXTERNAL_QWEN_ARTIFACT_FORMAT):
        ExternalQwenArtifact.from_dict(payload)

    payload["format"] = EXTERNAL_QWEN_ARTIFACT_FORMAT
    payload["prompt_token_ids"] = [100, -1, 102]
    with pytest.raises(ValueError, match="non-negative integers"):
        ExternalQwenArtifact.from_dict(payload)


def test_adapter_builds_native_qwen_tokens_prompt(monkeypatch):
    artifact = _artifact()

    prompt = _build_prompt(monkeypatch, artifact)

    assert prompt["prompt_token_ids"] == list(_TOKEN_IDS)
    image = prompt["multi_modal_data"]["image"]
    assert image["image_embeds"].shape == (1, 4)
    assert image["image_embeds"].dtype == torch.bfloat16
    assert image["image_grid_thw"].tolist() == [[1, 2, 2]]


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"enable_multimodal": False}, "enable-multimodal"),
        ({"engine_args": _engine_args(enable_mm_embeds=False)}, "enable-mm-embeds"),
        ({"engine_args": _engine_args(enforce_eager=False)}, "enforce-eager"),
        (
            {"engine_args": _engine_args(tensor_parallel_size=2)},
            "tensor_parallel_size=1",
        ),
        (
            {"vllm_config": _vllm_config(enable_prefix_caching=True)},
            "no-enable-prefix-caching",
        ),
        (
            {"vllm_config": _vllm_config(enable_chunked_prefill=True)},
            "no-enable-chunked-prefill",
        ),
    ],
)
def test_adapter_rejects_unsupported_runtime_modes(monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        _build_prompt(monkeypatch, _artifact(), **overrides)


@pytest.mark.parametrize(
    ("artifact_factory", "overrides_factory", "match"),
    [
        (
            lambda: _artifact(torch.zeros((1, 3), dtype=torch.bfloat16)),
            dict,
            "hidden size 3",
        ),
        (
            _artifact,
            lambda: {"model_config": _model_config(dtype=torch.float16)},
            "BF16 decoder",
        ),
        (
            lambda: _artifact(torch.zeros((2, 4), dtype=torch.bfloat16)),
            dict,
            "grid metadata requires 1",
        ),
        (
            lambda: _artifact(grid=[[2, 2, 2]]),
            dict,
            "requires T=1",
        ),
        (
            lambda: _artifact(token_ids=[_IMAGE_TOKEN_ID]),
            dict,
            "canonical vision triple",
        ),
        (
            _artifact,
            lambda: {"token_ids": [1, 2, 3]},
            "do not match request token_ids",
        ),
        (
            _artifact,
            lambda: {"model_name": "other/model"},
            "does not match worker model",
        ),
    ],
)
def test_adapter_validates_artifact_semantics(
    monkeypatch,
    artifact_factory,
    overrides_factory,
    match,
):
    with pytest.raises(ValueError, match=match):
        _build_prompt(
            monkeypatch,
            artifact_factory(),
            **overrides_factory(),
        )


def test_adapter_rejects_unvalidated_vllm_version(monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.multimodal_utils.external_qwen_adapter.package_version",
        lambda package: "99.0.0",
    )

    with pytest.raises(ValueError, match="no validated adapter"):
        build_external_qwen_prompt(
            external_mm_data=_artifact().to_dict(),
            token_ids=list(_TOKEN_IDS),
            model_name="Qwen/Qwen2.5-VL-3B-Instruct",
            model_config=_model_config(),
            engine_args=_engine_args(),
            vllm_config=_vllm_config(),
            enable_multimodal=True,
        )
