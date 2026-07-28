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

"""Build native vLLM prompts from external Qwen vision artifacts."""

from __future__ import annotations

from importlib.metadata import version as package_version
from typing import Any, Mapping

import torch
from vllm.inputs import TokensPrompt

from dynamo.vllm.multimodal_utils.external_qwen_artifact import ExternalQwenArtifact

_SUPPORTED_VLLM_VERSIONS = frozenset({"0.25.1"})
_SUPPORTED_ARCHITECTURES = frozenset(
    {
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
    }
)


def _hidden_size(model_config: Any) -> int:
    getter = getattr(model_config, "get_hidden_size", None)
    value = getter() if callable(getter) else None
    if value is None:
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None)
        value = getattr(text_config, "hidden_size", None)
        if value is None:
            value = getattr(hf_config, "hidden_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("external Qwen adapter could not resolve decoder hidden size")
    return value


def _model_architectures(model_config: Any) -> tuple[str, ...]:
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if architectures is None:
        architectures = getattr(model_config, "architectures", None)
    return tuple(str(architecture) for architecture in (architectures or ()))


def _spatial_merge_size(model_config: Any) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = getattr(hf_config, "vision_config", None)
    value = getattr(vision_config, "spatial_merge_size", None)
    if value is None:
        value = getattr(hf_config, "spatial_merge_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("external Qwen adapter could not resolve spatial_merge_size")
    return value


def _required_token_id(model_config: Any, name: str) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    value = getattr(hf_config, name, None)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"external Qwen adapter could not resolve {name}")
    return value


def _validate_runtime(engine_args: Any, vllm_config: Any) -> None:
    vllm_version = package_version("vllm").split("+", 1)[0]
    if vllm_version not in _SUPPORTED_VLLM_VERSIONS:
        raise ValueError(
            "external Qwen artifacts have no validated adapter for vLLM "
            f"{vllm_version}; supported versions are "
            f"{sorted(_SUPPORTED_VLLM_VERSIONS)}"
        )
    if not getattr(engine_args, "enable_mm_embeds", False):
        raise ValueError("external Qwen artifacts require --enable-mm-embeds")
    if not getattr(engine_args, "enforce_eager", False):
        raise ValueError("external Qwen artifacts require --enforce-eager")
    if getattr(engine_args, "language_model_only", False):
        raise ValueError(
            "external Qwen artifacts require the full registered model wrapper"
        )
    for name in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
    ):
        if getattr(engine_args, name, 1) != 1:
            raise ValueError(f"external Qwen artifacts require {name}=1")

    compilation_config = getattr(engine_args, "compilation_config", None)
    encoder_cudagraphs = (
        compilation_config.get("cudagraph_mm_encoder", False)
        if isinstance(compilation_config, dict)
        else getattr(compilation_config, "cudagraph_mm_encoder", False)
    )
    if encoder_cudagraphs:
        raise ValueError(
            "external Qwen artifacts are incompatible with multimodal encoder "
            "CUDA graphs"
        )

    cache_config = getattr(vllm_config, "cache_config", None)
    prefix_caching = getattr(
        cache_config,
        "enable_prefix_caching",
        getattr(engine_args, "enable_prefix_caching", False),
    )
    if prefix_caching:
        raise ValueError("external Qwen artifacts require --no-enable-prefix-caching")
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    chunked_prefill = getattr(
        scheduler_config,
        "enable_chunked_prefill",
        getattr(engine_args, "enable_chunked_prefill", False),
    )
    if chunked_prefill:
        raise ValueError("external Qwen artifacts require --no-enable-chunked-prefill")


def _validate_placeholders(
    token_ids: list[int],
    image_count: int,
    *,
    image_token_id: int,
    vision_start_token_id: int,
    vision_end_token_id: int,
    video_token_id: int,
) -> None:
    if video_token_id in token_ids:
        raise ValueError("external Qwen image artifacts reject video placeholders")

    canonical = [vision_start_token_id, image_token_id, vision_end_token_id]
    positions = [
        index
        for index, token_id in enumerate(token_ids)
        if token_id == vision_start_token_id
    ]
    if len(positions) != image_count:
        raise ValueError(
            "external Qwen artifacts require one canonical vision triple per image: "
            f"images={image_count}, starts={len(positions)}"
        )
    if any(token_ids[index : index + 3] != canonical for index in positions):
        raise ValueError(
            "external Qwen artifacts require canonical unexpanded "
            "<vision_start><image_pad><vision_end> groups"
        )
    if (
        token_ids.count(image_token_id) != image_count
        or token_ids.count(vision_end_token_id) != image_count
    ):
        raise ValueError(
            "external Qwen artifacts require exactly one canonical vision triple "
            "per image"
        )


def build_external_qwen_prompt(
    *,
    external_mm_data: Mapping[str, Any],
    token_ids: list[int],
    model_name: str,
    model_config: Any,
    engine_args: Any,
    vllm_config: Any,
    enable_multimodal: bool,
) -> TokensPrompt:
    """Validate an external artifact and build a native Qwen ``TokensPrompt``."""

    if model_config is None:
        raise ValueError("external Qwen artifacts require the vLLM ModelConfig")
    if not enable_multimodal:
        raise ValueError("external Qwen artifacts require --enable-multimodal")
    _validate_runtime(engine_args, vllm_config)

    architectures = _model_architectures(model_config)
    supported = [
        architecture
        for architecture in architectures
        if architecture in _SUPPORTED_ARCHITECTURES
    ]
    if len(supported) != 1:
        raise ValueError(
            "external Qwen artifacts require exactly one supported Qwen2/2.5-VL "
            f"architecture; got {architectures}"
        )

    artifact = ExternalQwenArtifact.from_dict(external_mm_data)
    if artifact.model != model_name:
        raise ValueError(
            f"external Qwen artifact model {artifact.model!r} does not match "
            f"worker model {model_name!r}"
        )
    if list(artifact.prompt_token_ids) != token_ids:
        raise ValueError(
            "external Qwen artifact prompt_token_ids do not match request token_ids"
        )

    image_embeds = artifact.load_image_embeds()
    hidden_size = _hidden_size(model_config)
    if image_embeds.shape[1] != hidden_size:
        raise ValueError(
            f"external Qwen image embeddings have hidden size {image_embeds.shape[1]}; "
            f"expected {hidden_size}"
        )
    model_dtype = getattr(model_config, "dtype", None)
    if not isinstance(model_dtype, torch.dtype):
        raise ValueError("external Qwen adapter could not resolve decoder dtype")
    if model_dtype != torch.bfloat16:
        raise ValueError(
            f"external Qwen artifacts require a BF16 decoder; got {model_dtype}"
        )
    if image_embeds.dtype != model_dtype:
        raise ValueError(
            f"external Qwen image embeddings use {image_embeds.dtype}; "
            f"expected {model_dtype}"
        )

    spatial_merge_size = _spatial_merge_size(model_config)
    expected_rows = 0
    for index, (temporal, height, width) in enumerate(artifact.image_grid_thw):
        if temporal != 1:
            raise ValueError(f"external Qwen image {index} requires T=1")
        if height % spatial_merge_size or width % spatial_merge_size:
            raise ValueError(
                f"external Qwen image {index} grid H/W must be divisible by "
                f"spatial_merge_size={spatial_merge_size}"
            )
        expected_rows += (
            temporal * height * width // spatial_merge_size // spatial_merge_size
        )
    if image_embeds.shape[0] != expected_rows:
        raise ValueError(
            f"external Qwen artifact has {image_embeds.shape[0]} projected rows; "
            f"grid metadata requires {expected_rows}"
        )

    _validate_placeholders(
        token_ids,
        len(artifact.image_grid_thw),
        image_token_id=_required_token_id(model_config, "image_token_id"),
        vision_start_token_id=_required_token_id(model_config, "vision_start_token_id"),
        vision_end_token_id=_required_token_id(model_config, "vision_end_token_id"),
        video_token_id=_required_token_id(model_config, "video_token_id"),
    )

    return TokensPrompt(
        prompt_token_ids=token_ids,
        multi_modal_data={
            "image": {
                "image_embeds": image_embeds,
                "image_grid_thw": torch.tensor(
                    artifact.image_grid_thw,
                    dtype=torch.int64,
                    device="cpu",
                ),
            }
        },
    )
