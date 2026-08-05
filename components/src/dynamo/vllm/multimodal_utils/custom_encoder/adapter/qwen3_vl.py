# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native external-multimodal adapter for Qwen3-VL decoders."""

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from vllm.inputs import TokensPrompt

from dynamo.vllm.multimodal_utils.custom_encoder.adapter.base import (
    CustomEncoderAdapter,
)
from dynamo.vllm.multimodal_utils.model_config import _model_architectures

_QWEN3_VL_ARCHITECTURES = frozenset(
    {
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
    }
)


def _is_qwen3_vl_model(model_config: Any) -> bool:
    """Whether vLLM resolved a supported dense or MoE Qwen3-VL model."""
    return any(
        architecture in _QWEN3_VL_ARCHITECTURES
        for architecture in _model_architectures(model_config)
    )


def _spatial_merge_size(model_config: Any) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = getattr(hf_config, "vision_config", None)
    value = getattr(vision_config, "spatial_merge_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("Qwen3-VL could not resolve spatial_merge_size")
    return value


@dataclass(frozen=True)
class Qwen3VLImageEncoding:
    """Packed Qwen3-VL image features and their pre-merge grid."""

    embeddings: torch.Tensor
    grid_thw: tuple[int, int, int]


class Qwen3VLNativeAdapter(CustomEncoderAdapter[Qwen3VLImageEncoding]):
    """Build a native external-MM ``TokensPrompt`` for Qwen3-VL."""

    def __init__(self, model_config: Any) -> None:
        self._spatial_merge_size = _spatial_merge_size(model_config)

    def prepare_prompt(
        self,
        token_ids: list[int],
        artifacts: Sequence[Qwen3VLImageEncoding],
    ) -> TokensPrompt:
        if not artifacts:
            raise ValueError("Qwen3-VL CustomEncoder artifacts must not be empty")

        encodings: list[Qwen3VLImageEncoding] = []
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, Qwen3VLImageEncoding):
                raise TypeError(
                    "Qwen3-VL CustomEncoder must return Qwen3VLImageEncoding; "
                    f"result {index} is {type(artifact).__name__}"
                )
            self._validate_encoding(index, artifact)
            encodings.append(artifact)

        return TokensPrompt(
            prompt_token_ids=token_ids,
            multi_modal_data={
                "image": {
                    "image_embeds": torch.cat(
                        [encoding.embeddings for encoding in encodings], dim=0
                    ),
                    "image_grid_thw": torch.tensor(
                        [encoding.grid_thw for encoding in encodings],
                        dtype=torch.int64,
                        device="cpu",
                    ),
                }
            },
        )

    def _validate_encoding(self, index: int, encoding: Qwen3VLImageEncoding) -> None:
        embeddings = encoding.embeddings
        if not isinstance(embeddings, torch.Tensor):
            raise TypeError(f"Qwen3-VL image {index} embeddings must be a tensor")
        if embeddings.dim() != 2:
            raise ValueError(
                f"Qwen3-VL image {index} embeddings must be 2D; "
                f"got shape {tuple(embeddings.shape)}"
            )
        if embeddings.shape[0] < 1:
            raise ValueError(f"Qwen3-VL image {index} has no embedding rows")
        if embeddings.device.type != "cpu":
            raise ValueError(f"Qwen3-VL image {index} embeddings must be on CPU")

        grid = encoding.grid_thw
        if (
            not isinstance(grid, tuple)
            or len(grid) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in grid
            )
            or any(value < 1 for value in grid)
        ):
            raise ValueError(
                f"Qwen3-VL image {index} grid_thw must contain three positive integers"
            )

        temporal, height, width = grid
        merge_size = self._spatial_merge_size
        if height % merge_size or width % merge_size:
            raise ValueError(
                f"Qwen3-VL image {index} grid H/W must be divisible by "
                f"spatial_merge_size={merge_size}"
            )
        expected_rows = temporal * height * width // merge_size**2
        if embeddings.shape[0] != expected_rows:
            raise ValueError(
                f"Qwen3-VL image {index} has {embeddings.shape[0]} embedding rows; "
                f"grid {grid} requires {expected_rows}"
            )
