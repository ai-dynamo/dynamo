# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native external-multimodal adapter for Qwen2/2.5-VL decoders."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, Sequence

import torch
from vllm.inputs import EmbedsPrompt, TokensPrompt

from dynamo.vllm.multimodal_utils.custom_encoder.adapter.base import (
    CustomEncoderAdapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.model_config import (
    _hidden_size,
    _required_token_id,
    _spatial_merge_size,
)

_QWEN2_VLLM_VERSIONS = frozenset({"0.25.1"})
_QWEN2_ARCHITECTURES = frozenset(
    {
        "Qwen2VLForConditionalGeneration",
        "Qwen2_5_VLForConditionalGeneration",
    }
)


@dataclass(frozen=True)
class Qwen2VLImageEncoding:
    """Projected Qwen image features and the grid needed by its decoder."""

    projected: torch.Tensor
    grid_thw: tuple[int, int, int]


class _Qwen2VLNativeAdapter(CustomEncoderAdapter[Qwen2VLImageEncoding]):
    """Build native external-MM ``TokensPrompt`` inputs for Qwen2/2.5-VL."""

    def __init__(
        self,
        model_config: Any,
        engine_args: Any,
        vllm_config: Any | None,
    ) -> None:
        vllm_version = version("vllm").split("+", 1)[0]
        if vllm_version not in _QWEN2_VLLM_VERSIONS:
            raise ValueError(
                "Qwen CustomEncoder has no validated adapter for vLLM "
                f"{vllm_version}; supported versions are "
                f"{sorted(_QWEN2_VLLM_VERSIONS)}"
            )
        if not getattr(engine_args, "enable_mm_embeds", False):
            raise ValueError("Qwen CustomEncoder output requires --enable-mm-embeds")
        if getattr(engine_args, "language_model_only", False):
            raise ValueError(
                "Qwen CustomEncoder MVP requires the full registered model wrapper"
            )
        for name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
        ):
            if getattr(engine_args, name, 1) != 1:
                raise ValueError(f"Qwen CustomEncoder MVP requires {name}=1")

        compilation_config = getattr(engine_args, "compilation_config", None)
        encoder_cudagraphs = (
            compilation_config.get("cudagraph_mm_encoder", False)
            if isinstance(compilation_config, dict)
            else getattr(compilation_config, "cudagraph_mm_encoder", False)
        )
        if encoder_cudagraphs:
            raise ValueError(
                "Qwen external embeddings are incompatible with vLLM "
                "multimodal encoder CUDA graphs"
            )

        cache_config = getattr(vllm_config, "cache_config", None)
        prefix_caching = getattr(
            cache_config,
            "enable_prefix_caching",
            getattr(engine_args, "enable_prefix_caching", False),
        )
        if prefix_caching:
            raise ValueError(
                "Qwen CustomEncoder MVP requires --no-enable-prefix-caching"
            )
        scheduler_config = getattr(vllm_config, "scheduler_config", None)
        chunked_prefill = getattr(
            scheduler_config,
            "enable_chunked_prefill",
            getattr(engine_args, "enable_chunked_prefill", False),
        )
        if chunked_prefill:
            raise ValueError(
                "Qwen CustomEncoder MVP requires --no-enable-chunked-prefill"
            )

        self._hidden_size = _hidden_size(model_config)
        self._spatial_merge_size = _spatial_merge_size(model_config)
        model_dtype = getattr(model_config, "dtype", None)
        if not isinstance(model_dtype, torch.dtype):
            raise ValueError("CustomEncoder could not resolve the decoder dtype")
        self._dtype = model_dtype
        self._image_token_id = _required_token_id(model_config, "image_token_id")
        self._vision_start_token_id = _required_token_id(
            model_config, "vision_start_token_id"
        )
        self._vision_end_token_id = _required_token_id(
            model_config, "vision_end_token_id"
        )
        self._video_token_id = _required_token_id(model_config, "video_token_id")

    def prepare_prompt(
        self,
        token_ids: list[int],
        artifacts: Sequence[Qwen2VLImageEncoding],
        *,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> EmbedsPrompt | TokensPrompt:
        if mm_processor_kwargs:
            raise ValueError(
                "native Qwen CustomEncoder requests do not support "
                "mm_processor_kwargs; apply geometry options inside the encoder"
            )
        qwen_encodings: list[Qwen2VLImageEncoding] = []
        for index, encoding in enumerate(artifacts):
            if not isinstance(encoding, Qwen2VLImageEncoding):
                raise TypeError(
                    "Qwen CustomEncoder must return Qwen2VLImageEncoding; "
                    f"result {index} is {type(encoding).__name__}"
                )
            self._validate_encoding(index, encoding)
            qwen_encodings.append(encoding)
        if not qwen_encodings:
            raise ValueError("Qwen CustomEncoder encodings must not be empty")
        self._validate_placeholders(token_ids, len(qwen_encodings))

        return TokensPrompt(
            prompt_token_ids=token_ids,
            multi_modal_data={
                "image": {
                    "image_embeds": torch.cat(
                        [encoding.projected for encoding in qwen_encodings], dim=0
                    ),
                    "image_grid_thw": torch.tensor(
                        [encoding.grid_thw for encoding in qwen_encodings],
                        dtype=torch.int64,
                        device="cpu",
                    ),
                }
            },
        )

    def _validate_encoding(self, index: int, encoding: Qwen2VLImageEncoding) -> None:
        projected = encoding.projected
        if projected.dim() != 2 or projected.shape[1] != self._hidden_size:
            raise ValueError(
                f"Qwen image {index} has projected shape {tuple(projected.shape)}; "
                f"expected 2D with decoder hidden size {self._hidden_size}"
            )
        if projected.shape[0] < 1:
            raise ValueError(f"Qwen image {index} has no projected rows")
        if projected.dtype != self._dtype:
            raise ValueError(
                f"Qwen image {index} has dtype {projected.dtype}; "
                f"expected {self._dtype}"
            )
        if projected.device.type != "cpu":
            raise ValueError(f"Qwen image {index} must be on CPU")
        if not projected.is_contiguous():
            raise ValueError(f"Qwen image {index} must be contiguous")
        if projected.requires_grad:
            raise ValueError(f"Qwen image {index} must not require gradients")
        if not torch.isfinite(projected).all().item():
            raise ValueError(f"Qwen image {index} contains NaN or Inf")

        grid = encoding.grid_thw
        if (
            len(grid) != 3
            or any(
                not isinstance(value, int) or isinstance(value, bool) for value in grid
            )
            or any(value < 1 for value in grid)
        ):
            raise ValueError(
                f"Qwen image {index} grid_thw must contain three positive integers"
            )
        temporal, height, width = grid
        if temporal != 1:
            raise ValueError(f"Qwen image {index} requires image T=1")
        if height % self._spatial_merge_size or width % self._spatial_merge_size:
            raise ValueError(
                f"Qwen image {index} grid H/W must be divisible by "
                f"spatial_merge_size={self._spatial_merge_size}"
            )
        expected_rows = temporal * height * width // self._spatial_merge_size**2
        if projected.shape[0] != expected_rows:
            raise ValueError(
                f"Qwen image {index} has {projected.shape[0]} projected rows; "
                f"grid {grid} requires {expected_rows}"
            )

    def _validate_placeholders(self, token_ids: list[int], image_count: int) -> None:
        if self._video_token_id in token_ids:
            raise ValueError("Qwen CustomEncoder image MVP rejects video placeholders")
        triple = [
            self._vision_start_token_id,
            self._image_token_id,
            self._vision_end_token_id,
        ]
        positions = [
            index
            for index, token_id in enumerate(token_ids)
            if token_id == self._vision_start_token_id
        ]
        if len(positions) != image_count:
            raise ValueError(
                "Qwen CustomEncoder requires one canonical vision triple per image: "
                f"images={image_count}, starts={len(positions)}"
            )
        if any(token_ids[index : index + 3] != triple for index in positions):
            raise ValueError(
                "Qwen CustomEncoder requires canonical unexpanded "
                "<vision_start><image_pad><vision_end> groups"
            )
        if (
            token_ids.count(self._image_token_id) != image_count
            or token_ids.count(self._vision_end_token_id) != image_count
        ):
            raise ValueError(
                "Qwen CustomEncoder requires exactly one canonical vision triple "
                "per image"
            )
