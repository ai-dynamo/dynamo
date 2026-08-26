# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn an external encoder result into a vLLM prompt."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs import EmbedsPrompt

from dynamo.common.external_encoder import (
    ExternalEncoderResult,
    decode_request_plane_tensor,
)
from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.linear import (
    build_mixed_embeds,
)
from dynamo.vllm.multimodal_utils.model_config import _hidden_size, _is_multimodal_model


class ExternalEncoderPromptLoader:
    """Decode packed visual features and build a mixed ``EmbedsPrompt``."""

    def __init__(self, model_config: Any, engine_args: Any) -> None:
        if model_config is None:
            raise RuntimeError(
                "external encoder results require the resolved vLLM ModelConfig"
            )
        if _is_multimodal_model(model_config):
            raise RuntimeError(
                "external linear embeddings currently require a text-only decoder"
            )
        if not getattr(engine_args, "enable_prompt_embeds", False):
            raise RuntimeError(
                "external linear embeddings require --enable-prompt-embeds"
            )

        self._hidden_size = _hidden_size(model_config)
        model_dtype = getattr(model_config, "dtype", None)
        self._dtype = model_dtype if isinstance(model_dtype, torch.dtype) else None

    async def load(
        self,
        encoder_result: Mapping[str, Any],
        token_ids: list[int],
    ) -> EmbedsPrompt:
        """Decode one packed feature tensor and adapt it to vLLM mixed mode."""

        try:
            parsed = ExternalEncoderResult.from_dict(encoder_result)
            packed = decode_request_plane_tensor(parsed.features)
            if packed.shape[1] != self._hidden_size:
                raise ValueError(
                    f"external encoder tensor has shape {tuple(packed.shape)}; "
                    f"expected 2D with decoder hidden size {self._hidden_size}"
                )
            if self._dtype is not None and packed.dtype != self._dtype:
                raise ValueError(
                    f"external encoder tensor has dtype {packed.dtype}; "
                    f"expected decoder dtype {self._dtype}"
                )
            if parsed.row_splits[-1] != packed.shape[0]:
                raise ValueError(
                    "external encoder row_splits do not cover the imported tensor rows"
                )

            rows = [
                packed[start:end]
                for start, end in zip(parsed.row_splits, parsed.row_splits[1:])
            ]
            if any(row.shape[0] == 0 for row in rows):
                raise ValueError("external encoder row_splits contain an empty image")
            prompt_embeds, prompt_token_ids, prompt_is_token_ids = build_mixed_embeds(
                token_ids,
                rows,
                parsed.image_token_id,
            )
        except (TypeError, ValueError) as error:
            raise InvalidArgument(
                f"invalid external encoder result: {error}"
            ) from error
        return EmbedsPrompt(
            prompt_embeds=prompt_embeds,
            prompt_token_ids=prompt_token_ids,
            prompt_is_token_ids=prompt_is_token_ids,
        )
