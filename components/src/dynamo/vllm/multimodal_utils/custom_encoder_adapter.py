# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-selected adapters for in-process custom vision encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

import torch
from vllm.inputs import EmbedsPrompt, TokensPrompt

from dynamo.vllm.multimodal_utils.embed_assembler import build_mixed_embeds
from dynamo.vllm.multimodal_utils.vision_encoder_backend import VisionEncoderBackend


class CustomEncoderAdapter(ABC):
    """Translate encoder artifacts for one resolved downstream decoder."""

    @abstractmethod
    def prepare_prompt(
        self,
        token_ids: list[int],
        encodings: Sequence[torch.Tensor],
    ) -> EmbedsPrompt | TokensPrompt:
        """Validate encoder artifacts and build the final vLLM prompt."""


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
        raise ValueError("CustomEncoder could not resolve the decoder hidden size")
    return value


def _is_multimodal_model(model_config: Any) -> bool:
    value = getattr(model_config, "is_multimodal_model", False)
    return bool(value() if callable(value) else value)


class _LinearEmbedsAdapter(CustomEncoderAdapter):
    """Build mixed ``EmbedsPrompt`` inputs for a text-only decoder."""

    def __init__(
        self,
        backend: VisionEncoderBackend,
        model_config: Any,
        engine_args: Any,
    ) -> None:
        if model_config is None:
            raise ValueError("CustomEncoder requires the resolved vLLM ModelConfig")
        if _is_multimodal_model(model_config):
            raise ValueError(
                "CustomEncoder does not yet support this multimodal decoder; "
                "the linear EmbedsPrompt adapter is only valid for text-only models"
            )
        if not getattr(engine_args, "enable_prompt_embeds", False):
            raise ValueError(
                "text-only CustomEncoder output requires --enable-prompt-embeds"
            )
        image_token_id = getattr(backend, "image_token_id", None)
        if not isinstance(image_token_id, int) or isinstance(image_token_id, bool):
            raise ValueError(
                "text-only CustomEncoder output requires an integer image_token_id"
            )

        self._image_token_id = image_token_id
        self._hidden_size = _hidden_size(model_config)
        model_dtype = getattr(model_config, "dtype", None)
        self._dtype = model_dtype if isinstance(model_dtype, torch.dtype) else None

    def prepare_prompt(
        self,
        token_ids: list[int],
        encodings: Sequence[torch.Tensor],
    ) -> EmbedsPrompt | TokensPrompt:
        rows = list(encodings)
        for index, tensor in enumerate(rows):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    "text-only CustomEncoder must return tensors; "
                    f"result {index} is {type(tensor).__name__}"
                )
            if tensor.dim() != 2 or tensor.shape[1] != self._hidden_size:
                raise ValueError(
                    f"image tensor {index} has shape {tuple(tensor.shape)}; "
                    f"expected 2D with decoder hidden size {self._hidden_size}"
                )
            if self._dtype is not None and tensor.dtype != self._dtype:
                raise ValueError(
                    f"image tensor {index} has dtype {tensor.dtype}; "
                    f"expected decoder dtype {self._dtype}"
                )

        prompt_embeds, prompt_token_ids, prompt_is_token_ids = build_mixed_embeds(
            token_ids, rows, self._image_token_id
        )
        return EmbedsPrompt(
            prompt_embeds=prompt_embeds,
            prompt_token_ids=prompt_token_ids,
            prompt_is_token_ids=prompt_is_token_ids,
        )


def create_custom_encoder_adapter(
    backend: VisionEncoderBackend,
    model_config: Any,
    engine_args: Any,
    vllm_config: Any | None = None,
) -> CustomEncoderAdapter:
    """Create the adapter selected by the resolved downstream decoder.

    The first slice supports text-only decoders. ``vllm_config`` is accepted at
    this stable factory boundary for model-specific adapters added later.
    """

    del vllm_config
    return _LinearEmbedsAdapter(backend, model_config, engine_args)
