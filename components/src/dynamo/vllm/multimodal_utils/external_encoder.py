# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn a workflow-produced external encoder result into a vLLM prompt."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

import torch
from vllm.inputs import EmbedsPrompt

from dynamo.common.external_encoder import ExternalEncoderResult
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.linear import (
    build_mixed_embeds,
)
from dynamo.vllm.multimodal_utils.model_config import _hidden_size, _is_multimodal_model


class TensorImporter(Protocol):
    """Consumer-side subset of the workflow NIXL carrier."""

    async def import_tensor(self, reference: Mapping[str, Any]) -> Any:
        ...


def _make_cpu_nixl_importer() -> TensorImporter:
    # Keep the workflow/NIXL dependency off the ordinary vLLM startup path.
    # The package (and its native connector) is loaded only when the first
    # request actually carries an external encoder result.
    from dynamo.experimental.workflow.nixl import NixlTensorCarrier

    return NixlTensorCarrier(receive_device="cpu")


class ExternalEncoderPromptLoader:
    """Import packed visual features and build a mixed ``EmbedsPrompt``."""

    def __init__(
        self,
        model_config: Any,
        engine_args: Any,
        *,
        importer_factory: Callable[[], TensorImporter] = _make_cpu_nixl_importer,
    ) -> None:
        if model_config is None:
            raise ValueError(
                "external encoder results require the resolved vLLM ModelConfig"
            )
        if _is_multimodal_model(model_config):
            raise ValueError(
                "external linear embeddings currently require a text-only decoder"
            )
        if not getattr(engine_args, "enable_prompt_embeds", False):
            raise ValueError(
                "external linear embeddings require --enable-prompt-embeds"
            )

        self._hidden_size = _hidden_size(model_config)
        model_dtype = getattr(model_config, "dtype", None)
        self._dtype = model_dtype if isinstance(model_dtype, torch.dtype) else None
        self._importer_factory = importer_factory
        self._importer: TensorImporter | None = None

    async def load(
        self,
        encoder_result: Mapping[str, Any],
        token_ids: list[int],
    ) -> EmbedsPrompt:
        """Read one packed feature tensor and adapt it to vLLM's mixed mode."""

        parsed = ExternalEncoderResult.from_dict(encoder_result)
        if self._importer is None:
            self._importer = self._importer_factory()
        packed = await self._importer.import_tensor(parsed.features)
        if not isinstance(packed, torch.Tensor):
            raise TypeError(
                "external encoder NIXL import must produce a torch.Tensor; "
                f"got {type(packed).__name__}"
            )
        if packed.device.type != "cpu":
            raise ValueError(
                f"external encoder tensor is on {packed.device}; expected CPU"
            )
        if packed.dim() != 2 or packed.shape[1] != self._hidden_size:
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
        prompt_embeds, prompt_token_ids, prompt_is_token_ids = build_mixed_embeds(
            token_ids,
            rows,
            parsed.image_token_id,
        )
        return EmbedsPrompt(
            prompt_embeds=prompt_embeds,
            prompt_token_ids=prompt_token_ids,
            prompt_is_token_ids=prompt_is_token_ids,
        )
