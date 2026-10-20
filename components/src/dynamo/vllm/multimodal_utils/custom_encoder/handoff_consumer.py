# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consume an external custom encoder handoff in aggregated vLLM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from vllm.inputs import EmbedsPrompt

from dynamo.llm.exceptions import InvalidArgument
from dynamo.vllm.multimodal_utils.custom_encoder.adapter.linear import (
    LinearEmbedsPromptBuilder,
)
from dynamo.vllm.multimodal_utils.custom_encoder.handoff import ExternalEncoderResult


class ExternalEncoderHandoffConsumer:
    """Decode an encoder handoff into the aggregated worker's prompt."""

    def __init__(self, model_config: Any, engine_args: Any) -> None:
        try:
            self._prompt_builder = LinearEmbedsPromptBuilder(
                model_config,
                engine_args,
            )
        except (TypeError, ValueError) as error:
            raise RuntimeError(str(error)) from error

    def prepare_prompt(
        self,
        encoder_result: Mapping[str, Any],
        token_ids: list[int],
    ) -> EmbedsPrompt:
        """Decode packed feature rows and adapt them to vLLM mixed mode."""

        try:
            parsed = ExternalEncoderResult.from_dict(encoder_result)
            artifacts = parsed.to_artifacts()
            return self._prompt_builder.prepare_prompt(
                token_ids,
                artifacts,
                image_token_id=parsed.image_token_id,
            )
        except (TypeError, ValueError) as error:
            raise InvalidArgument(
                f"invalid external encoder result: {error}"
            ) from error
