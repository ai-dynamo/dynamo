# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-selected adapters for in-process custom vision encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, Sequence

from vllm.inputs import EmbedsPrompt, TokensPrompt

from dynamo.vllm.multimodal_utils.custom_encoder.backend import ArtifactT


class CustomEncoderAdapter(ABC, Generic[ArtifactT]):
    """Translate encoder artifacts for one resolved downstream decoder."""

    @abstractmethod
    def prepare_prompt(
        self,
        token_ids: list[int],
        artifacts: Sequence[ArtifactT],
        *,
        mm_processor_kwargs: dict[str, Any] | None = None,
    ) -> EmbedsPrompt | TokensPrompt:
        """Validate encoder artifacts and build the final vLLM prompt."""
