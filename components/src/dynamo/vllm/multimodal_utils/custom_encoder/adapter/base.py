# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-selected adapters for in-process custom vision encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence

from vllm.inputs import EmbedsPrompt, TokensPrompt

from dynamo.vllm.multimodal_utils.custom_encoder.backend import ArtifactT


class CustomEncoderAdapter(ABC, Generic[ArtifactT]):
    """Translate encoder artifacts for one resolved downstream decoder."""

    @abstractmethod
    def prepare_prompt(
        self,
        token_ids: list[int],
        artifacts: Sequence[ArtifactT],
    ) -> EmbedsPrompt | TokensPrompt:
        """Validate encoder artifacts and build the final vLLM prompt.

        Args:
            token_ids: Tokenized prompt containing the image placeholders.
            artifacts: Opaque values returned by the encoder backend, in image
                order. Each adapter defines and validates its concrete artifact
                contract.
        """
