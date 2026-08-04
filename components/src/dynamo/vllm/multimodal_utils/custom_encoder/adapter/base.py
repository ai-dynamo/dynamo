# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consumer-selected adapters for in-process custom vision encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

import torch
from vllm.inputs import EmbedsPrompt, TokensPrompt


class CustomEncoderAdapter(ABC):
    """Translate encoder artifacts for one resolved downstream decoder."""

    @abstractmethod
    def prepare_prompt(
        self,
        token_ids: list[int],
        encodings: Sequence[torch.Tensor],
    ) -> EmbedsPrompt | TokensPrompt:
        """Validate encoder artifacts and build the final vLLM prompt."""
