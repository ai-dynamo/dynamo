# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

import torch

from dynamo.logits_processing import BaseLogitsProcessor


class TemperatureProcessor(BaseLogitsProcessor):
    """
    Example logits processor that applies temperature scaling.

    This is a simple demonstration of how to implement a logits processor
    that can be used with any Dynamo backend.
    """

    def __init__(self, temperature: float = 1.0):
        """
        Args:
            temperature: Scaling factor. Higher values make distribution more uniform,
                        lower values make it more peaked. Must be positive.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self.temperature = temperature

    def __call__(self, input_ids: Sequence[int], logits: torch.Tensor):
        """
        Apply temperature scaling to logits.

        Args:
            input_ids: Token IDs generated so far (unused in this simple example)
            logits: Raw logits tensor from model

        The processor is expected to modify the logits in-place.
        """
        if self.temperature == 1.0:
            return
        logits.div_(self.temperature)
