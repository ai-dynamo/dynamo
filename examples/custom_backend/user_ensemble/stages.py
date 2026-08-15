# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-owned classifier for the remote user ensemble."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch

from dynamo.llm.exceptions import InvalidArgument
from dynamo.workflow import StageContext, StageContract, ValueSpec

ENCODER_FEATURES = ValueSpec(type="tensor")


class DummyClassifier:
    """Replaceable classifier that consumes the encoder's shared tensor."""

    contract = StageContract(
        id="embedding-classifier",
        inputs={"encoder_features": ENCODER_FEATURES},
        outputs={"scores": ValueSpec(type="json")},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        features = inputs["encoder_features"]
        if not isinstance(features, torch.Tensor):
            raise InvalidArgument("classifier features must be a torch.Tensor")
        mean = float(features.float().mean().item())
        if not math.isfinite(mean):
            raise InvalidArgument("classifier features must contain finite values")
        positive = (math.tanh(mean) + 1.0) / 2.0
        return {
            "scores": {
                "positive-mean": positive,
                "negative-mean": 1.0 - positive,
            }
        }
