# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-specific stages shared by both ensemble implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from dynamo.experimental.workflow import StageContext, StageContract
from dynamo.llm.exceptions import InvalidArgument


class DummyClassifier:
    """Read encoder features and return deterministic demonstration scores."""

    contract = StageContract(
        id="dummy-encoder-classifier",
        inputs={"encoder_features"},
        outputs={"scores"},
    )

    async def run(
        self,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> Mapping[str, Any]:
        del context
        features = inputs["encoder_features"]
        if not isinstance(features, torch.Tensor) or features.dim() != 2:
            raise InvalidArgument("classifier requires a 2D encoder tensor")
        return {
            "scores": {
                "dummy-positive": 1.0,
                "feature-rows": float(features.shape[0]),
            }
        }


class EnsembleResponseStage:
    """Attach classifier scores to the completed token response."""

    contract = StageContract(
        id="ensemble-response",
        inputs={"completion", "scores"},
        outputs={"chunk"},
    )

    async def run(
        self,
        inputs: Mapping[str, Any],
        context: StageContext,
    ) -> Mapping[str, Any]:
        del context
        completion = inputs["completion"]
        scores = inputs["scores"]
        if not isinstance(completion, Mapping):
            raise TypeError("generator completion must be an object")
        if not isinstance(scores, Mapping):
            raise TypeError("classifier scores must be an object")

        chunk = dict(completion)
        engine_data_value = chunk.get("engine_data") or {}
        if not isinstance(engine_data_value, Mapping):
            raise TypeError("generator engine_data must be an object")
        engine_data = dict(engine_data_value)
        engine_data["user_ensemble"] = {"classifier_scores": dict(scores)}
        chunk["engine_data"] = engine_data
        return {"chunk": chunk}
