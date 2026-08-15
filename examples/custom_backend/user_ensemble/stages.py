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


class EnsembleResponseStage:
    """Join the completed generation and classifier result for the frontend."""

    contract = StageContract(
        id="ensemble-response",
        inputs={
            "completion": ValueSpec(type="json"),
            "scores": ValueSpec(type="json"),
        },
        outputs={"chunk": ValueSpec(type="json")},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        completion = inputs["completion"]
        scores = inputs["scores"]
        if not isinstance(completion, Mapping):
            raise TypeError("ensemble response requires a completion object")
        if not isinstance(scores, Mapping):
            raise TypeError("ensemble response requires a scores object")

        chunk = dict(completion)
        engine_data = chunk.get("engine_data") or {}
        if not isinstance(engine_data, Mapping):
            raise TypeError("completion engine_data must be an object when present")
        merged_engine_data = dict(engine_data)
        ensemble = merged_engine_data.get("ensemble") or {}
        if not isinstance(ensemble, Mapping):
            raise TypeError(
                "completion ensemble metadata must be an object when present"
            )
        merged_ensemble = dict(ensemble)
        merged_ensemble["classifier_scores"] = dict(scores)
        merged_engine_data["ensemble"] = merged_ensemble
        chunk["engine_data"] = merged_engine_data
        return {"chunk": chunk}
