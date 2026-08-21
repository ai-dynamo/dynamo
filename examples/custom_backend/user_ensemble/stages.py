# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Application-owned classifier and response stages for the user ensemble."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynamo.workflow import StageContext, StageContract


class DummyClassifier:
    """Replaceable request classifier used to demonstrate remote fan-out."""

    contract = StageContract(
        id="request-classifier",
        inputs={"request"},
        outputs={"scores"},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        request = inputs["request"]
        if not isinstance(request, Mapping):
            raise TypeError("classifier request must be an object")
        return {
            "scores": {
                "relevant": 0.75,
                "not_relevant": 0.25,
            }
        }


class EnsembleResponseStage:
    """Join the completed generation and classifier result for the frontend."""

    contract = StageContract(
        id="ensemble-response",
        inputs={"completion", "scores"},
        outputs={"chunk"},
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
