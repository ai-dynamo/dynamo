# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One static encoder fan-out workflow, independent of remote placement."""

from collections.abc import Mapping
from typing import Any

from dynamo.vllm.workflow.components import DynamoVllmStage, EncoderStage
from dynamo.workflow import ValueSpec, Workflow
from examples.custom_backend.user_ensemble.stages import DummyClassifier


def define_workflow() -> Workflow:
    workflow = Workflow("encoder-classifier-llm")
    request = workflow.input("request", ValueSpec(type="json"))
    encoder = workflow.stage("encoder", EncoderStage.contract, request=request)
    classifier = workflow.stage(
        "classifier",
        DummyClassifier.contract,
        encoder_features=encoder.encoder_features,
    )
    generator = workflow.stage(
        "generator",
        DynamoVllmStage.contract,
        request=request,
        encoder_features=encoder.encoder_features,
        encoder_metadata=encoder.encoder_metadata,
    )
    workflow.output("scores", classifier.scores)
    workflow.output("chunk", generator.chunk)
    return workflow


def adapt_workflow_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Attach classifier scores to the stock decoder's terminal chunk."""

    chunk = result.get("chunk")
    scores = result.get("scores")
    if not isinstance(chunk, Mapping):
        raise TypeError("user ensemble workflow result requires a chunk object")
    if not isinstance(scores, Mapping):
        raise TypeError("user ensemble workflow result requires a scores object")

    adapted = dict(chunk)
    engine_data = adapted.get("engine_data") or {}
    if not isinstance(engine_data, Mapping):
        raise TypeError("decoder chunk engine_data must be an object when present")
    merged_engine_data = dict(engine_data)
    ensemble = merged_engine_data.get("ensemble") or {}
    if not isinstance(ensemble, Mapping):
        raise TypeError("decoder ensemble metadata must be an object when present")
    merged_ensemble = dict(ensemble)
    merged_ensemble["classifier_scores"] = dict(scores)
    merged_engine_data["ensemble"] = merged_ensemble
    adapted["engine_data"] = merged_engine_data
    return adapted
