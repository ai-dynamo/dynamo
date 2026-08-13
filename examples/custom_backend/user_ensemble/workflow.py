# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authored workflow and shared result boundary for the user ensemble example."""

from collections.abc import Mapping
from typing import Any

from dynamo.vllm.decoder_stage import VllmDecoderStage
from dynamo.workflow import Workflow
from examples.custom_backend.user_ensemble.stages import DummyClassifier, EncoderStage


def define_workflow() -> Workflow:
    """Declare the logical pipeline independently from worker construction."""

    workflow = Workflow("encoder-classifier-llm")
    image_url = workflow.input("image_url", type="text")
    request = workflow.input(
        "request", type="object", class_id="dynamo.common.backend.GenerateRequest"
    )

    encoder = workflow.stage(
        "encoder", EncoderStage, image_url=image_url, request=request
    )
    classifier = workflow.stage(
        "classifier", DummyClassifier, artifacts=encoder.artifacts
    )
    generator = workflow.stage(
        "generator", VllmDecoderStage, request=request, prompt=encoder.prompt
    )

    workflow.output("scores", classifier.scores)
    workflow.output("chunk", generator.chunk)
    return workflow


def adapt_workflow_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Attach classifier output to the decoder chunk at either serving boundary."""

    chunk = result.get("chunk")
    scores = result.get("scores")
    if not isinstance(chunk, Mapping):
        raise TypeError("user ensemble workflow result requires a chunk object")
    if not isinstance(scores, Mapping):
        raise TypeError("user ensemble workflow result requires a scores object")

    adapted = dict(chunk)
    engine_data = adapted.get("engine_data")
    if engine_data is None:
        engine_data = {}
    if not isinstance(engine_data, Mapping):
        raise TypeError("decoder chunk engine_data must be an object when present")
    merged_engine_data = dict(engine_data)
    ensemble = merged_engine_data.get("ensemble")
    if ensemble is None:
        ensemble = {}
    if not isinstance(ensemble, Mapping):
        raise TypeError(
            "decoder chunk ensemble metadata must be an object when present"
        )
    merged_ensemble = dict(ensemble)
    merged_ensemble["classifier_scores"] = dict(scores)
    merged_engine_data["ensemble"] = merged_ensemble
    adapted["engine_data"] = merged_engine_data
    return adapted
