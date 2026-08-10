# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Authored workflow for the user ensemble example."""

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
