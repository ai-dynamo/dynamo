# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One static encoder fan-out workflow, independent of remote placement."""

from dynamo.vllm.workflow.components import DynamoVllmStage, EncoderStage
from dynamo.workflow import Workflow
from examples.custom_backend.user_ensemble.stages import (
    DummyClassifier,
    EnsembleResponseStage,
)


def define_workflow() -> Workflow:
    workflow = Workflow("encoder-classifier-llm")
    request = workflow.input("request")
    encoder = workflow.stage("encoder", EncoderStage.contract, request=request)
    classifier = workflow.stage(
        "classifier",
        DummyClassifier.contract,
        encoder_features=encoder.encoder_features,
    )
    generator = workflow.stage(
        "generator",
        DynamoVllmStage.external_encoder_complete_contract,
        request=request,
        encoder_features=encoder.encoder_features,
        encoder_metadata=encoder.encoder_metadata,
    )
    response = workflow.stage(
        "response",
        EnsembleResponseStage.contract,
        completion=generator.completion,
        scores=classifier.scores,
    )
    workflow.output("chunk", response.chunk)
    return workflow
