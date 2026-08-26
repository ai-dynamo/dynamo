# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author the placement-neutral user ensemble."""

from dynamo.experimental.workflow import Workflow
from dynamo.experimental.workflow.vllm import (
    DynamoVllmStage,
    EncoderStage,
    ExternalEncoderRequestStage,
)
from examples.experimental.workflow.user_ensemble.common.stages import (
    DummyClassifier,
    EnsembleResponseStage,
)


def define_workflow() -> Workflow:
    """Declare application dataflow independently from stage placement."""

    workflow = Workflow("user-ensemble")
    request = workflow.input("request")
    encoder = workflow.stage("encoder", EncoderStage.contract, request=request)
    classifier = workflow.stage(
        "classifier",
        DummyClassifier.contract,
        encoder_features=encoder.encoder_features,
    )
    prepared = workflow.stage(
        "request_adapter",
        ExternalEncoderRequestStage.contract,
        request=request,
        encoder_features=encoder.encoder_features,
        encoder_metadata=encoder.encoder_metadata,
    )
    generator = workflow.stage(
        "generator",
        DynamoVllmStage.request_complete_contract,
        request=prepared.request,
    )
    response = workflow.stage(
        "response",
        EnsembleResponseStage.contract,
        completion=generator.completion,
        scores=classifier.scores,
    )
    workflow.output("chunk", response.chunk)
    return workflow
