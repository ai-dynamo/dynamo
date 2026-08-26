# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author and compile the mixed-placement user ensemble."""

from dynamo.experimental.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    GenerateEndpointBinding,
    InlineBinding,
    Workflow,
    compile_workflow,
)
from dynamo.experimental.workflow.vllm import (
    DynamoVllmStage,
    EncoderStage,
    ExternalEncoderRequestStage,
)
from examples.experimental.workflow.user_ensemble.common.config import (
    GENERATOR_ENDPOINT,
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


def compile_user_ensemble() -> ExecutionPlan:
    """Bind four inline stages and one discovered stock Generate endpoint."""

    return compile_workflow(
        define_workflow(),
        DeploymentSpec(
            bindings={
                "encoder": InlineBinding("encoder"),
                "classifier": InlineBinding("classifier"),
                "request_adapter": InlineBinding("request_adapter"),
                "generator": GenerateEndpointBinding(GENERATOR_ENDPOINT),
                "response": InlineBinding("response"),
            }
        ),
    )
