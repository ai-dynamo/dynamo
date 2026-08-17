# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark provider for metadata-control and tensor-fanout classifiers."""

from __future__ import annotations

import os
from typing import Any

from dynamo.experimental.workflow.vllm import DynamoVllmStage, EncoderStage
from dynamo.experimental.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    GenerateEndpointBinding,
    InlineBinding,
    RemoteBinding,
    ValueSpec,
    Workflow,
    WorkflowOrchestrator,
    compile_workflow,
)
from examples.experimental.workflow.user_ensemble.remote.bindings import (
    CLASSIFIER_ENDPOINT,
    ENCODER_ENDPOINT,
    GENERATOR_ENDPOINT,
)
from examples.experimental.workflow.user_ensemble.stages import (
    DummyMetadataClassifier,
    EnsembleResponseStage,
)
from examples.experimental.workflow.user_ensemble.workflow import define_workflow

CLASSIFIER_INPUT_ENV = "DYN_BENCH_CLASSIFIER_INPUT"
CLASSIFIER_INPUTS = frozenset({"metadata", "tensor"})


def _metadata_classifier_workflow() -> Workflow:
    workflow = Workflow("encoder-metadata-classifier-vllm-qualification")
    request = workflow.input("request", ValueSpec(type="json"))
    encoder = workflow.stage("encoder", EncoderStage.contract, request=request)
    classifier = workflow.stage(
        "classifier",
        DummyMetadataClassifier.contract,
        encoder_metadata=encoder.encoder_metadata,
    )
    generator = workflow.stage(
        "generator",
        DynamoVllmStage.complete_contract,
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


def compile_benchmark_workflow(classifier_input: str) -> ExecutionPlan:
    if classifier_input not in CLASSIFIER_INPUTS:
        raise ValueError(
            f"{CLASSIFIER_INPUT_ENV} must be one of {sorted(CLASSIFIER_INPUTS)}, "
            f"got {classifier_input!r}"
        )
    workflow = (
        _metadata_classifier_workflow()
        if classifier_input == "metadata"
        else define_workflow()
    )
    return compile_workflow(
        workflow,
        DeploymentSpec(
            {
                "encoder": RemoteBinding(
                    ENCODER_ENDPOINT,
                    tensor_carrier="nixl",
                ),
                "classifier": RemoteBinding(
                    CLASSIFIER_ENDPOINT,
                    tensor_carrier=("nixl" if classifier_input == "tensor" else None),
                ),
                "generator": GenerateEndpointBinding(GENERATOR_ENDPOINT),
                "response": InlineBinding("response"),
            }
        ),
    )


async def provide_workflow(runtime: Any) -> WorkflowOrchestrator:
    """Bind the selected classifier qualification graph."""

    return await WorkflowOrchestrator.bind(
        compile_benchmark_workflow(os.environ.get(CLASSIFIER_INPUT_ENV, "metadata")),
        runtime=runtime,
        inline_runners={"response": EnsembleResponseStage()},
    )
