# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark orchestrator for metadata-control and tensor-fanout classifiers."""

from __future__ import annotations

import asyncio
import os

from dynamo.experimental.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    GenerateEndpointBinding,
    InlineBinding,
    RemoteBinding,
    Workflow,
    compile_workflow,
)
from dynamo.experimental.workflow.vllm import DynamoVllmStage, EncoderStage
from dynamo.runtime import DistributedRuntime, dynamo_worker
from examples.experimental.workflow.user_ensemble.remote.bindings import (
    CLASSIFIER_ENDPOINT,
    ENCODER_ENDPOINT,
    GENERATOR_ENDPOINT,
)
from examples.experimental.workflow.user_ensemble.remote.orchestrator_worker import (
    serve_orchestrator,
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
    request = workflow.input("request")
    encoder = workflow.stage("encoder", EncoderStage.contract, request=request)
    classifier = workflow.stage(
        "classifier",
        DummyMetadataClassifier.contract,
        encoder_metadata=encoder.encoder_metadata,
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
                "generator": GenerateEndpointBinding(
                    GENERATOR_ENDPOINT,
                    tensor_carrier="nixl",
                ),
                "response": InlineBinding("response"),
            }
        ),
    )


@dynamo_worker()
async def benchmark_orchestrator_worker(runtime: DistributedRuntime) -> None:
    """Serve the selected classifier qualification graph."""

    await serve_orchestrator(
        runtime,
        compile_benchmark_workflow(os.environ.get(CLASSIFIER_INPUT_ENV, "metadata")),
    )


def main() -> None:
    asyncio.run(benchmark_orchestrator_worker())


if __name__ == "__main__":
    main()
