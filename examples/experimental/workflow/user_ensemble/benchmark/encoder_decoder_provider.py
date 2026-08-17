# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark provider for remote encoder -> stock vLLM -> inline response."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dynamo.experimental.workflow.vllm import DynamoVllmStage, EncoderStage
from dynamo.experimental.workflow import (
    DeploymentSpec,
    GenerateEndpointBinding,
    InlineBinding,
    RemoteBinding,
    StageContext,
    StageContract,
    ValueSpec,
    Workflow,
    WorkflowOrchestrator,
    compile_workflow,
)
from examples.experimental.workflow.user_ensemble.remote.bindings import (
    ENCODER_ENDPOINT,
    GENERATOR_ENDPOINT,
)


class _CompletionResponseStage:
    """Return the completed token chunk without benchmark-only decoration."""

    contract = StageContract(
        id="benchmark-completion-response",
        inputs={"completion": ValueSpec(type="json")},
        outputs={"chunk": ValueSpec(type="json")},
    )

    async def run(
        self, inputs: Mapping[str, Any], context: StageContext
    ) -> Mapping[str, Any]:
        context.raise_if_cancelled()
        completion = inputs["completion"]
        if not isinstance(completion, Mapping):
            raise TypeError("benchmark response requires a completion object")
        return {"chunk": dict(completion)}


def _compile_encoder_decoder_workflow():
    workflow = Workflow("encoder-vllm-write-qualification")
    request = workflow.input("request", ValueSpec(type="json"))
    encoder = workflow.stage("encoder", EncoderStage.contract, request=request)
    generator = workflow.stage(
        "generator",
        DynamoVllmStage.complete_contract,
        request=request,
        encoder_features=encoder.encoder_features,
        encoder_metadata=encoder.encoder_metadata,
    )
    response = workflow.stage(
        "response",
        _CompletionResponseStage.contract,
        completion=generator.completion,
    )
    workflow.output("chunk", response.chunk)
    return compile_workflow(
        workflow,
        DeploymentSpec(
            {
                "encoder": RemoteBinding(
                    ENCODER_ENDPOINT,
                    tensor_carrier="nixl",
                ),
                "generator": GenerateEndpointBinding(GENERATOR_ENDPOINT),
                "response": InlineBinding("response"),
            }
        ),
    )


async def provide_workflow(runtime: Any) -> WorkflowOrchestrator:
    """Bind the exact encoder -> aggregated vLLM qualification graph."""

    return await WorkflowOrchestrator.bind(
        _compile_encoder_decoder_workflow(),
        runtime=runtime,
        inline_runners={"response": _CompletionResponseStage()},
    )
