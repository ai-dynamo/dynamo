# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static request fan-out workflow, independent of physical placement."""

from dynamo.vllm.workflow.components import DynamoVllmStage
from dynamo.workflow import Workflow
from examples.custom_backend.user_ensemble.stages import (
    DummyClassifier,
    EnsembleResponseStage,
)


def define_workflow() -> Workflow:
    workflow = Workflow("integrated-encoder-ensemble")
    request = workflow.input("request")
    classifier = workflow.stage(
        "classifier",
        DummyClassifier.contract,
        request=request,
    )
    generator = workflow.stage(
        "generator",
        DynamoVllmStage.request_complete_contract,
        request=request,
    )
    response = workflow.stage(
        "response",
        EnsembleResponseStage.contract,
        completion=generator.completion,
        scores=classifier.scores,
    )
    workflow.output("chunk", response.chunk)
    return workflow
