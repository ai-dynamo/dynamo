# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.experimental.workflow import ValueSpec, Workflow
from examples.custom_backend.workflow_remote.stages import (
    KeywordClassifier,
    TextEncoder,
    TextGenerator,
)


def define_workflow() -> Workflow:
    workflow = Workflow("remote-text-fanout")
    text = workflow.input("text", ValueSpec(type="text"))
    encoder = workflow.stage("encoder", TextEncoder.contract, text=text)
    classifier = workflow.stage(
        "classifier", KeywordClassifier.contract, tokens=encoder.tokens
    )
    generator = workflow.stage(
        "generator", TextGenerator.contract, tokens=encoder.tokens
    )
    workflow.output("scores", classifier.scores)
    workflow.output("text", generator.text)
    return workflow
