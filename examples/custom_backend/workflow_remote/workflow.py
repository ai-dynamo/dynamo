# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.workflow import Workflow
from examples.custom_backend.workflow_remote.stages import (
    KeywordClassifier,
    TextEncoder,
    TextGenerator,
)


def define_workflow() -> Workflow:
    workflow = Workflow("remote-text-fanout")
    text = workflow.input("text", type="text")
    encoder = workflow.stage("encoder", TextEncoder, text=text)
    classifier = workflow.stage("classifier", KeywordClassifier, tokens=encoder.tokens)
    generator = workflow.stage("generator", TextGenerator, tokens=encoder.tokens)
    workflow.output("scores", classifier.scores)
    workflow.output("text", generator.text)
    return workflow
