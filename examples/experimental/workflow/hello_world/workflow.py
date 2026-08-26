# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author the declarative Hello World workflow."""

from dynamo.experimental.workflow import Workflow
from examples.experimental.workflow.hello_world.stages import (
    HelloStage,
    MergeStage,
    WorldStage,
)


def define_workflow() -> Workflow:
    workflow = Workflow("hello-world")
    request = workflow.input("request")
    hello = workflow.stage("hello", HelloStage.contract, request=request)
    world = workflow.stage("world", WorldStage.contract, request=request)
    merge = workflow.stage(
        "merge",
        MergeStage.contract,
        hello=hello.text,
        world=world.text,
    )
    workflow.output("chunk", merge.chunk)
    return workflow
