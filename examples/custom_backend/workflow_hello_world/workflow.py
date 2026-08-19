# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Author and compile the declarative Hello World workflow."""

from dynamo.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    ValueSpec,
    Workflow,
    compile_workflow,
)
from examples.custom_backend.workflow_hello_world.stages import (
    HelloStage,
    MergeStage,
    WorldStage,
)

ENDPOINTS = {
    "hello": "workflow-hello-world.hello.generate",
    "world": "workflow-hello-world.world.generate",
    "merge": "workflow-hello-world.merge.generate",
}


def define_workflow() -> Workflow:
    workflow = Workflow("hello-world")
    request = workflow.input("request", ValueSpec(type="json"))
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


def compile_remote_workflow() -> ExecutionPlan:
    return compile_workflow(
        define_workflow(),
        DeploymentSpec.remote(**ENDPOINTS),
    )
