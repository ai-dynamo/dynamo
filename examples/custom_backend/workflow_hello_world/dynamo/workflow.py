# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Author and bind the declarative workflow."""

from typing import Any

from dynamo.workflow import (
    DeploymentSpec,
    ExecutionPlan,
    ValueSpec,
    Workflow,
    WorkflowOrchestrator,
    compile_workflow,
)
from examples.custom_backend.workflow_hello_world.dynamo.stages import (
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


async def provide_workflow(runtime: Any) -> WorkflowOrchestrator:
    return await WorkflowOrchestrator.bind(
        compile_remote_workflow(),
        runtime=runtime,
    )
