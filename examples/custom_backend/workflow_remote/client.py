# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.experimental.workflow import DeploymentSpec, WorkflowExecutor, compile_workflow
from examples.custom_backend.workflow_remote.workflow import define_workflow


@dynamo_worker()
async def client(runtime: DistributedRuntime) -> None:
    plan = compile_workflow(
        define_workflow(),
        DeploymentSpec.remote(
            encoder="workflows.encoder.generate",
            classifier="workflows.classifier.generate",
            generator="workflows.generator.generate",
        ),
    )
    executor = await WorkflowExecutor.from_runtime(plan, runtime)
    result = await executor.run(
        {"text": "Dynamo workflow runs across processes"},
        attempt_id="remote-example-1",
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(client())
