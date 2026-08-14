# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.workflow import DeploymentSpec, WorkflowExecutor, compile_workflow
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
    executor = await WorkflowExecutor.bind(plan, runtime=runtime)
    result = await executor.run(
        {"text": "Dynamo workflow runs across processes"},
        attempt_id="remote-cpu-e2e",
    )
    expected = {
        "scores": {"workflow": 0.2, "other": 0.8},
        "text": "processes across runs workflow dynamo",
    }
    if result != expected:
        raise RuntimeError(f"unexpected remote workflow result: {result!r}")
    print(result)


if __name__ == "__main__":
    asyncio.run(client())
