# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the Hello World workflow as one discovered model worker."""

import asyncio
import os

from dynamo.experimental.workflow import WorkflowEndpointHandler, WorkflowOrchestrator
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime, dynamo_worker
from examples.experimental.workflow.hello_world.workflow import compile_remote_workflow

ORCHESTRATOR_ENDPOINT = "workflow-hello-world.orchestrator.generate"
MODEL_NAME = "hello-world"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    orchestrator = await WorkflowOrchestrator.bind(
        compile_remote_workflow(),
        runtime=runtime,
    )
    endpoint = runtime.endpoint(ORCHESTRATOR_ENDPOINT)
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        os.environ.get("DYN_MODEL", DEFAULT_MODEL),
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
        ignore_weights=True,
    )
    await endpoint.serve_endpoint(WorkflowEndpointHandler(orchestrator).generate)


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
