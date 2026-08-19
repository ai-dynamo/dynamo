# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve the declarative Hello World workflow as one discovered worker."""

import asyncio
import os

from dynamo.llm import (
    EngineType,
    EntrypointArgs,
    PythonAsyncEngine,
    make_engine,
    run_input,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.workflow import WorkflowOrchestrator, WorkflowTokenEngine
from examples.custom_backend.workflow_hello_world.dynamo.workflow import (
    compile_remote_workflow,
)

ORCHESTRATOR_ENDPOINT = "dyn://workflow-hello-world.orchestrator.generate"
MODEL_NAME = "hello-world"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    orchestrator = await WorkflowOrchestrator.bind(
        compile_remote_workflow(),
        runtime=runtime,
    )
    loop = asyncio.get_running_loop()
    token_engine = PythonAsyncEngine(
        WorkflowTokenEngine(orchestrator).generate,
        loop,
    )
    engine = await make_engine(
        runtime,
        EntrypointArgs(
            EngineType.InProcessTokens,
            model_path=os.environ.get("DYN_MODEL", DEFAULT_MODEL),
            model_name=MODEL_NAME,
            endpoint_id=ORCHESTRATOR_ENDPOINT,
            in_process_token_engine=token_engine,
        ),
    )
    await run_input(runtime, ORCHESTRATOR_ENDPOINT, engine)


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
