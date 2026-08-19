# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a user-orchestrated Hello World workflow as one Dynamo worker."""

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
from examples.custom_backend.workflow_hello_world.dynamo_bespoke.engine import (
    AggregatedHelloWorldEngine,
)

ENDPOINT = "dyn://my.endpoint.generate"
MODEL_NAME = "hello-world"
DEFAULT_MODEL = "Qwen/Qwen3-0.6B"


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    loop = asyncio.get_running_loop()
    token_engine = PythonAsyncEngine(AggregatedHelloWorldEngine().generate, loop)
    engine = await make_engine(
        runtime,
        EntrypointArgs(
            EngineType.InProcessTokens,
            model_path=os.environ.get("DYN_MODEL", DEFAULT_MODEL),
            model_name=MODEL_NAME,
            endpoint_id=ENDPOINT,
            in_process_token_engine=token_engine,
        ),
    )
    await run_input(runtime, ENDPOINT, engine)


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
