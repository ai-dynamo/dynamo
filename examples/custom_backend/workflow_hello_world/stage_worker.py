# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve one Hello World workflow stage through Dynamo discovery."""

import argparse
import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.workflow import RemoteStageServer
from examples.custom_backend.workflow_hello_world.stages import STAGES
from examples.custom_backend.workflow_hello_world.workflow import ENDPOINTS


@dynamo_worker()
async def worker(runtime: DistributedRuntime, stage_id: str) -> None:
    runner = STAGES[stage_id]()
    server = RemoteStageServer(stage_id, runner)
    await runtime.endpoint(ENDPOINTS[stage_id]).serve_endpoint(server.generate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    args = parser.parse_args()
    asyncio.run(worker(args.stage))


if __name__ == "__main__":
    main()
