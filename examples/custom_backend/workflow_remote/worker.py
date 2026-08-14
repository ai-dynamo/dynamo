# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.workflow import RemoteStageServer
from examples.custom_backend.workflow_remote.stages import STAGES


def endpoint_id(stage_id: str) -> str:
    return f"workflows.{stage_id}.generate"


@dynamo_worker()
async def worker(runtime: DistributedRuntime, stage_id: str) -> None:
    runner = STAGES[stage_id]()
    server = RemoteStageServer(stage_id, runner)
    await runtime.endpoint(endpoint_id(stage_id)).serve_endpoint(server.generate)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    args = parser.parse_args()
    asyncio.run(worker(args.stage))


if __name__ == "__main__":
    main()
