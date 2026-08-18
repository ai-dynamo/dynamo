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

"""Serve one workflow stage through Dynamo discovery."""

import argparse
import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker
from dynamo.workflow import RemoteStageServer
from examples.custom_backend.workflow_hello_world.dynamo.stages import STAGES
from examples.custom_backend.workflow_hello_world.dynamo.workflow import ENDPOINTS


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
