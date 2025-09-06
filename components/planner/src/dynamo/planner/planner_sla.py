# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import logging

from pydantic import BaseModel

from dynamo.planner.utils.planner_argparse import create_sla_planner_parser
from dynamo.planner.utils.planner_core import start_sla_planner
from dynamo.runtime import DistributedRuntime, dynamo_worker

logger = logging.getLogger(__name__)

# start planner 30 seconds after the other components to make sure planner can see them
# TODO: remove this delay
INIT_PLANNER_START_DELAY = 30


class RequestType(BaseModel):
    text: str


@dynamo_worker(static=False)
async def init_planner(runtime: DistributedRuntime, args):
    await asyncio.sleep(INIT_PLANNER_START_DELAY)

    await start_sla_planner(runtime, args)

    component = runtime.namespace(args.namespace).component("Planner")
    await component.create_service()

    async def generate(request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"

    generate_endpoint = component.endpoint("generate")
    await generate_endpoint.serve_endpoint(generate)


if __name__ == "__main__":
    parser = create_sla_planner_parser()
    args = parser.parse_args()
    asyncio.run(init_planner(args))
