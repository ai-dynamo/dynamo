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

import asyncio
import logging

from pydantic import BaseModel

from dynamo.planner.planner_args import PlannerConfig
from dynamo.planner.utils.agg_planner import AggPlanner
from dynamo.planner.utils.decode_planner import DecodePlanner
from dynamo.planner.utils.disagg_planner import DisaggPlanner
from dynamo.planner.utils.planner_argparse import create_sla_planner_parser
from dynamo.planner.utils.prefill_planner import PrefillPlanner
from dynamo.runtime import DistributedRuntime, dynamo_worker

logger = logging.getLogger(__name__)

# start planner 30 seconds after the other components to make sure planner can see them
# TODO: remove this delay
INIT_PLANNER_START_DELAY = 30


class RequestType(BaseModel):
    text: str


async def start_sla_planner(runtime: DistributedRuntime, config: PlannerConfig):
    mode = config.mode
    if mode == "disagg":
        planner = DisaggPlanner(runtime, config)
    elif mode == "prefill":
        planner = PrefillPlanner(runtime, config)
    elif mode == "decode":
        planner = DecodePlanner(runtime, config)
    elif mode == "agg":
        planner = AggPlanner(runtime, config)
    else:
        raise ValueError(f"Invalid planner mode: {mode}")
    await planner._async_init()
    await planner.run()


@dynamo_worker()
async def init_planner(runtime: DistributedRuntime, config: PlannerConfig):
    await asyncio.sleep(INIT_PLANNER_START_DELAY)

    await start_sla_planner(runtime, config)

    component = runtime.namespace(config.namespace).component("Planner")

    async def generate(request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"

    generate_endpoint = component.endpoint("generate")
    await generate_endpoint.serve_endpoint(generate)


if __name__ == "__main__":
    parser = create_sla_planner_parser()
    args = parser.parse_args()
    config = PlannerConfig.from_cli_args(args)
    config.validate()
    asyncio.run(init_planner(config))
