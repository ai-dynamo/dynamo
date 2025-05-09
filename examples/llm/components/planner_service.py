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

import argparse
import logging

from pydantic import BaseModel

from components.planner import start_planner  # type: ignore[attr-defined]
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)


class RequestType(BaseModel):
    text: str


@service(
    dynamo={
        "namespace": "dynamo",
        "component_type": "planner",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
    image=DYNAMO_IMAGE,
)
class Planner:
    def __init__(self):
        configure_dynamo_logging(service_name="Planner")
        logger.info("Starting planner")
        self.runtime = dynamo_context["runtime"]

        config = ServiceConfig.get_instance()

        # Get namespace directly from dynamo_context as it contains the active namespace
        self.namespace = dynamo_context["namespace"]
        self.environment = config.get("Planner", {}).get("environment", "local")
        self.no_operation = config.get("Planner", {}).get("no-operation", True)

        # Create args with all parameters from planner.py, using defaults except for namespace and environment
        self.args = argparse.Namespace(
            namespace=self.namespace,
            environment=self.environment,
            served_model_name="vllm",
            no_operation=self.no_operation,
            log_dir=None,
            adjustment_interval=10,
            metric_pulling_interval=1,
            max_gpu_budget=8,
            min_endpoint=1,
            decode_kv_scale_up_threshold=0.9,
            decode_kv_scale_down_threshold=0.5,
            prefill_queue_scale_up_threshold=5,
            prefill_queue_scale_down_threshold=0.2,
            decode_engine_num_gpu=1,
            prefill_engine_num_gpu=1,
        )

    @async_on_start
    async def async_init(self):
        import asyncio

        await asyncio.sleep(60)
        logger.info("Calling start_planner")
        await start_planner(self.runtime, self.args)
        logger.info("Planner started")

    @dynamo_endpoint()
    async def generate(self, request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"
