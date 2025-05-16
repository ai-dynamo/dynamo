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
from dynamo.planner.defaults import PlannerDefaults
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import async_on_start, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.core.protocol.interface import ComponentType
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)


class RequestType(BaseModel):
    text: str


@service(
    dynamo={
        "namespace": "dynamo",
        "component_type": ComponentType.PLANNER,
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
        config_instance = config.get("Planner", {})

        self.args = argparse.Namespace(
            namespace=self.namespace,
            environment=config_instance.get("environment", PlannerDefaults.environment),
            served_model_name=config_instance.get(
                "served-model-name", PlannerDefaults.served_model_name
            ),
            no_operation=config_instance.get(
                "no-operation", PlannerDefaults.no_operation
            ),
            log_dir=config_instance.get("log-dir", PlannerDefaults.log_dir),
            adjustment_interval=config_instance.get(
                "adjustment-interval", PlannerDefaults.adjustment_interval
            ),
            metric_pulling_interval=config_instance.get(
                "metric-pulling-interval", PlannerDefaults.metric_pulling_interval
            ),
            max_gpu_budget=config_instance.get(
                "max-gpu-budget", PlannerDefaults.max_gpu_budget
            ),
            min_endpoint=config_instance.get(
                "min-endpoint", PlannerDefaults.min_endpoint
            ),
            decode_kv_scale_up_threshold=config_instance.get(
                "decode-kv-scale-up-threshold",
                PlannerDefaults.decode_kv_scale_up_threshold,
            ),
            decode_kv_scale_down_threshold=config_instance.get(
                "decode-kv-scale-down-threshold",
                PlannerDefaults.decode_kv_scale_down_threshold,
            ),
            prefill_queue_scale_up_threshold=config_instance.get(
                "prefill-queue-scale-up-threshold",
                PlannerDefaults.prefill_queue_scale_up_threshold,
            ),
            prefill_queue_scale_down_threshold=config_instance.get(
                "prefill-queue-scale-down-threshold",
                PlannerDefaults.prefill_queue_scale_down_threshold,
            ),
            decode_engine_num_gpu=config_instance.get(
                "decode-engine-num-gpu", PlannerDefaults.decode_engine_num_gpu
            ),
            prefill_engine_num_gpu=config_instance.get(
                "prefill-engine-num-gpu", PlannerDefaults.prefill_engine_num_gpu
            ),
        )

    @async_on_start
    async def async_init(self):
        import asyncio

        await asyncio.sleep(30)
        logger.info("Calling start_planner")
        await start_planner(self.runtime, self.args)
        logger.info("Planner started")

    @dynamo_endpoint()
    async def generate(self, request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"
