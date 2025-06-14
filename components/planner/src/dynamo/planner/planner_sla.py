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
import asyncio
import logging

from pydantic import BaseModel

from dynamo.planner.defaults import SLAPlannerDefaults
from dynamo.planner.utils.planner_core import start_sla_planner
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.sdk import async_on_start, dynamo_context, endpoint, service
from dynamo.sdk.core.protocol.interface import ComponentType
from dynamo.sdk.lib.config import ServiceConfig
from dynamo.sdk.lib.image import DYNAMO_IMAGE

logger = logging.getLogger(__name__)

# start planner 30 seconds after the other components to make sure planner can see them
# TODO: remove this delay
INIT_PLANNER_START_DELAY = 30


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
            environment=config_instance.get(
                "environment", SLAPlannerDefaults.environment
            ),
            no_operation=config_instance.get(
                "no-operation", SLAPlannerDefaults.no_operation
            ),
            log_dir=config_instance.get("log-dir", SLAPlannerDefaults.log_dir),
            adjustment_interval=config_instance.get(
                "adjustment-interval", SLAPlannerDefaults.adjustment_interval
            ),
            max_gpu_budget=config_instance.get(
                "max-gpu-budget", SLAPlannerDefaults.max_gpu_budget
            ),
            min_endpoint=config_instance.get(
                "min-endpoint", SLAPlannerDefaults.min_endpoint
            ),
            decode_engine_num_gpu=config_instance.get(
                "decode-engine-num-gpu", SLAPlannerDefaults.decode_engine_num_gpu
            ),
            prefill_engine_num_gpu=config_instance.get(
                "prefill-engine-num-gpu", SLAPlannerDefaults.prefill_engine_num_gpu
            ),
            prometheus_endpoint=config_instance.get(
                "prometheus-endpoint", SLAPlannerDefaults.prometheus_endpoint
            ),
            profile_results_dir=config_instance.get(
                "profile-results-dir", SLAPlannerDefaults.profile_results_dir
            ),
            isl=config_instance.get("isl", SLAPlannerDefaults.isl),
            osl=config_instance.get("osl", SLAPlannerDefaults.osl),
            ttft=config_instance.get("ttft", SLAPlannerDefaults.ttft),
            itl=config_instance.get("itl", SLAPlannerDefaults.itl),
            load_predictor=config_instance.get(
                "load-predictor", SLAPlannerDefaults.load_predictor
            ),
            load_prediction_window_size=config_instance.get(
                "load-prediction-window-size",
                SLAPlannerDefaults.load_prediction_window_size,
            ),
        )

    @async_on_start
    async def async_init(self):
        await asyncio.sleep(INIT_PLANNER_START_DELAY)
        logger.info("Calling start_planner")
        await start_sla_planner(self.runtime, self.args)
        logger.info("Planner started")

    @endpoint()
    async def generate(self, request: RequestType):
        """Dummy endpoint to satisfy that each component has an endpoint"""
        yield "mock endpoint"
