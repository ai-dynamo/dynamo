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

import logging

from pydantic import BaseModel

from dynamo.sdk import async_on_start, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)


class RequestType(BaseModel):
    text: str


class ResponseType(BaseModel):
    text: str


@service(
    dynamo={"namespace": "FaultTolerance"},
    workers=1,
)
class Processor:
    # TODO: Why depends is needed?
    # worker = depends(Worker)

    def __init__(self) -> None:
        logger.info("Starting Processor...")
        self._runtime = dynamo_context["runtime"]

    @async_on_start
    async def async_init(self):
        logger.info("Starting Processor async...")
        self._worker_client = (
            await self._runtime.namespace("FaultTolerance")
            .component("Worker")
            .endpoint("generate")
            .client()
        )

    @endpoint(name="generate")
    async def generate(self, request: RequestType):
        logger.info(f"Processor received: {request.text}")

        remote_request = RequestType(
            text=request.text,
        )

        try:
            remote_responses = await self._worker_client.round_robin(
                remote_request.model_dump_json()
            )
            try:
                async for response in remote_responses:
                    # logger.info(f"Processor sending: {response.data()}")
                    yield response.data()
            except Exception as e:
                logger.error(f"Processor response stream error: {e}")
        except Exception as e:
            logger.error(f"Processor error: {e}")
