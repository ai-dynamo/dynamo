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
        async for response in self._generate_with_migration(remote_request):
            yield response

    async def _generate_with_migration(
        self,
        original_request: RequestType,
        sent_responses: list[ResponseType] = [],
        retry_count: int = 0,
    ):
        sent_responses_ = sent_responses[:]  # do not modify the default argument
        remote_request = self._new_request_with_sent_responses(
            original_request, sent_responses_, retry_count
        )
        restart = False
        try:
            remote_responses = await self._worker_client.round_robin(
                remote_request.model_dump_json()
            )
            try:
                async for response in remote_responses:
                    logger.info(f"Processor sending: {response.data()}")
                    sent_responses_.append(response.data())
                    yield response.data()
            except Exception as e:
                logger.warn(f"Processor error while streaming response: {e}")
                restart = True
        except Exception as e:
            logger.warn(f"Processor error while making the request: {e}")
            restart = True
        if restart:
            logger.info("Processor migrating the request...")
            async for response in self._generate_with_migration(
                original_request, sent_responses_, retry_count + 1
            ):
                yield response

    @staticmethod
    def _new_request_with_sent_responses(
        original_request: RequestType,
        sent_responses: list[ResponseType],
        retry_count: int,
    ) -> RequestType:
        # stop infinite retry
        if retry_count > 3:
            raise RuntimeError("Processor request migration retry limit reached")
        # instruct the worker to skip sent responses
        original_request_text_list = original_request.text.split(" ")
        new_request_text_list = original_request_text_list[len(sent_responses) :]
        new_request_text = " ".join(new_request_text_list)
        return RequestType(text=new_request_text)
