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
from enum import Enum

from dynamo.sdk import depends, service, dynamo_endpoint
from vllm.entrypoints.openai.protocol import CompletionRequest

from components.worker import VllmWorker, VllmPrefillWorker, VllmDecodeWorker

logger = logging.getLogger(__name__)


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor:
    worker = depends(VllmWorker)
    prefill_worker = depends(VllmPrefillWorker)
    decode_worker = depends(VllmDecodeWorker)

    async def send_prefill_request(self, req_data: dict):

        prefill_req_data = req_data.copy()
        prefill_req_data["do_remote_decode"] = True
        prefill_req_data["stream"] = False
        prefill_req_data["min_tokens"] = 1
        prefill_req_data["max_tokens"] = 1
        if "stream_options" in prefill_req_data:
            del prefill_req_data["stream_options"]

        logger.info(f"Prefill request: {prefill_req_data}")

        async for prefill_response in self.prefill_worker.completions(prefill_req_data):
            logger.info(f"Prefill response: {prefill_response}")
            return prefill_response

    async def send_decode_request(self, req_data: dict, prefill_response: dict):

        decode_req_data = req_data.copy()
        decode_req_data["stream"] = True
        decode_req_data["do_remote_prefill"] = True
        decode_req_data["remote_block_ids"] = prefill_response.get("remote_block_ids", [])
        decode_req_data['remote_engine_id'] = prefill_response.get("remote_engine_id", "")
        decode_req_data["remote_host"] = prefill_response.get("remote_host", "")
        decode_req_data["remote_port"] = prefill_response.get("remote_port", 0)

        logger.info(f"Decode request: {decode_req_data}")

        async for decode_response in self.decode_worker.completions(decode_req_data):
            logger.info(f"Decode response: {decode_response}")
            yield decode_response

    @dynamo_endpoint()
    async def completions(self, request: CompletionRequest):

        logger.info(f"Processor received completion request: {request}")

        req_data = request.model_dump()

        prefill_response = await self.send_prefill_request(req_data)

        logger.info(f"Prefill response: {prefill_response}")

        async for decode_response in self.send_decode_request(req_data, prefill_response):
            yield decode_response
    
