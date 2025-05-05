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
import uuid
from enum import Enum
from typing import Union

from components.worker import SGLangWorker
from sglang.srt.openai_api.protocol import ChatCompletionRequest, CompletionRequest
from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.protocol import SGLangGenerateRequest
from utils.sglang import parse_sglang_args

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

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
class Processor(ProcessMixIn):
    """
    SGLang pre and post processing
    """

    sglang_worker = depends(SGLangWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        self.chat_processor = ChatProcessor()
        self.completions_processor = CompletionsProcessor()
        # init tokenizer manager
        super().__init__()

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        comp_ns, comp_name = SGLangWorker.dynamo_address()  # type: ignore
        self.worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )
        logger.warning("processor initialized")

    async def _generate(
        self,
        raw_request: Union[CompletionRequest, ChatCompletionRequest],
        request_type: RequestType,
    ):
        request_id = str(uuid.uuid4())
        logger.debug(f"Got raw request: {raw_request}")
        request, sampling_params = await self._parse_raw_request(raw_request)

        logger.warning(sampling_params)
        logger.warning(request)

        request_obj = SGLangGenerateRequest(
            request_id=request_id,
            input_ids=request.input_ids,
            sampling_params=sampling_params,
        ).model_dump_json()

        logger.warning(request_obj)

        # Get the async generator from worker
        output_generator = await self.worker_client.generate(request_obj)

        # Properly yield each result from the generator
        async for result in output_generator:
            yield result

    @dynamo_endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest):
        # Get a stream generator from _generate
        sglang_generator = self._generate(raw_request, RequestType.CHAT)

        # Process the stream using generate_stream_response from ChatProcessor
        async for response in self.chat_processor.generate_stream_response(
            raw_request, sglang_generator, self.tokenizer_manager
        ):
            yield response

    # @dynamo_endpoint()
    # async def completions(self, raw_request: CompletionRequest):
    #     async for response in self._generate(raw_request, RequestType.COMPLETION):
    #         yield response
