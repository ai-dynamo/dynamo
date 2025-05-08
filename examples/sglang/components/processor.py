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

import json
import logging
import random
import uuid
from enum import Enum
from typing import Union

from components.worker import SGLangDecodeWorker, SGLangWorker
from sglang.srt.openai_api.protocol import ChatCompletionRequest, CompletionRequest
from utils.chat_processor import ChatProcessor, CompletionsProcessor, ProcessMixIn
from utils.protocol import SGLangGenerateRequest
from utils.sglang import parse_sglang_args

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

logger = logging.getLogger(__name__)


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
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
        self.enable_disagg = self._check_disagg_mode()
        logger.info(f"Processor initialized with enable_disagg: {self.enable_disagg}")
        self.chat_processor = ChatProcessor()
        self.completions_processor = CompletionsProcessor()
        self._cached_prefill_urls = {}
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

        # disag case
        if self.enable_disagg:
            self.worker_get_url_client = (
                await runtime.namespace(comp_ns)
                .component(comp_name)
                .endpoint("get_url")
                .client()
            )
            comp_ns, comp_name = SGLangDecodeWorker.dynamo_address()  # type: ignore
            self.decode_client = (
                await runtime.namespace(comp_ns)
                .component(comp_name)
                .endpoint("generate")
                .client()
            )
            logger.info("Processor initialized")

    async def _generate(
        self,
        raw_request: Union[CompletionRequest, ChatCompletionRequest],
        request_type: RequestType,
    ):
        request_id = str(uuid.uuid4())
        logger.debug(f"Got raw request: {raw_request}")
        request, sampling_params = await self._parse_raw_request(raw_request)

        request_obj = SGLangGenerateRequest(
            request_id=request_id,
            input_ids=request.input_ids,
            sampling_params=sampling_params,
        )

        prefill_resp = None
        if self.enable_disagg:
            prefill_id, decode_id = self._select_random_disagg_endpoint_pair()

            logger.info(f"Using prefill id: {prefill_id} and decode id: {decode_id}")

            if prefill_id not in self._cached_prefill_urls:
                logger.info(f"Getting prefill url for {prefill_id}")
                async for response in await self.worker_get_url_client.direct(
                    {}, prefill_id
                ):
                    bootstrap_info = json.loads(response.data())
                    logger.info(f"Caching bootstrap info: {bootstrap_info}")
                    self._cached_prefill_urls[prefill_id] = bootstrap_info

            bootstrap_info = self._cached_prefill_urls[prefill_id]
            logger.info(f"Using bootstrap info: {bootstrap_info}")

            hostname = bootstrap_info.get("host")
            port = bootstrap_info.get("port")
            logger.info(f"Hostname: {hostname}")

            # update the request object with this data
            request_obj.bootstrap_host = hostname
            request_obj.bootstrap_port = port
            request_obj.bootstrap_room = self._generate_bootstrap_room()

            prefill_resp = self.worker_client.direct(
                request_obj.model_dump_json(), prefill_id
            )
            output_generator = await self.decode_client.direct(
                request_obj.model_dump_json(), decode_id
            )
        else:
            output_generator = await self.worker_client.generate(
                request_obj.model_dump_json()
            )

        async for result in output_generator:
            yield result

        if prefill_resp:
            await prefill_resp

    def _select_random_disagg_endpoint_pair(self):
        prefill_ids = self.worker_client.endpoint_ids()
        decode_ids = self.decode_client.endpoint_ids()
        return random.choice(prefill_ids), random.choice(decode_ids)

    def _generate_bootstrap_room(self):
        return random.randint(0, 2**63 - 1)

    def _check_disagg_mode(self):
        # TODO: hacking this in tonight cause I'm tired but instead should add functionality to config.py to have an
        # ignore prefix so that we can include a _enable_disagg in the config YAML but it doesn't get
        # passed to the sglang args
        config = ServiceConfig.get_instance()
        is_disagg = config.get("SGLangWorker", {}).get("disaggregation-mode", None)
        if is_disagg:
            return True
        else:
            return False

    @dynamo_endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest):
        sglang_generator = self._generate(raw_request, RequestType.CHAT)

        async for response in self.chat_processor.generate_stream_response(
            raw_request, sglang_generator, self.tokenizer_manager
        ):
            yield response

    @dynamo_endpoint(name="completions")
    async def completions(self, raw_request: CompletionRequest):
        sglang_generator = self._generate(raw_request, RequestType.COMPLETION)

        async for response in self.completions_processor.generate_stream_response(
            raw_request, sglang_generator, self.tokenizer_manager
        ):
            yield response
