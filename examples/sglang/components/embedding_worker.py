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

"""
Using SGLang and Dynamo to serve embedding models!
"""

import logging

import sglang as sgl
from utils.protocol import EmbeddingRequest
from utils.sglang import parse_sglang_args

from dynamo.llm import ModelType, register_llm
from dynamo.sdk import async_on_start, dynamo_context, endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1},
    workers=1,
)
class SGLangEmbeddingWorker:
    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        self.engine = sgl.Engine(server_args=self.engine_args)

        logger.info("SGLangEmbeddingWorker initialized")

    @async_on_start
    async def async_init(self):
        runtime = dynamo_context["runtime"]
        logger.info("Registering LLM for discovery")
        comp_ns, comp_name = SGLangEmbeddingWorker.dynamo_address()  # type: ignore
        endpoint = runtime.namespace(comp_ns).component(comp_name).endpoint("generate")
        await register_llm(
            ModelType.Embedding,
            endpoint,
            self.engine_args.model_path,
            self.engine_args.served_model_name,
        )

    @endpoint()
    async def generate(self, request: EmbeddingRequest):
        # SGL has an open bug in which it cannot take list that contains a single string
        # https://github.com/sgl-project/sglang/issues/6568
        # We internally convert this to a single string

        if isinstance(request.input, str):
            prompt = request.input
        elif isinstance(request.input, list):
            if len(request.input) == 1:
                prompt = request.input[0]
            else:
                prompt = [i for i in request.input]
        else:
            raise ValueError(f"Invalid input type: {type(request.input)}")

        g = await self.engine.async_encode(
            prompt=prompt,
        )

        response = self._transform_response(g, request.model)
        yield response

    def _transform_response(self, ret, model_name):
        """Transform SGLang response to OpenAI embedding format"""
        if not isinstance(ret, list):
            ret = [ret]

        embedding_objects = []
        prompt_tokens = 0

        for idx, ret_item in enumerate(ret):
            embedding_objects.append(
                {
                    "object": "embedding",
                    "embedding": ret_item["embedding"],
                    "index": idx,
                }
            )
            prompt_tokens += ret_item["meta_info"]["prompt_tokens"]

        return {
            "object": "list",
            "data": embedding_objects,
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }
