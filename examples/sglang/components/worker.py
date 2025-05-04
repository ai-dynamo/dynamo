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
SGLang disaggregated serving flow is 

Processor -> PrefillWorker -> DecodeWorker

This is different from how we've implemented the vLLM disaggregated flow. 

For now - the SGLangWorker will be responsible for aggreagted and prefill and we will
have a separate DecodeWorker. 
"""

import logging
from dynamo.sdk import service, depends, dynamo_context, dynamo_endpoint
from utils.sglang import parse_sglang_args
from dynamo.sdk.lib.service import LeaseConfig
import sglang as sgl
from utils.protocol import SGLangGenerateRequest, MyRequestOutput

logger = logging.getLogger(__name__)

@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
        "custom_lease": LeaseConfig(ttl=1),  # 1 second
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class SGLangWorker:

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        self.engine = sgl.Engine(server_args=self.engine_args)
        logger.warning("worker initialized")

    def shutdown_sglang_engine(self, signum, frame):
        logger.info("Shutting down SGLang engine")
        self.engine.shutdown()

    @dynamo_endpoint()
    async def generate(self, request: SGLangGenerateRequest):
        g = await self.engine.async_generate(
            input_ids=request.input_ids,
            sampling_params=request.sampling_params,
            stream=True
        )

        logger.warning("WE ARE HERE")
        
        # Make sure each result is properly serialized to be streamable
        async for result in g:
            # Convert the result to a serializable format if needed
            # Wrap the SGLang result in MyRequestOutput
            response = MyRequestOutput(text=result)
            logger.warning(response)
            yield response.model_dump_json()



        