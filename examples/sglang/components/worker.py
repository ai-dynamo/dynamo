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
import socket

from components.decode_worker import SGLangDecodeWorker
from sglang.srt.utils import get_ip
from utils.protocol import BootstrapInfo, MyRequestOutput, SGLangGenerateRequest
from utils.sglang import parse_sglang_args

import sglang as sgl
from dynamo.sdk import depends, dynamo_endpoint, service

logger = logging.getLogger(__name__)

"""
In SGLang, PD Disaggregation goes from Prefill -> Decode
"""


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1},
    workers=1,
)
class SGLangWorker:
    sglang_decode_worker = depends(SGLangDecodeWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_sglang_args(class_name, "")
        self.engine = sgl.Engine(server_args=self.engine_args)
        logger.info("SGLangWorker initialized")

    def shutdown_sglang_engine(self, signum, frame):
        logger.info("Shutting down SGLang engine")
        self.engine.shutdown()

    def _get_bootstrap_info(self):
        inner_tm = self.engine.tokenizer_manager
        bootstrap_port = inner_tm.server_args.disaggregation_bootstrap_port

        # multinode check
        if inner_tm.server_args.dist_init_addr:
            bootstrap_host = socket.gethostbyname(
                inner_tm.server_args.dist_init_addr.split(":")[0]
            )
        else:
            bootstrap_host = get_ip()

        return BootstrapInfo(host=bootstrap_host, port=bootstrap_port)

    @dynamo_endpoint()
    async def generate(self, request: SGLangGenerateRequest):
        g = await self.engine.async_generate(
            input_ids=request.input_ids,
            sampling_params=request.sampling_params,
            stream=True,
            bootstrap_host=request.bootstrap_host,
            bootstrap_port=request.bootstrap_port,
            bootstrap_room=request.bootstrap_room,
        )

        async for result in g:
            yield MyRequestOutput(text=result).model_dump_json()

    @dynamo_endpoint()
    async def get_url(self, request: dict):
        """
        The bootstrap server details are stored in each engines internal tokenizer manager state
        """
        yield self._get_bootstrap_info().model_dump_json()
