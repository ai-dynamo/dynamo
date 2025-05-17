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


import asyncio
import logging
import os
import signal
import socket
from typing import Optional

from utils.args import parse_vllm_args
from utils.protocol import MyRequestOutput, vLLMGenerateRequest
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)

from dynamo.sdk import async_on_start, dynamo_endpoint, service

logger = logging.getLogger(__name__)


class VllmBaseWorker:
    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

        self.set_side_channel_port()

    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

        logger.info("VllmWorker has been initialized")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("VllmWorker shutdown complete")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    @dynamo_endpoint()
    async def generate(self, request: vLLMGenerateRequest):
        gen = self.engine_client.generate(
            prompt=request.prompt,
            sampling_params=request.sampling_params,
            request_id=request.request_id,
        )

        async for response in gen:
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
                metrics=response.metrics,
                kv_transfer_params=response.kv_transfer_params,
            ).model_dump_json()

    def set_side_channel_port(self, port: Optional[int] = None):
        """vLLM V1 NixlConnector creates a side channel to exchange metadata with other NIXL connectors.
        This sets the port number for the side channel.
        """
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))  # Bind to a free port provided by the host.
                port = s.getsockname()[1]  # Get the port number assigned.
        logger.debug("Setting VLLM_NIXL_SIDE_CHANNEL_PORT to %s", port)
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(port)


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmPrefillWorker(VllmBaseWorker):
    @async_on_start
    async def async_init(self):
        await super().async_init()
        logger.info("VllmPrefillWorker has been initialized")


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmDecodeWorker(VllmBaseWorker):
    @async_on_start
    async def async_init(self):
        await super().async_init()
        logger.info("VllmDecodeWorker has been initialized")
