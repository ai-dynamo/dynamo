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
import json
import logging
import os
import socket
import signal
from typing import Optional

from utils.args import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
from vllm.entrypoints.openai.protocol import CompletionRequest

from dynamo.sdk import async_on_start, dynamo_endpoint, service

logger = logging.getLogger(__name__)


class VllmBaseWorker:

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

        self.set_side_channel_port()

        print(f"self.engine_args: {self.engine_args}")

    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")

        model_config = self.engine_args.create_model_config()
        model_paths = [BaseModelPath(name, name) for name in self.engine_args.served_model_name]

        print(f"model_path: {model_paths}, type: {type(model_paths)}")

        oai_serving_models = OpenAIServingModels(
            self.engine_client,
            model_config,
            model_paths,
        )

        self.openai_serving_completions = OpenAIServingCompletion(
            self.engine_client,
            model_config,
            oai_serving_models,
            request_logger=None,
        )

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
    async def completions(self, request: CompletionRequest):

        logger.info(f"VllmWorker received completion request: {request}")

        response = await self.openai_serving_completions.create_completion(request)

        if request.stream:
            logger.info(f"VllmWorker streaming response")
            async for chunk in response:
                if chunk.startswith("data: [DONE]"):
                    break
                response = json.loads(chunk.lstrip("data: "))
                yield response

        else:
            logger.info(f"VllmWorker response: {response}, type: {type(response)}")
            yield response.model_dump()


    def set_side_channel_port(self, port: Optional[int] = None):
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', 0))  # Bind to a free port provided by the host.
                port = s.getsockname()[1]  # Get the port number assigned.
                print(f"Unused port: {port}")
        logger.info(f"Setting VLLM_NIXL_SIDE_CHANNEL_PORT to {port}")
        os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(port)



@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmWorker(VllmBaseWorker):

    @async_on_start
    async def async_init(self):
        await super().async_init()
        logger.info("VllmWorker has been initialized")


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