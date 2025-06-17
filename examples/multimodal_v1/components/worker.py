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
import connect
import torch
from utils.model import get_vision_embeddings_info
from utils.args import parse_vllm_args
from utils.protocol import MyRequestOutput, vLLMGenerateRequest, vLLMMultimodalRequest
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt

from dynamo.sdk import async_on_start, endpoint, service


logger = logging.getLogger(__name__)


class VllmBaseWorker:
    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

        self.set_side_channel_host_and_port()

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

    @endpoint()
    async def generate(self, request: vLLMMultimodalRequest):
        logger.debug(f"Received generate request: {{ id: {request.request_id} }}.")

        embeddings, descriptor = self._embeddings_descriptor
        read_op = await self._connector.begin_read(
            request.serialized_request,
            descriptor
        )
        await read_op.wait_for_completion()
        logger.debug(f"========== in decode worker, image features: {embeddings}")

        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=request.engine_prompt["prompt_token_ids"],
                multi_modal_data={"image": embeddings}
            ),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
        )

        async for response in gen:
            # logger.debug(f"Response kv_transfer_params: {response.kv_transfer_params}")
            # logger.debug(f"Response outputs: {response.outputs}")
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

    def set_side_channel_host_and_port(
        self, hostname: Optional[str] = None, port: Optional[int] = None
    ):
        """vLLM V1 NixlConnector creates a side channel to exchange metadata with other NIXL connectors.
        This sets the port number for the side channel.
        """
        if hostname is None:
            hostname = socket.gethostname()
        if port is None:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("", 0))  # Bind to a free port provided by the host.
                port = s.getsockname()[1]  # Get the port number assigned.
        logger.debug("Setting VLLM_NIXL_SIDE_CHANNEL_HOST to %s", hostname)
        os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] = hostname
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

        EMBEDDINGS_DTYPE = torch.float16
        EMBEDDINGS_DEVICE = "cpu"
        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        await self._connector.initialize()

        # embeddings_shape, self.embeddings_dtype = get_vision_embeddings_info(
        #     self.engine_args.model, self.engine_args.num_patches
        # )
        embeddings_shape = (1, 577, 4096)
        logger.debug(f"Embeddings shape: {embeddings_shape}")
        self.embedding_size = embeddings_shape[1]

        embeddings = torch.empty(
            embeddings_shape, dtype=EMBEDDINGS_DTYPE, device=EMBEDDINGS_DEVICE
        )

        descriptor = connect.Descriptor(embeddings)

        # Register the descriptor w/ NIXL (this is optional, if not done here the connect subsytem will take care of this automatically).
        # descriptor.register_memory(self._connector)
        self._embeddings_descriptor = (embeddings, descriptor)

        logger.info("VllmDecodeWorker has been initialized")
