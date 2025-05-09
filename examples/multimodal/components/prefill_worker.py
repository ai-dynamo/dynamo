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
import connect
import logging
import os
import signal
import sys
import torch

from components.encode_worker import VllmEncodeWorker
from pydantic import BaseModel
from utils.logging import check_required_workers
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import EncodeRequest, EncodeResponse
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)

# Constants for the shape and dtype of the embeddings tensor.
EMBEDDINGS_SHAPE = (1, 577, 4096)
EMBEDDINGS_DTYPE = torch.float16
EMBEDDINGS_DEVICE = "cuda"


class RequestType(BaseModel):
    text: str


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmPrefillWorker:
    encode_worker = depends(VllmEncodeWorker)

    def __init__(self):
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self._loaded_metadata = set()
        self.initialized = False
        self.min_workers = 1
        if self.engine_args.enable_chunked_prefill is not False:
            logger.info("Chunked prefill is not supported yet, setting to False")
            self.engine_args.enable_chunked_prefill = False

        if self.engine_args.pipeline_parallel_size != 1:
            logger.info("Pipeline parallel size is not supported yet, setting to 1")
            self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.disable_async_output_proc is not True:
            logger.info("Async output processing is not supported yet, setting to True")
            self.engine_args.disable_async_output_proc = True

        if self.engine_args.enforce_eager is not True:
            logger.info("Prefill must be done eagerly, setting to True")
            self.engine_args.enforce_eager = True

        if self.engine_args.enable_prefix_caching is not False:
            logger.info(
                "Prefix caching is not supported yet in prefill worker, setting to False"
            )
            self.engine_args.enable_prefix_caching = False

        signal.signal(signal.SIGTERM, self.shutdown_vllm_engine)
        signal.signal(signal.SIGINT, self.shutdown_vllm_engine)

    @async_on_start
    async def async_init(self):
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
        else:
            raise RuntimeError("Failed to initialize engine client")
        runtime = dynamo_context["runtime"]

        enc_comp_ns, enc_comp_name = VllmEncodeWorker.dynamo_address()  # type: ignore
        self.encode_worker_client = (
            await runtime.namespace(enc_comp_ns)
            .component(enc_comp_name)
            .endpoint("encode")
            .client()
        )

        self._connector = connect.Connector(runtime=runtime, namespace=enc_comp_ns)
        await self._connector.initialize()

        # Create a longer-lived buffer for receiving the image embeddings.
        embeddings = torch.empty(EMBEDDINGS_SHAPE, dtype=EMBEDDINGS_DTYPE, device=EMBEDDINGS_DEVICE)
        descriptor = connect.Descriptor(embeddings)
        # Register the descriptor w/ NIXL (this is optional, if not done here the connect subsytem will take care of this automatically).
        descriptor.register_memory(self._connector)
        self._embeddings_descriptor = (embeddings, descriptor)

        await check_required_workers(self.encode_worker_client, self.min_workers)

        metadata = self.engine_client.nixl_metadata
        self._metadata_store = NixlMetadataStore("dynamo", runtime)
        await self._metadata_store.put(metadata.engine_id, metadata)
        task = asyncio.create_task(self.prefill_queue_handler())

        def prefill_queue_handler_cb(fut):
            try:
                fut.result()
                logger.info("prefill queue handler exited successfully")
            except Exception as e:
                logger.error(f"[ERROR] prefill queue handler failed: {e!r}")
                sys.exit(1)

        task.add_done_callback(prefill_queue_handler_cb)
        logger.info("Initialization complete.")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Shutdown started, signal {signum} received.")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()
        logger.info("Shutdown complete.")

    async def prefill_queue_handler(self):
        logger.info("Prefill queue handler entered")
        prefill_queue_nats_server = os.getenv("NATS_SERVER", "nats://localhost:4222")
        prefill_queue_stream_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )
        logger.info(
            f"Prefill queue: {prefill_queue_nats_server}:{prefill_queue_stream_name}"
        )
        self.initialized = True
        # TODO: integrate prefill_queue to a dynamo endpoint
        async with PrefillQueue.get_instance(
            nats_server=prefill_queue_nats_server,
            stream_name=prefill_queue_stream_name,
        ) as prefill_queue:
            logger.info("prefill queue handler started")
            while True:
                # TODO: this might add a small overhead to pull prefill from nats
                # need to test and check how much overhead it is
                prefill_request = await prefill_queue.dequeue_prefill_request()
                if prefill_request is not None:
                    logger.info(
                        f"Dequeued prefill request: {prefill_request.request_id}"
                    )
                    async for _ in self.generate(prefill_request):
                        pass

    async def generate(self, request: RemotePrefillRequest):
        if request.multimodal_data_source["image_url"] is None:
            raise ValueError("No image url provided for prefill request")

        request_id = request.request_id
        engine_id = request.engine_id
        image_url = request.multimodal_data_source["image_url"]

        logger.info(f"Received prefill request {{ id: {request_id}, engine_id: {engine_id}, image_url: '{image_url}' }}.")

        # Extract the pre-allocated, reusable image embeddings tensor and its descriptor.
        # Doing this avoids unnessesary memory de/registration with NIXL.
        embeddings, descriptor = self._embeddings_descriptor

        # Create a new writable operation from the descriptor.
        with self._connector.create_writable(descriptor) as writable:
            # Extract serialized metadata about the operation from the writable operation,
            # and use it to create a new EncodeRequest.
            encode_generator = await self.encode_worker_client.round_robin(
                EncodeRequest(
                    request_id=request_id,
                    image_url=image_url,
                    serialized_request=writable.to_serialized(),
                ).model_dump_json()
            )
            async for encode_response in encode_generator:
                encode_output = EncodeResponse.model_validate_json(encode_response.data())

            # Wait for the write operation to complete.
            # This will block until the write operation is complete.
            # This await should be a no-op since we've already received a response from the encode worker.
            await writable.wait_for_completion()
            # At this point, the `embeddings` tensor is filled with the image embeddings from the remote encode worker.

            sampling_params = request.sampling_params
            sampling_params.max_tokens = 1
            sampling_params.min_tokens = 1

            remote_prefill_params = RemotePrefillParams(
                is_remote_decode=True,
                decode_block_ids=request.block_ids,
                decode_engine_id=engine_id,
                decode_computed_block_ids=request.computed_block_ids,
            )

            # TODO check if metadata has changed
            # and reload - currently only loading once
            if engine_id not in self._loaded_metadata:
                remote_metadata = await self._metadata_store.get(request.engine_id)
                await self.engine_client.add_remote_nixl_metadata(remote_metadata)
                logger.info(
                    f"Loaded nixl metadata from engine {engine_id} into "
                    f"engine {self.engine_client.nixl_metadata.engine_id}"
                )
                self._loaded_metadata.add(engine_id)

            # To make sure the decode worker can pre-allocate the memory with the correct size for the prefill worker to transfer the kv cache,
            # some placeholder dummy tokens were inserted based on the embedding size in the worker.py.
            # The structure of the prompt is "\nUSER: <image> <dummy_tokens>\n<user_prompt>\nASSISTANT:", need to remove the dummy tokens after the image token.
            IMAGE_TOKEN_ID = 32000
            embedding_size = embeddings.shape[1]
            padding_size = embedding_size - 1
            image_token_index = request.prompt_token_ids.index(IMAGE_TOKEN_ID)
            dummy_token_index = image_token_index + 1
            prompt_token_ids = (
                request.prompt_token_ids[:dummy_token_index]
                + request.prompt_token_ids[dummy_token_index + padding_size :]
            )

            async for _ in self.engine_client.generate(
                request_id=request_id,
                prompt=TokensPrompt(
                    prompt_token_ids=prompt_token_ids,
                    multi_modal_data={"image": embeddings},
                ),
                sampling_params=sampling_params,
                remote_prefill_params=remote_prefill_params,
            ):
                yield

    @dynamo_endpoint()
    async def mock(self, req: RequestType):
        yield f"mock_response: {req}"
