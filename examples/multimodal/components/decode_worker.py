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
from typing import Optional

import connect
import torch
from components.disagg_router import PyDisaggregatedRouter
from components.encode_worker import VllmEncodeWorker
from components.prefill_worker import VllmPrefillWorker
from transformers import LlavaForConditionalGeneration
from utils.logging import check_required_workers
from utils.nixl import NixlMetadataStore
from utils.prefill_queue import PrefillQueue
from utils.protocol import (
    EncodeRequest,
    EncodeResponse,
    MyRequestOutput,
    vLLMMultimodalRequest,
)
from utils.vllm import parse_vllm_args
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)
from vllm.inputs.data import TokensPrompt
from vllm.remote_prefill import RemotePrefillParams, RemotePrefillRequest
from vllm.sampling_params import RequestOutputKind

from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service

logger = logging.getLogger(__name__)


@service(
    dynamo={
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class VllmDecodeWorker:
    # For disaggregated serving, we need to link the prefill worker to the vllm worker
    prefill_worker = depends(VllmPrefillWorker)
    # For aggregated serving, we need to link the encode worker to the vllm worker.
    encode_worker = depends(VllmEncodeWorker)

    def __init__(self):
        self.client = None
        self.min_workers = 1
        self.disaggregated_router: Optional[PyDisaggregatedRouter] = None
        class_name = self.__class__.__name__
        self.engine_args = parse_vllm_args(class_name, "")
        self.do_remote_prefill = self.engine_args.remote_prefill
        self.model_name = (
            self.engine_args.served_model_name
            if self.engine_args.served_model_name is not None
            else "vllm"
        )
        self._prefill_queue_nats_server = os.getenv(
            "NATS_SERVER", "nats://localhost:4222"
        )
        self._prefill_queue_stream_name = self.model_name
        logger.info(
            f"Prefill queue: {self._prefill_queue_nats_server}:{self._prefill_queue_stream_name}"
        )

        if self.engine_args.remote_prefill:
            if self.engine_args.enable_chunked_prefill is not False:
                logger.info("Chunked prefill is not supported yet, setting to False")
                self.engine_args.enable_chunked_prefill = False

            if self.engine_args.preemption_mode != "swap":
                logger.info("Preemption mode is not supported yet, setting to swap")
                self.engine_args.preemption_mode = "swap"

            if self.engine_args.pipeline_parallel_size != 1:
                logger.info("Pipeline parallel size is not supported yet, setting to 1")
                self.engine_args.pipeline_parallel_size = 1

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

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

        if self.engine_args.router == "kv":
            raise NotImplementedError(
                "Multimodal requests are not supported for kv router mode"
            )

        runtime = dynamo_context["runtime"]

        if self.do_remote_prefill:
            metadata = self.engine_client.nixl_metadata
            metadata_store = NixlMetadataStore("dynamo", runtime)
            await metadata_store.put(metadata.engine_id, metadata)

            if self.engine_args.conditional_disagg:
                self.disaggregated_router = PyDisaggregatedRouter(
                    runtime,
                    self.model_name,
                    max_local_prefill_length=self.engine_args.max_local_prefill_length,
                    max_prefill_queue_size=self.engine_args.max_prefill_queue_size,
                )
                await self.disaggregated_router.async_init()
            else:
                self.disaggregated_router = None

            model = LlavaForConditionalGeneration.from_pretrained(
                self.engine_args.model
            )
            vision_tower = model.vision_tower
            self.embedding_size = (
                vision_tower.vision_model.embeddings.position_embedding.num_embeddings
            )
        else:
            EMBEDDINGS_SHAPE = (1, 577, 4096)
            EMBEDDINGS_DTYPE = torch.float16
            EMBEDDINGS_DEVICE = "cuda"

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
            embeddings = torch.empty(
                EMBEDDINGS_SHAPE, dtype=EMBEDDINGS_DTYPE, device=EMBEDDINGS_DEVICE
            )
            descriptor = connect.Descriptor(embeddings)
            # Register the descriptor w/ NIXL (this is optional, if not done here the connect subsytem will take care of this automatically).
            descriptor.register_memory(self._connector)
            self._embeddings_descriptor = (embeddings, descriptor)

            await check_required_workers(self.encode_worker_client, self.min_workers)
            self.disaggregated_router = None

        logger.info("Initialization complete.")

    def shutdown_vllm_engine(self, signum, frame):
        """Shutdown the background loop"""
        logger.info(f"Received signal {signum}, shutting down")
        loop = asyncio.get_event_loop()
        try:
            self.engine_client.close()
            logger.info("Shutdown complete.")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            loop.stop()

    def get_remote_prefill_request_callback(self):
        async def callback(request: RemotePrefillRequest):
            async with PrefillQueue.get_instance(
                nats_server=self._prefill_queue_nats_server,
                stream_name=self._prefill_queue_stream_name,
            ) as prefill_queue:
                await prefill_queue.enqueue_prefill_request(request)

        return callback

    @dynamo_endpoint()
    async def generate(self, request: vLLMMultimodalRequest):
        request_id = request.request_id
        image_url = request.image_url
        logger.info(
            f"Received multimodal request {{ id: {request_id}, image_url: '{image_url}' }}."
        )
        embeddings = None
        if self.do_remote_prefill:
            logger.debug(
                f"Disaggregated: request {{ id: {request_id}, image_url: '{image_url}' }}"
                " prefill worker will populate the decode model's key-value cache ahead of time;"
                " no direct encode worker interaction required."
            )
            if self.disaggregated_router is not None:
                async with PrefillQueue.get_instance(
                    nats_server=self._prefill_queue_nats_server,
                    stream_name=self._prefill_queue_stream_name,
                ) as prefill_queue:
                    prefill_queue_size = await prefill_queue.get_queue_size()
                disagg_router_decision = await self.disaggregated_router.prefill_remote(
                    len(request.engine_prompt["prompt_token_ids"]),
                    request.prefix_hit_rate,
                    prefill_queue_size,
                )
            else:
                # always prefill remotely if no disaggregated router is provided
                disagg_router_decision = True

            if self.do_remote_prefill and disagg_router_decision:
                logger.debug(
                    f"Prefilling remotely for request {{ id: {request_id}, image_url: '{image_url}' }} with length {len(request.engine_prompt['prompt_token_ids'])}"
                )
                remote_prefill_params = RemotePrefillParams(
                    is_remote_prefill=True,
                    remote_prefill_request_callback=self.get_remote_prefill_request_callback(),
                    # Pass the image url as part of the RemotePrefillParams, which will be passed to the prefill worker via RemotePrefillRequest
                    multimodal_data_source={
                        "image_url": image_url,
                    },
                )
            else:
                remote_prefill_params = None
                logger.debug(
                    f"Prefilling locally for request {{ id: {request_id}, image_url: '{image_url}' }} with length {len(request.engine_prompt['prompt_token_ids'])}"
                )

            # The decode worker will pre-allocate the memory based on the prompt token length for the prefill worker to transfer the kv cache.
            # As a workaround, here we manually insert some placeholder dummy tokens based on the embedding size
            # so that decode worker can pre-allocate the memory with the correct size.
            # The structure of the prompt will be like: "\nUSER: <image> <dummy_tokens>\n<user_prompt>\nASSISTANT:".
            # Since the "<image>" token is included in the prompt, only need to insert (embedding_size - 1) dummy tokens after the image token.
            IMAGE_TOKEN_ID = 32000
            DUMMY_TOKEN_ID = 0
            # Find the index of the image token in the prompt token ids
            image_token_index = request.engine_prompt["prompt_token_ids"].index(
                IMAGE_TOKEN_ID
            )
            dummy_token_index = image_token_index + 1
            prompt_ids = (
                request.engine_prompt["prompt_token_ids"][:dummy_token_index]
                + [DUMMY_TOKEN_ID] * (self.embedding_size - 1)
                + request.engine_prompt["prompt_token_ids"][dummy_token_index:]
            )

        else:
            logger.debug(
                f"Aggregated: request {{ id: {request_id}, image_url: '{image_url}' }}"
                " no prefill worker available, embeddings directly from encode worker."
            )
            # Extract the pre-allocated, reusable image embeddings tensor and its descriptor.
            # Doing this avoids unnessesary memory de/registration with NIXL.
            embeddings, descriptor = self._embeddings_descriptor

            with self._connector.create_writable(descriptor) as writable:
                # Extract serialized metadata about the operation from the writable operation,
                # and use it to create a new EncodeRequest.
                encode_request = EncodeRequest(
                    request_id=request_id,
                    image_url=image_url,
                    serialized_request=writable.to_serialized(),
                )
                logger.debug(f"Encode request: {encode_request.model_dump_json()}")
                encode_generator = await self.encode_worker_client.round_robin(
                    encode_request.model_dump_json()
                )

                async for encode_response in encode_generator:
                    encode_output = EncodeResponse.model_validate_json(
                        encode_response.data()
                    )
                    logger.info(
                        f"Received response: {{ id: {encode_output.request_id} }}"
                    )

                # Wait for the write operation to complete.
                # This will block until the write operation is complete.
                # This await should be a no-op since we've already received a response from the encode worker.
                await writable.wait_for_completion()
                # At this point, the `embeddings` tensor is filled with the image embeddings from the remote encode worker.

            remote_prefill_params = None
            logger.info(
                f"Prefilling locally for request {{ id: {request_id}, image_url: '{image_url}' }} with length {len(request.engine_prompt['prompt_token_ids'])}"
            )
            prompt_ids = request.engine_prompt["prompt_token_ids"]

        # rust HTTP requires Delta streaming
        request.sampling_params.output_kind = RequestOutputKind.DELTA

        # When using aggregated serving, the encode worker will have provided the key-value cache updates via the prefill worker.
        # When using disaggregated serving, the encode worker will have provided the key-value cache updates via the encode worker.
        if embeddings is not None:
            logger.debug(
                "Aggregated: embedding data from encode worker provided via multi-modal data to decode model."
            )
            multi_modal_data = {"image": embeddings}
        else:
            logger.debug(
                "Disaggregated: no embedding data required as prefill will have provided key-value cache updates via encode worker."
            )
            multi_modal_data = None

        async for response in self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=prompt_ids,
                multi_modal_data=multi_modal_data,
            ),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
            remote_prefill_params=remote_prefill_params,
        ):
            logger.debug(
                f"Yeilding response {{ id: {response.request_id}, prompt: '{response.prompt}' }}"
            )
            yield MyRequestOutput(
                request_id=response.request_id,
                prompt=response.prompt,
                prompt_token_ids=response.prompt_token_ids,
                prompt_logprobs=response.prompt_logprobs,
                outputs=response.outputs,
                finished=response.finished,
            ).model_dump_json()
