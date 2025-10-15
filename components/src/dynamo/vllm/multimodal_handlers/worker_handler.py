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

import copy
import logging
import os

import torch
from vllm.inputs.data import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

import dynamo.nixl_connect as connect
from dynamo.llm import ZmqKvEventPublisher, ZmqKvEventPublisherConfig
from dynamo.runtime import Client, Component, DistributedRuntime, Endpoint

from ..multimodal_utils import (
    ImageLoader,
    MyRequestOutput,
    construct_mm_data,
    vLLMMultimodalRequest,
)
from ..publisher import StatLoggerFactory

logger = logging.getLogger(__name__)


class MultimodalBaseWorker:
    """Base class for multimodal workers"""

    def __init__(
        self,
        component: Component,
        endpoint: Endpoint,
        engine_client: AsyncLLM,
        config,
    ):
        self.component = component
        self.endpoint = endpoint
        self.engine_client = engine_client
        self.config = config
        self.enable_disagg = config.is_prefill_worker
        self.kv_publisher = None

        # Set up stats logger
        self.stats_logger = StatLoggerFactory(
            component,
            config.engine_args.data_parallel_rank or 0,
            metrics_labels=[("model", config.model)],
        )

        # TODO Hack to get data, move this to registering in ETCD
        vllm_config = config.engine_args.create_engine_config()
        self.stats_logger.set_num_gpu_blocks_all(
            vllm_config.cache_config.num_gpu_blocks
        )
        self.stats_logger.set_request_total_slots_all(
            vllm_config.scheduler_config.max_num_seqs
        )
        self.stats_logger.init_publish()

    async def async_init(self, runtime: DistributedRuntime):
        pass

    def cleanup(self):
        """Override in subclasses if cleanup is needed."""
        pass

    async def clear_kv_blocks(self, request=None):
        try:
            await self.engine_client.reset_prefix_cache()
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}


class MultimodalDecodeWorkerHandler(MultimodalBaseWorker):
    """Decode worker for disaggregated multimodal serving"""

    async def generate(self, request: vLLMMultimodalRequest):
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received decode request: {{ id: {request.request_id} }}.")

        # Decode worker doesn't process embeddings, so we pass None or empty tensor
        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=request.engine_prompt["prompt_token_ids"],
            ),
            sampling_params=request.sampling_params,
            request_id=request.request_id,
        )

        async for response in gen:
            logger.debug(f"Response kv_transfer_params: {response.kv_transfer_params}")
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


class MultimodalPDWorkerHandler(MultimodalBaseWorker):
    """Prefill/Decode or Prefill-only worker for multimodal serving"""

    def __init__(
        self,
        component: Component,
        endpoint: Endpoint,
        engine_client: AsyncLLM,
        config,
        decode_worker_client: Client = None,
    ):
        super().__init__(component, endpoint, engine_client, config)
        self.decode_worker_client = decode_worker_client
        self._connector = None
        self.image_loader = None

    async def async_init(self, runtime: DistributedRuntime):
        logger.info("Multimodal PD Worker startup started.")

        if "video" in self.config.model.lower():
            self.EMBEDDINGS_DTYPE = torch.uint8
        else:
            self.EMBEDDINGS_DTYPE = torch.float16

        self.EMBEDDINGS_DEVICE = "cpu"

        # Create and initialize a dynamo connector for this worker.
        # We'll needs this to move data between this worker and remote workers efficiently.
        self._connector = connect.Connector()
        await self._connector.initialize()

        self.image_loader = ImageLoader()

        logger.info("Multimodal PD Worker has been initialized")

    async def generate(self, request: vLLMMultimodalRequest):
        logger.debug(f"Got raw request: {request}")
        if type(request) is not vLLMMultimodalRequest:
            if type(request) is str:
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received PD request: {{ id: {request.request_id} }}.")

        if (
            request.multimodal_input.image_url is None
            and request.multimodal_input.video_url is None
        ):
            # Process embeddings using the connector
            # Create a descriptor based on the embedding shape.
            embeddings = torch.empty(
                request.embeddings_shape,
                dtype=self.EMBEDDINGS_DTYPE,
                device=self.EMBEDDINGS_DEVICE,
            )
            descriptor = connect.Descriptor(embeddings)

            if descriptor is None:
                raise RuntimeError(
                    "Descriptor is None in PD worker - cannot process embeddings"
                )

            read_op = await self._connector.begin_read(
                request.serialized_request, descriptor
            )
            await read_op.wait_for_completion()
            if "video" in self.config.model.lower():
                video_numpy = embeddings.numpy()
                multi_modal_data = construct_mm_data(
                    self.config.model,
                    self.EMBEDDINGS_DTYPE,
                    video_numpy=video_numpy,
                )
            else:
                multi_modal_data = construct_mm_data(
                    self.config.model,
                    self.EMBEDDINGS_DTYPE,
                    image_embeds=embeddings,
                    image_grid_thw=request.image_grid_thw,
                )
        else:
            # Use PIL image instead of image embeddings
            multi_modal_data = {
                "image": await self.image_loader.load_image(
                    request.multimodal_input.image_url
                )
            }

        # Remove the image features from the request as they are not required
        request.multimodal_input.image_url = None
        request.multimodal_input.video_url = None
        request.serialized_request = None

        pd_request = copy.deepcopy(request)
        # Do prefill and remote decode if enable_disagg is true
        if self.enable_disagg and self.decode_worker_client:
            extra_args = pd_request.sampling_params.extra_args or {}
            extra_args["kv_transfer_params"] = {
                "do_remote_decode": True,
            }
            pd_request.sampling_params.extra_args = extra_args
            pd_request.sampling_params.max_tokens = 1
            pd_request.sampling_params.min_tokens = 1

            logger.debug("Prefill request: %s", pd_request)

        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=pd_request.engine_prompt["prompt_token_ids"],
                multi_modal_data=multi_modal_data,
            ),
            sampling_params=pd_request.sampling_params,
            request_id=pd_request.request_id,
        )

        if self.enable_disagg and self.decode_worker_client:
            decode_request = copy.deepcopy(request)
            async for prefill_response in gen:
                # Update the prompt token id in the decode request to the one
                # in response, which has image templated filled in. So that
                # the decode worker will fetch correct amount of KV blocks.
                decode_request.engine_prompt[
                    "prompt_token_ids"
                ] = prefill_response.prompt_token_ids
                logger.debug(
                    f"Prefill response kv_transfer_params: {prefill_response.kv_transfer_params}"
                )
                extra_args = decode_request.sampling_params.extra_args or {}
                extra_args["kv_transfer_params"] = prefill_response.kv_transfer_params
                extra_args.pop("serialized_request", None)
                decode_request.sampling_params.extra_args = extra_args
                logger.debug("Decode request: %s", decode_request)
                async for (
                    decode_response
                ) in await self.decode_worker_client.round_robin(
                    decode_request.model_dump_json()
                ):
                    output = MyRequestOutput.model_validate_json(decode_response.data())
                    yield MyRequestOutput(
                        request_id=output.request_id,
                        prompt=output.prompt,
                        prompt_token_ids=output.prompt_token_ids,
                        prompt_logprobs=output.prompt_logprobs,
                        outputs=output.outputs,
                        finished=output.finished,
                        metrics=output.metrics,
                        kv_transfer_params=output.kv_transfer_params,
                    ).model_dump_json()

        else:
            async for response in gen:
                logger.debug(
                    f"Response kv_transfer_params: {response.kv_transfer_params}"
                )
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

