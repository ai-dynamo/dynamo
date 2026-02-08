# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from collections import defaultdict
from typing import Any

import torch
from vllm.inputs.data import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, Component, DistributedRuntime

from ..handlers import BaseWorkerHandler
from ..multimodal_utils import ImageLoader, MyRequestOutput, vLLMMultimodalRequest
from ..multimodal_utils.model import construct_qwen_decode_mm_data, is_qwen_vl_model
from ..multimodal_utils.prefill_worker_utils import (
    accumulate_embeddings,
    load_embeddings,
)

logger = logging.getLogger(__name__)


class MultimodalDecodeWorkerHandler(BaseWorkerHandler):
    """Decode worker for disaggregated multimodal serving"""

    def __init__(
        self,
        runtime,
        component,
        engine_client,
        config,
        shutdown_event=None,
    ):
        # Get default_sampling_params from config
        default_sampling_params = (
            config.engine_args.create_model_config().get_diff_sampling_param()
        )

        # Call BaseWorkerHandler.__init__ with proper parameters
        super().__init__(
            runtime,
            component,
            engine_client,
            default_sampling_params,
            enable_multimodal=config.enable_multimodal,
            shutdown_event=shutdown_event,
        )

        self.config = config
        self.enable_disagg = config.is_prefill_worker

    async def async_init(self, runtime: DistributedRuntime):
        """Async initialization - connector needs async setup"""
        self._connector = connect.Connector()
        logger.info("Multimodal Decode Worker async initialization completed.")

    async def generate(self, request: vLLMMultimodalRequest, context):
        logger.debug(f"Got raw request: {request}")
        if not isinstance(request, vLLMMultimodalRequest):
            if isinstance(request, str):
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received decode request: {{ id: {request.request_id} }}.")

        # For Qwen VL models with mRoPE, we need to pass multi_modal_data containing
        # image_grid_thw for position embeddings calculation. The decode worker
        # receives the ORIGINAL unexpanded prompt (with placeholders), and vLLM
        # will expand it using the multi_modal_data, ensuring the block count
        # matches what prefill computed.
        #
        # We pass unique placeholder embeddings (seeded by request_id) since the
        # actual embeddings are already in the KV cache from prefill. The unique
        # values prevent incorrect prefix cache matches between different images.
        multi_modal_data = None
        if is_qwen_vl_model(self.config.model):
            image_grid_thw = getattr(request, "image_grid_thw", None)
            embeddings_shape = getattr(request, "embeddings_shape", None)
            if image_grid_thw is None or embeddings_shape is None:
                logger.warning(
                    "Missing Qwen VL decode fields (image_grid_thw/embeddings_shape); "
                    "skipping multi_modal_data construction."
                )
            else:
                multi_modal_data = construct_qwen_decode_mm_data(
                    image_grid_thw, embeddings_shape, request.request_id
                )

        gen = self.engine_client.generate(
            prompt=TokensPrompt(
                prompt_token_ids=request.engine_prompt["prompt_token_ids"],
                multi_modal_data=multi_modal_data,
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


class MultimodalPDWorkerHandler(BaseWorkerHandler):
    """Prefill/Decode or Prefill-only worker for multimodal serving"""

    def __init__(
        self,
        runtime,
        component: Component,
        engine_client: AsyncLLM,
        config,
        decode_worker_client: Client = None,
        shutdown_event=None,
    ):
        # Get default_sampling_params from config
        default_sampling_params = (
            config.engine_args.create_model_config().get_diff_sampling_param()
        )

        # Call BaseWorkerHandler.__init__ with proper parameters
        super().__init__(
            runtime,
            component,
            engine_client,
            default_sampling_params,
            enable_multimodal=config.enable_multimodal,
            shutdown_event=shutdown_event,
        )

        self.config = config
        self.decode_worker_client = decode_worker_client
        self.enable_disagg = config.is_prefill_worker

        # Initialize multimodal-specific components
        logger.info("Multimodal PD Worker startup started.")

        if "video" in self.config.model.lower():
            self.EMBEDDINGS_DTYPE = torch.uint8
        else:
            self.EMBEDDINGS_DTYPE = torch.float16

        self.EMBEDDINGS_DEVICE = "cpu"

        # Create and initialize a dynamo connector for this worker.
        # We'll need this to move data between this worker and remote workers efficiently.
        # Note: This is synchronous initialization, async initialization happens in async_init
        self._connector = None  # Will be initialized in async_init
        self.image_loader = ImageLoader()

        logger.info("Multimodal PD Worker has been initialized")

    async def async_init(self, runtime: DistributedRuntime):
        """Async initialization for connector that requires async setup"""
        # Initialize the connector asynchronously
        self._connector = connect.Connector()
        logger.info("Multimodal PD Worker async initialization completed.")

    async def generate(self, request: vLLMMultimodalRequest, context):
        logger.debug(f"Got raw request: {request}")
        if type(request) is not vLLMMultimodalRequest:
            if type(request) is str:
                request = vLLMMultimodalRequest.model_validate_json(request)
            else:
                request = vLLMMultimodalRequest.model_validate(request)
        logger.debug(f"Received PD request: {{ id: {request.request_id} }}.")

        multi_modal_data: dict[str, Any] = defaultdict(list)
        for mi in request.multimodal_inputs:
            if mi.multimodal_input.image_url:
                # PIL image path — used by both EC consumer mode
                # (vLLM looks up cached embeddings via mm_hash) and
                # non-disaggregated mode (vLLM encodes inline).
                multi_modal_data["image"].append(
                    await self.image_loader.load_image(mi.multimodal_input.image_url)
                )
            else:
                # Pre-computed embeddings via NIXL RDMA or local safetensors
                embeddings = await load_embeddings(
                    mi,
                    self.EMBEDDINGS_DTYPE,
                    self.EMBEDDINGS_DEVICE,
                    self._connector,
                )
                accumulate_embeddings(
                    multi_modal_data,
                    self.config.model,
                    self.EMBEDDINGS_DTYPE,
                    embeddings,
                    mi.image_grid_thw,
                )

        # For Qwen VL (mRoPE), capture the accumulated image grid + embedding shape
        # from the constructed multimodal data so decode can reconstruct its
        # multi_modal_data consistently for multiple images.
        if is_qwen_vl_model(self.config.model) and isinstance(
            multi_modal_data.get("image"), dict
        ):
            image_data = multi_modal_data["image"]
            image_grid_thw = image_data.get("image_grid_thw")
            image_embeds = image_data.get("image_embeds")
            if image_grid_thw is not None:
                request.image_grid_thw = (
                    image_grid_thw.tolist()
                    if isinstance(image_grid_thw, torch.Tensor)
                    else image_grid_thw
                )
            if image_embeds is not None:
                request.embeddings_shape = list(image_embeds.shape)

        # Remove the image features from the request as they are not required
        # Use empty list instead of None to satisfy Pydantic validation on decode worker after vllm upgrade
        request.multimodal_inputs = []

        logger.info(f"Prepared multimodal data size: {len(multi_modal_data['image'])}")
        logger.info(f"{multi_modal_data}")

        # Deepcopy the request to avoid modifying the original
        # when we adjust sampling params for prefill

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
                # For Qwen VL models with mRoPE: Keep the ORIGINAL unexpanded prompt.
                # The decode worker will pass multi_modal_data which causes vLLM to
                # expand the prompt identically to prefill, ensuring block counts match.
                #
                # For other models: Use the expanded prompt from prefill response.
                # These models don't pass multi_modal_data in decode, so they need
                # the already-expanded prompt to match the KV cache layout.
                if not is_qwen_vl_model(self.config.model):
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
                logger.debug(
                    f"length of expanded prompt ids: {len(response.prompt_token_ids)}"
                )
                # logger.info(f"Response outputs: {response.outputs}")
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
