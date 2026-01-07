# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from typing import Any, Dict, Optional

import torch
from vllm.inputs.data import TokensPrompt
from vllm.v1.engine.async_llm import AsyncLLM

import dynamo.nixl_connect as connect
from dynamo.runtime import Client, Component, DistributedRuntime

from ..handlers import BaseWorkerHandler
from ..multimodal_utils import (
    ImageLoader,
    MyRequestOutput,
    construct_mm_data,
    is_qwen_vl_model,
    vLLMMultimodalRequest,
)

logger = logging.getLogger(__name__)


def _construct_qwen_decode_mm_data(
    image_grid_thw: list,
    hidden_size: int = 3584,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Construct multimodal data for Qwen VL models during decode phase.

    For Qwen2.5-VL and similar models, the decode worker needs image_grid_thw
    for MRoPE (multi-resolution rotary position embeddings) calculation,
    even though it doesn't process image embeddings (those are in KV cache).

    vLLM's Qwen parser requires BOTH 'image_embeds' and 'image_grid_thw' keys.
    We provide placeholder zeros for image_embeds to satisfy the parser.
    The actual visual features are already in the KV cache from prefill.

    Args:
        image_grid_thw: List of [temporal, height, width] grid dimensions per image
        hidden_size: Model hidden dimension (3584 for Qwen2.5-VL-7B)

    Returns:
        Dictionary with image data for vLLM, or None if no grid provided
    """
    if image_grid_thw is None or len(image_grid_thw) == 0:
        return None

    grid_thw_tensor = torch.tensor(image_grid_thw)

    # Calculate total visual tokens from grid dimensions
    # Each [t, h, w] entry represents one image's grid
    # Visual tokens = sum of (t * h * w) for each image
    total_visual_tokens = sum(int(t) * int(h) * int(w) for t, h, w in image_grid_thw)

    # Create placeholder embeddings with shape (num_visual_tokens, hidden_dim)
    # This satisfies vLLM's parser requirement for 'image_embeds' key
    # The zeros won't be used since KV cache already has the real visual features
    image_embeds = torch.zeros(
        (total_visual_tokens, hidden_size),
        dtype=torch.float16,
        device="cpu",
    )

    return {
        "image": {
            "image_embeds": image_embeds,
            "image_grid_thw": grid_thw_tensor,
        }
    }


class MultimodalDecodeWorkerHandler(BaseWorkerHandler):
    """Decode worker for disaggregated multimodal serving"""

    def __init__(
        self,
        runtime,
        component,
        engine_client,
        config,
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

        # For Qwen VL models, we need to pass image_grid_thw for MRoPE position calculation
        # even during decode (the embeddings are already in KV cache from prefill)
        # vLLM's Qwen parser requires both image_embeds and image_grid_thw keys
        multi_modal_data = None
        if is_qwen_vl_model(self.config.model) and request.image_grid_thw:
            # Get hidden size from model config, default to Qwen2.5-VL-7B value
            hidden_size = getattr(
                self.config.engine_args.create_model_config().hf_text_config,
                "hidden_size",
                3584,
            )
            multi_modal_data = _construct_qwen_decode_mm_data(
                request.image_grid_thw, hidden_size=hidden_size
            )
            logger.debug(
                f"Constructed Qwen decode mm_data with image_grid_thw: {request.image_grid_thw}, "
                f"hidden_size: {hidden_size}"
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
